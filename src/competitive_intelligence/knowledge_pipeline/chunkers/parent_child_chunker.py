"""Hierarchical parent-child chunker implementation for Knowledge Pipeline subsystem.

Generates a two-level hierarchy of larger parent context chunks and smaller
child retrieval units using LlamaIndex HierarchicalNodeParser internally.
"""

from typing import Any
import uuid

from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import Document as LlamaDocument

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseDocumentChunker,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkMetadata,
    ChunkingParameters,
    ChunkingStrategy,
    DocumentChunk,
    DocumentVersion,
)
from competitive_intelligence.knowledge_pipeline.normalizer import compute_content_hash
from competitive_intelligence.source_acquisition.models import Source

DEFAULT_PARENT_CHUNK_SIZE = 2000
DEFAULT_CHILD_CHUNK_SIZE = 400
DEFAULT_PARENT_CHILD_OVERLAP = 50


def derive_parent_child_chunk_id(
    version_id: str,
    level: int,
    index: int,
    parent_chunk_id: str | None,
    parent_chunk_size: int,
    child_chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Derive deterministic chunk UUID for hierarchical chunking using UUID v5."""
    parent_token = parent_chunk_id or "root"
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"urn:ci:chunk:hierarchical:{version_id}:lvl{level}:idx{index}:p{parent_token}:{parent_chunk_size}:{child_chunk_size}:{chunk_overlap}",
        )
    )


class ParentChildChunker(BaseDocumentChunker):
    """Hierarchical chunker creating two-level parent-child chunk structures.

    Hierarchy:
    - Parent chunks: chunk_level = 0, parent_chunk_id = None (larger context blocks).
    - Child chunks: chunk_level = 1, parent_chunk_id = parent's chunk_id (smaller retrieval blocks).

    Invariants:
    - Every child chunk references exactly one valid parent chunk.
    - Every parent chunk has at least one associated child chunk.
    - Preserves sequential document order and full provenance.
    - Output is purely DocumentChunk domain models without LlamaIndex node types.
    """

    def __init__(
        self,
        default_parent_chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
        default_child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
        default_chunk_overlap: int = DEFAULT_PARENT_CHILD_OVERLAP,
    ) -> None:
        self.default_parent_chunk_size = default_parent_chunk_size
        self.default_child_chunk_size = default_child_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    def chunk(
        self,
        version: DocumentVersion,
        parameters: ChunkingParameters | None = None,
        source: Source | None = None,
    ) -> list[DocumentChunk]:
        """Partition DocumentVersion content into a hierarchical parent-child structure.

        Args:
            version: The DocumentVersion to chunk.
            parameters: Optional chunking parameters specifying parent/child sizes and overlap.
            source: Optional Source domain model for enriched provenance metadata.

        Returns:
            A list of DocumentChunk instances containing parents (level 0) and children (level 1).

        Raises:
            ValueError: If version.content is empty or whitespace-only.
        """
        if not version.content or not version.content.strip():
            raise ValueError("Cannot chunk DocumentVersion with empty or whitespace-only content")

        parent_chunk_size = (
            parameters.parent_chunk_size
            if parameters is not None and parameters.parent_chunk_size is not None
            else self.default_parent_chunk_size
        )
        child_chunk_size = (
            parameters.child_chunk_size
            if parameters is not None and parameters.child_chunk_size is not None
            else self.default_child_chunk_size
        )
        overlap_val = (
            parameters.custom_params.get("parent_child_overlap")
            if parameters is not None and "parent_child_overlap" in parameters.custom_params
            else (
                parameters.chunk_overlap
                if parameters is not None and parameters.chunk_overlap is not None
                else self.default_chunk_overlap
            )
        )
        chunk_overlap: int = int(overlap_val) if overlap_val is not None else self.default_chunk_overlap

        # 1. Use LlamaIndex HierarchicalNodeParser with 2 levels [parent, child]
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[parent_chunk_size, child_chunk_size],
            chunk_overlap=chunk_overlap,
        )

        llama_doc = LlamaDocument(text=version.content)
        nodes = parser.get_nodes_from_documents([llama_doc])

        if not nodes:
            # Fallback for single node if parser returns empty
            nodes = [llama_doc]

        # Separate parent nodes (level 0) and child nodes (level 1)
        parent_nodes = [n for n in nodes if getattr(n, "parent_node", None) is None]
        child_nodes = [n for n in nodes if getattr(n, "parent_node", None) is not None]

        # Edge case: If no children were produced (e.g. single node fallback), make the single node both parent and child
        if parent_nodes and not child_nodes:
            child_nodes = parent_nodes

        # Map LlamaIndex node_id to our deterministic chunk_id
        node_id_to_chunk_id: dict[str, str] = {}
        all_chunks: list[DocumentChunk] = []

        total_nodes = len(parent_nodes) + len(child_nodes)
        global_chunk_index = 0
        search_start = 0

        # 2. Build parent DocumentChunks (Level 0)
        parent_chunks: list[DocumentChunk] = []
        for p_idx, p_node in enumerate(parent_nodes):
            chunk_id = derive_parent_child_chunk_id(
                version_id=version.version_id,
                level=0,
                index=p_idx,
                parent_chunk_id=None,
                parent_chunk_size=parent_chunk_size,
                child_chunk_size=child_chunk_size,
                chunk_overlap=chunk_overlap,
            )
            node_id_to_chunk_id[p_node.node_id] = chunk_id

            node_text = p_node.get_content()
            start_char = version.content.find(node_text, search_start)
            if start_char == -1:
                start_char = version.content.find(node_text)
                if start_char == -1:
                    start_char = 0
            end_char = start_char + len(node_text)
            search_start = max(0, start_char)

            metadata = self._build_chunk_metadata(
                version=version,
                source=source,
                chunk_index=global_chunk_index,
                total_chunks=total_nodes,
                start_char=start_char,
                end_char=end_char,
                chunk_level=0,
                parent_chunk_id=None,
                parent_chunk_size=parent_chunk_size,
                child_chunk_size=child_chunk_size,
                chunk_overlap=chunk_overlap,
            )

            parent_chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=version.document_id,
                version_id=version.version_id,
                content=node_text,
                chunk_index=global_chunk_index,
                metadata=metadata,
                content_hash=compute_content_hash(node_text),
                parent_chunk_id=None,
                chunk_level=0,
                created_at=version.acquired_at,
            )
            parent_chunks.append(parent_chunk)
            all_chunks.append(parent_chunk)
            global_chunk_index += 1

        # 3. Build child DocumentChunks (Level 1)
        search_start = 0
        for c_idx, c_node in enumerate(child_nodes):
            # Resolve parent chunk_id
            parent_rel = getattr(c_node, "parent_node", None)
            if parent_rel is not None and parent_rel.node_id in node_id_to_chunk_id:
                parent_chunk_id: str | None = node_id_to_chunk_id[parent_rel.node_id]
            elif parent_chunks:
                # Fallback to first parent if relationship pointer missing
                parent_chunk_id = parent_chunks[0].chunk_id
            else:
                parent_chunk_id = None

            child_chunk_id = derive_parent_child_chunk_id(
                version_id=version.version_id,
                level=1,
                index=c_idx,
                parent_chunk_id=parent_chunk_id,
                parent_chunk_size=parent_chunk_size,
                child_chunk_size=child_chunk_size,
                chunk_overlap=chunk_overlap,
            )

            node_text = c_node.get_content()
            start_char = version.content.find(node_text, search_start)
            if start_char == -1:
                start_char = version.content.find(node_text)
                if start_char == -1:
                    start_char = 0
            end_char = start_char + len(node_text)
            search_start = max(0, start_char)

            metadata = self._build_chunk_metadata(
                version=version,
                source=source,
                chunk_index=global_chunk_index,
                total_chunks=total_nodes,
                start_char=start_char,
                end_char=end_char,
                chunk_level=1,
                parent_chunk_id=parent_chunk_id,
                parent_chunk_size=parent_chunk_size,
                child_chunk_size=child_chunk_size,
                chunk_overlap=chunk_overlap,
            )

            child_chunk = DocumentChunk(
                chunk_id=child_chunk_id,
                document_id=version.document_id,
                version_id=version.version_id,
                content=node_text,
                chunk_index=global_chunk_index,
                metadata=metadata,
                content_hash=compute_content_hash(node_text),
                parent_chunk_id=parent_chunk_id,
                chunk_level=1,
                created_at=version.acquired_at,
            )
            all_chunks.append(child_chunk)
            global_chunk_index += 1

        return all_chunks

    def _build_chunk_metadata(
        self,
        version: DocumentVersion,
        source: Source | None,
        chunk_index: int,
        total_chunks: int,
        start_char: int,
        end_char: int,
        chunk_level: int,
        parent_chunk_id: str | None,
        parent_chunk_size: int,
        child_chunk_size: int,
        chunk_overlap: int,
    ) -> ChunkMetadata:
        """Construct provenance ChunkMetadata for a hierarchical chunk."""
        if source is not None:
            source_id = source.identity.source_id
            company_id = source.identity.company_id
            workspace_id = source.identity.workspace_id
            source_url = source.location.url_or_identifier
            source_category = source.classification.category.value
        else:
            source_id = str(version.metadata.get("source_id", f"src_{version.document_id}"))
            company_id = str(version.metadata.get("company_id", "unknown_company"))
            workspace_id = str(version.metadata.get("workspace_id", "default_workspace"))
            source_url = str(version.metadata.get("source_url", "https://example.com/document"))
            source_category = str(version.metadata.get("source_category", "general"))

        custom_tags: dict[str, Any] = dict(version.metadata)
        custom_tags["strategy"] = ChunkingStrategy.PARENT_CHILD.value
        custom_tags["chunk_level"] = chunk_level
        if parent_chunk_id is not None:
            custom_tags["parent_chunk_id"] = parent_chunk_id
        custom_tags["parent_chunk_size"] = parent_chunk_size
        custom_tags["child_chunk_size"] = child_chunk_size
        custom_tags["chunk_overlap"] = chunk_overlap

        return ChunkMetadata(
            source_id=source_id,
            document_id=version.document_id,
            version_id=version.version_id,
            company_id=company_id,
            workspace_id=workspace_id,
            source_url=source_url,
            source_category=source_category,
            collected_at=version.acquired_at,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            language="en",
            custom_tags=custom_tags,
        )
