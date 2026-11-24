"""Fixed-size chunker implementation for Knowledge Pipeline subsystem using LlamaIndex.

Partitions DocumentVersion content into deterministic fixed-size text chunks
with configurable overlap using LlamaIndex SentenceSplitter internally.
"""

from typing import Any
import uuid

from llama_index.core.node_parser import SentenceSplitter

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

DEFAULT_FIXED_CHUNK_SIZE = 500
DEFAULT_FIXED_CHUNK_OVERLAP = 50


def derive_fixed_chunk_id(
    version_id: str, chunk_index: int, chunk_size: int, chunk_overlap: int
) -> str:
    """Derive deterministic chunk UUID for fixed chunking using UUID v5."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"urn:ci:chunk:fixed:{version_id}:{chunk_index}:{chunk_size}:{chunk_overlap}",
        )
    )


class FixedChunker(BaseDocumentChunker):
    """Deterministic chunker splitting DocumentVersion content into fixed-size chunks using LlamaIndex.

    Responsibilities:
    - Verifies content presence (rejects empty/whitespace content).
    - Uses LlamaIndex SentenceSplitter with configured chunk_size and chunk_overlap.
    - Generates deterministic chunk_id values for each chunk.
    - Preserves sequential document ordering (chunk_index 0, 1, 2...).
    - Populates ChunkMetadata with provenance and character offset bounds.
    - Strictly isolates LlamaIndex types from domain outputs.
    """

    def __init__(
        self,
        default_chunk_size: int = DEFAULT_FIXED_CHUNK_SIZE,
        default_chunk_overlap: int = DEFAULT_FIXED_CHUNK_OVERLAP,
    ) -> None:
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    def chunk(
        self,
        version: DocumentVersion,
        parameters: ChunkingParameters | None = None,
        source: Source | None = None,
    ) -> list[DocumentChunk]:
        """Partition DocumentVersion content into fixed-size chunks.

        Args:
            version: The DocumentVersion snapshot to chunk.
            parameters: Optional chunking parameters (chunk_size, chunk_overlap).
            source: Optional Source domain model for enriched provenance metadata.

        Returns:
            A list of DocumentChunk instances.

        Raises:
            ValueError: If version.content is empty or whitespace-only.
        """
        if not version.content or not version.content.strip():
            raise ValueError("Cannot chunk DocumentVersion with empty or whitespace-only content")

        chunk_size = (
            parameters.chunk_size
            if parameters is not None and parameters.chunk_size is not None
            else self.default_chunk_size
        )
        chunk_overlap = (
            parameters.chunk_overlap
            if parameters is not None and parameters.chunk_overlap is not None
            else self.default_chunk_overlap
        )

        # Use LlamaIndex SentenceSplitter for fixed-size splitting
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        raw_splits = splitter.split_text(version.content)
        if not raw_splits:
            # Fallback if splitter returns empty for non-empty text
            raw_splits = [version.content]

        total_chunks = len(raw_splits)
        chunks: list[DocumentChunk] = []
        search_start = 0

        for i, split_text in enumerate(raw_splits):
            chunk_id = derive_fixed_chunk_id(
                version_id=version.version_id,
                chunk_index=i,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Character offset calculation
            start_char = version.content.find(split_text, search_start)
            if start_char == -1:
                start_char = version.content.find(split_text)
                if start_char == -1:
                    start_char = search_start
            end_char = start_char + len(split_text)
            search_start = max(0, start_char)

            metadata = self._build_chunk_metadata(
                version=version,
                source=source,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=start_char,
                end_char=end_char,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            content_hash = compute_content_hash(split_text)

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=version.document_id,
                version_id=version.version_id,
                content=split_text,
                chunk_index=i,
                metadata=metadata,
                content_hash=content_hash,
                parent_chunk_id=None,
                chunk_level=0,
                created_at=version.acquired_at,
            )
            chunks.append(chunk)

        return chunks

    def _build_chunk_metadata(
        self,
        version: DocumentVersion,
        source: Source | None,
        chunk_index: int,
        total_chunks: int,
        start_char: int,
        end_char: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> ChunkMetadata:
        """Construct provenance ChunkMetadata for a fixed chunk."""
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
        custom_tags["strategy"] = ChunkingStrategy.FIXED.value
        custom_tags["chunk_size"] = chunk_size
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
