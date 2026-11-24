"""Recursive structure-aware chunker implementation for Knowledge Pipeline subsystem.

Combines LlamaIndex MarkdownNodeParser (for structure and heading hierarchy)
and SentenceSplitter (for recursive sentence-aware splitting of oversized sections).
"""

import re
from typing import Any
import uuid

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
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

DEFAULT_RECURSIVE_CHUNK_SIZE = 1000
DEFAULT_RECURSIVE_CHUNK_OVERLAP = 150


def derive_recursive_chunk_id(
    version_id: str, chunk_index: int, chunk_size: int, chunk_overlap: int
) -> str:
    """Derive deterministic chunk UUID for recursive chunking using UUID v5."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"urn:ci:chunk:recursive:{version_id}:{chunk_index}:{chunk_size}:{chunk_overlap}",
        )
    )


class RecursiveChunker(BaseDocumentChunker):
    """Structure-aware recursive chunker using LlamaIndex.

    Responsibilities:
    - Verifies content presence (rejects empty/whitespace content).
    - Extracts Markdown sections and attaches ancestor heading context.
    - Preserves headings/context with the content they describe.
    - Recursively splits oversized sections using sentence-aware splitting.
    - Preserves sequential document order.
    - Produces deterministic chunk_id values.
    - Strictly isolates LlamaIndex types from domain outputs.
    """

    def __init__(
        self,
        default_chunk_size: int = DEFAULT_RECURSIVE_CHUNK_SIZE,
        default_chunk_overlap: int = DEFAULT_RECURSIVE_CHUNK_OVERLAP,
    ) -> None:
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap

    def chunk(
        self,
        version: DocumentVersion,
        parameters: ChunkingParameters | None = None,
        source: Source | None = None,
    ) -> list[DocumentChunk]:
        """Partition DocumentVersion content using structure-aware recursive splitting.

        Args:
            version: The DocumentVersion to chunk.
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

        # 1. Extract structural sections with heading context
        section_splits = self._extract_sections_with_headings(version.content, chunk_size, chunk_overlap)

        # 2. Build DocumentChunk instances
        total_chunks = len(section_splits)
        chunks: list[DocumentChunk] = []
        search_start = 0

        for i, (chunk_text, header_meta) in enumerate(section_splits):
            chunk_id = derive_recursive_chunk_id(
                version_id=version.version_id,
                chunk_index=i,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Character offset calculation against original text
            # Search for the core body text without prepended ancestor headers if needed
            body_match = chunk_text.split("\n\n")[-1] if "\n\n" in chunk_text else chunk_text
            start_char = version.content.find(body_match, search_start)
            if start_char == -1:
                start_char = version.content.find(body_match)
                if start_char == -1:
                    start_char = search_start
            end_char = start_char + len(body_match)
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
                header_meta=header_meta,
            )

            content_hash = compute_content_hash(chunk_text)

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=version.document_id,
                version_id=version.version_id,
                content=chunk_text,
                chunk_index=i,
                metadata=metadata,
                content_hash=content_hash,
                parent_chunk_id=None,
                chunk_level=0,
                created_at=version.acquired_at,
            )
            chunks.append(chunk)

        return chunks

    def _extract_sections_with_headings(
        self,
        content: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Extract structured sections, retaining heading context, and recursively split oversized sections."""
        # Check if content has markdown headings
        has_markdown_headers = bool(re.search(r"^#{1,6}\s", content, re.MULTILINE))

        sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not has_markdown_headers:
            # Fallback: Plain text splitting with SentenceSplitter
            splits = sentence_splitter.split_text(content)
            if not splits:
                splits = [content]
            return [(s, {}) for s in splits]

        # Use LlamaIndex MarkdownNodeParser to parse structure
        markdown_parser = MarkdownNodeParser()
        llama_doc = LlamaDocument(text=content)
        nodes = markdown_parser.get_nodes_from_documents([llama_doc])

        if not nodes:
            # Fallback to SentenceSplitter if parser returns no nodes
            splits = sentence_splitter.split_text(content)
            return [(s, {}) for s in splits] if splits else [(content, {})]

        # Process each node and prepend ancestor heading context
        raw_sections: list[tuple[str, dict[str, Any]]] = []
        for idx, node in enumerate(nodes):
            header_path = str(node.metadata.get("header_path", "/"))
            path_parts = [p for p in header_path.split("/") if p]
            node_text = node.get_content().strip()

            if not node_text:
                continue

            lines = node_text.split("\n")
            is_pure_header = len(lines) == 1 and lines[0].startswith("#")

            # If node is just a top-level header without body and there are subsequent children, skip it
            if is_pure_header and idx < len(nodes) - 1:
                continue

            # Build ancestor headers prefix
            ancestors: list[str] = []
            for level, part in enumerate(path_parts, 1):
                ancestors.append("#" * level + " " + part)

            if ancestors:
                prefix = "\n".join(ancestors) + "\n"
                formatted_text = prefix + node_text
            else:
                formatted_text = node_text

            header_meta = {
                "header_path": header_path,
                "headings": path_parts,
            }
            raw_sections.append((formatted_text, header_meta))

        if not raw_sections:
            return [(content, {})]

        # Check if entire document fits into 1 chunk
        if len(raw_sections) == 1 and len(raw_sections[0][0]) <= chunk_size:
            return raw_sections

        # If multiple sections exist, check each section for oversized content
        final_splits: list[tuple[str, dict[str, Any]]] = []
        for section_text, header_meta in raw_sections:
            # Check if section fits within chunk_size
            sub_splits = sentence_splitter.split_text(section_text)
            if len(sub_splits) <= 1:
                final_splits.append((section_text, header_meta))
            else:
                # Oversized section was recursively split
                for sub in sub_splits:
                    final_splits.append((sub, header_meta))

        return final_splits

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
        header_meta: dict[str, Any],
    ) -> ChunkMetadata:
        """Construct provenance ChunkMetadata for a recursive chunk."""
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
        custom_tags["strategy"] = ChunkingStrategy.RECURSIVE.value
        custom_tags["chunk_size"] = chunk_size
        custom_tags["chunk_overlap"] = chunk_overlap

        if "header_path" in header_meta:
            custom_tags["header_path"] = header_meta["header_path"]
        if "headings" in header_meta:
            custom_tags["headings"] = header_meta["headings"]

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
