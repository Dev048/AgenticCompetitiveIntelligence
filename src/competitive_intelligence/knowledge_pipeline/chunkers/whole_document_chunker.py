"""Whole-document chunker implementation for Knowledge Pipeline subsystem.

Produces a single DocumentChunk containing the entire DocumentVersion content
when document splitting is unnecessary.
"""

from typing import Any
import uuid

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseDocumentChunker,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkMetadata,
    ChunkingParameters,
    DocumentChunk,
    DocumentVersion,
)
from competitive_intelligence.source_acquisition.models import Source


def derive_whole_document_chunk_id(version_id: str) -> str:
    """Derive deterministic chunk UUID for whole document chunking using UUID v5."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:ci:chunk:{version_id}:0"))


class WholeDocumentChunker(BaseDocumentChunker):
    """Deterministic chunker that wraps complete DocumentVersion content into a single DocumentChunk.

    Responsibilities:
    - Verifies content presence (rejects empty/whitespace content).
    - Preserves complete original content without modification or slicing.
    - Generates a deterministic chunk_id for the same DocumentVersion.
    - Sets chunk_index=0, chunk_level=0, parent_chunk_id=None.
    - Populates ChunkMetadata with full text character bounds and provenance.
    - Purely deterministic and offline; no external libraries or LLMs.
    """

    def chunk(
        self,
        version: DocumentVersion,
        parameters: ChunkingParameters | None = None,
        source: Source | None = None,
    ) -> list[DocumentChunk]:
        """Partition DocumentVersion into exactly one whole-document DocumentChunk.

        Args:
            version: The DocumentVersion to chunk.
            parameters: Optional chunking parameters (ignored for whole document strategy).
            source: Optional Source domain model for enriched provenance metadata.

        Returns:
            A list containing exactly one DocumentChunk.

        Raises:
            ValueError: If version.content is empty or whitespace-only.
        """
        if not version.content or not version.content.strip():
            raise ValueError("Cannot chunk DocumentVersion with empty or whitespace-only content")

        chunk_id = derive_whole_document_chunk_id(version.version_id)
        metadata = self._build_chunk_metadata(version, source)

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=version.document_id,
            version_id=version.version_id,
            content=version.content,
            chunk_index=0,
            metadata=metadata,
            content_hash=version.content_hash,
            parent_chunk_id=None,
            chunk_level=0,
            created_at=version.acquired_at,
        )

        return [chunk]

    def _build_chunk_metadata(
        self, version: DocumentVersion, source: Source | None
    ) -> ChunkMetadata:
        """Construct provenance ChunkMetadata for the whole-document chunk."""
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

        return ChunkMetadata(
            source_id=source_id,
            document_id=version.document_id,
            version_id=version.version_id,
            company_id=company_id,
            workspace_id=workspace_id,
            source_url=source_url,
            source_category=source_category,
            collected_at=version.acquired_at,
            start_char=0,
            end_char=len(version.content),
            chunk_index=0,
            total_chunks=1,
            language="en",
            custom_tags=custom_tags,
        )
