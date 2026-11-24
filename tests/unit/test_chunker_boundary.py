"""Unit tests for Knowledge Pipeline Chunker Boundary and DocumentChunk domain model.

Verifies DocumentChunk provenance, hierarchy (parent_child), validation rules,
BaseDocumentChunker interface compliance, and strict LlamaIndex isolation.
"""

from datetime import datetime, timezone

from pydantic import ValidationError
import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseDocumentChunker,
    ChunkingParameters,
    ChunkMetadata,
    DocumentChunk,
    DocumentChunker,
    DocumentVersion,
    compute_content_hash,
)
from competitive_intelligence.source_acquisition import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    ContentType,
    PlatformType,
    Source,
    SourceCategory,
    SourceClassification,
    SourceIdentity,
    SourceLocation,
    ToolChain,
)


# ==============================================================================
# Helper Factories
# ==============================================================================


def create_test_metadata(
    source_id: str = "src_1",
    document_id: str = "doc_1",
    version_id: str = "ver_1",
    chunk_index: int = 0,
) -> ChunkMetadata:
    """Helper to construct valid ChunkMetadata."""
    return ChunkMetadata(
        source_id=source_id,
        document_id=document_id,
        version_id=version_id,
        company_id="comp_1",
        workspace_id="ws_1",
        source_url="https://example.com/docs/api",
        source_category="documentation",
        collected_at=datetime.now(timezone.utc),
        chunk_index=chunk_index,
        total_chunks=1,
    )


def create_test_version(
    doc_id: str = "doc_1",
    version_id: str = "ver_1",
    content: str = "# API Documentation\nEndpoint: /v1/pricing",
) -> DocumentVersion:
    """Helper to construct a valid DocumentVersion."""
    return DocumentVersion(
        version_id=version_id,
        document_id=doc_id,
        content=content,
        content_hash=compute_content_hash(content),
        acquired_at=datetime.now(timezone.utc),
    )


# ==============================================================================
# DocumentChunk Domain Model Tests
# ==============================================================================


def test_valid_document_chunk_instantiation() -> None:
    """Test creating a valid DocumentChunk with full attributes."""
    meta = create_test_metadata(chunk_index=0)
    chunk = DocumentChunk(
        chunk_id="chk_123",
        document_id="doc_1",
        version_id="ver_1",
        content="Paragraph chunk text.",
        chunk_index=0,
        metadata=meta,
        content_hash="hash_abc",
        parent_chunk_id=None,
        chunk_level=0,
    )

    assert chunk.chunk_id == "chk_123"
    assert chunk.document_id == "doc_1"
    assert chunk.version_id == "ver_1"
    assert chunk.content == "Paragraph chunk text."
    assert chunk.chunk_index == 0
    assert chunk.metadata == meta
    assert chunk.parent_chunk_id is None
    assert chunk.chunk_level == 0
    assert chunk.created_at.tzinfo is not None


def test_document_chunk_syncs_provenance_from_metadata() -> None:
    """Test that document_id, version_id, chunk_index, and hash are auto-derived if omitted."""
    meta = create_test_metadata(document_id="doc_auto", version_id="ver_auto", chunk_index=3)
    chunk = DocumentChunk(
        content="Auto-synced chunk content",
        metadata=meta,
    )

    assert chunk.document_id == "doc_auto"
    assert chunk.version_id == "ver_auto"
    assert chunk.chunk_index == 3
    assert len(chunk.content_hash) > 0
    assert chunk.content_hash == compute_content_hash("Auto-synced chunk content")


def test_parent_child_hierarchy_representation() -> None:
    """Test representing hierarchical parent-child relationships between chunks."""
    meta_parent = create_test_metadata(chunk_index=0)
    parent_chunk = DocumentChunk(
        chunk_id="chk_parent_1",
        content="# Main Section Header\nOverview and child topics follow.",
        metadata=meta_parent,
        parent_chunk_id=None,
        chunk_level=0,
    )

    meta_child = create_test_metadata(chunk_index=1)
    child_chunk = DocumentChunk(
        chunk_id="chk_child_1",
        content="Detailed subsection content under main section.",
        metadata=meta_child,
        parent_chunk_id=parent_chunk.chunk_id,
        chunk_level=1,
    )

    assert parent_chunk.parent_chunk_id is None
    assert parent_chunk.chunk_level == 0
    assert child_chunk.parent_chunk_id == "chk_parent_1"
    assert child_chunk.chunk_level == 1


@pytest.mark.parametrize("empty_content", ["", "   ", "\t\n  "])
def test_empty_content_rejected(empty_content: str) -> None:
    """Test that empty or whitespace-only content raises ValidationError."""
    meta = create_test_metadata()
    with pytest.raises(ValidationError, match="content cannot be empty"):
        DocumentChunk(
            content=empty_content,
            metadata=meta,
        )


@pytest.mark.parametrize(
    "invalid_index",
    [-1, -5],
)
def test_negative_chunk_index_or_level_rejected(invalid_index: int) -> None:
    """Test that negative chunk_index or chunk_level raises ValidationError."""
    meta = create_test_metadata()
    with pytest.raises(ValidationError, match="cannot be negative"):
        DocumentChunk(
            content="Valid content",
            chunk_index=invalid_index,
            metadata=meta,
        )

    with pytest.raises(ValidationError, match="cannot be negative"):
        DocumentChunk(
            content="Valid content",
            chunk_level=invalid_index,
            metadata=meta,
        )


# ==============================================================================
# Chunker Contract & Library Isolation Tests
# ==============================================================================


class FakeTestChunker(BaseDocumentChunker):
    """Mock implementation of BaseDocumentChunker for contract testing."""

    def chunk(
        self,
        version: DocumentVersion,
        parameters: ChunkingParameters | None = None,
        source: Source | None = None,
    ) -> list[DocumentChunk]:
        params = parameters or ChunkingParameters()
        chunk_size = params.chunk_size or 500
        text = version.content

        # Simple naive slicing for test purposes
        chunks: list[DocumentChunk] = []
        for i, start in enumerate(range(0, len(text), chunk_size)):
            slice_text = text[start : start + chunk_size]
            meta = ChunkMetadata(
                source_id=source.identity.source_id if source else "src_test",
                document_id=version.document_id,
                version_id=version.version_id,
                company_id=source.identity.company_id if source else "comp_test",
                workspace_id=source.identity.workspace_id if source else "ws_test",
                source_url=source.location.url_or_identifier if source else "https://example.com",
                source_category="test",
                collected_at=version.acquired_at,
                start_char=start,
                end_char=start + len(slice_text),
                chunk_index=i,
            )
            chunks.append(
                DocumentChunk(
                    content=slice_text,
                    metadata=meta,
                    chunk_index=i,
                )
            )
        return chunks


def test_chunker_contract_compliance() -> None:
    """Test that BaseDocumentChunker contract can be implemented and executed without LlamaIndex types."""
    assert DocumentChunker is BaseDocumentChunker

    chunker = FakeTestChunker()
    version = create_test_version(content="Section 1. Section 2. Section 3.")
    params = ChunkingParameters(chunk_size=10)

    chunks = chunker.chunk(version=version, parameters=params)

    assert len(chunks) == 4
    for i, c in enumerate(chunks):
        assert isinstance(c, DocumentChunk)
        assert c.chunk_index == i
        assert c.document_id == version.document_id
        assert c.version_id == version.version_id


def test_llamaindex_isolation_from_domain_models() -> None:
    """Verify that domain interfaces and models do not expose or import LlamaIndex objects."""
    import competitive_intelligence.knowledge_pipeline.interfaces as iface_mod
    import competitive_intelligence.knowledge_pipeline.models as models_mod

    for mod in (iface_mod, models_mod):
        mod_file = getattr(mod, "__file__", "")
        assert mod_file != ""
        with open(mod_file, "r", encoding="utf-8") as f:
            code = f.read()
        assert "llama_index" not in code
        assert "BaseNode" not in code
        assert "TextNode" not in code
