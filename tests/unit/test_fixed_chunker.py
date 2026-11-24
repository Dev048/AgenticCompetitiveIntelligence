"""Unit tests for Knowledge Pipeline FixedChunker using LlamaIndex SentenceSplitter.

Verifies fixed-size text chunking, overlap application, document ordering,
provenance preservation, deterministic ID generation, validation rules,
and strict LlamaIndex isolation.
"""

from datetime import datetime, timezone

import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseDocumentChunker,
    ChunkingParameters,
    DocumentChunk,
    DocumentVersion,
    FixedChunker,
    compute_content_hash,
    derive_fixed_chunk_id,
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


def create_test_version(
    doc_id: str = "doc_fixed_1",
    version_id: str = "ver_fixed_1",
    content: str = (
        "Artificial intelligence is rapidly evolving. "
        "Language models now possess advanced reasoning capabilities. "
        "Competitive intelligence requires real-time evidence tracking. "
        "Agents must verify facts against grounded sources. "
        "Deterministic pipeline logic prevents hallucinations. "
        "Vector search enables rapid contextual retrieval. "
        "Knowledge graphs enrich relational signals."
    ),
    metadata: dict[str, str] | None = None,
) -> DocumentVersion:
    """Helper to construct a valid DocumentVersion."""
    return DocumentVersion(
        version_id=version_id,
        document_id=doc_id,
        content=content,
        content_hash=compute_content_hash(content),
        acquired_at=datetime.now(timezone.utc),
        metadata=metadata or {"source_url": "https://example.com/ai-report", "author": "Research Team"},
    )


def create_test_source(
    source_id: str = "src_ai_report",
    company_id: str = "comp_openai",
    workspace_id: str = "ws_primary",
    url: str = "https://example.com/ai-report",
) -> Source:
    """Helper to construct a valid Source domain model."""
    return Source(
        identity=SourceIdentity(
            source_id=source_id,
            company_id=company_id,
            workspace_id=workspace_id,
            display_name="AI Progress Report",
        ),
        classification=SourceClassification(
            category=SourceCategory.DOCUMENTATION,
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier=url,
            access_pattern=AccessPattern.PUBLIC_WEB,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
        ),
    )


# ==============================================================================
# FixedChunker Unit Tests
# ==============================================================================


def test_chunker_is_instance_of_base_chunker() -> None:
    """Test that FixedChunker implements the BaseDocumentChunker interface."""
    chunker = FixedChunker()
    assert isinstance(chunker, BaseDocumentChunker)


def test_short_document_produces_single_chunk() -> None:
    """Test that content smaller than chunk_size produces exactly one chunk."""
    chunker = FixedChunker(default_chunk_size=500, default_chunk_overlap=50)
    version = create_test_version(content="Short single sentence.")

    chunks = chunker.chunk(version=version)

    assert len(chunks) == 1
    assert chunks[0].content == "Short single sentence."
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata.total_chunks == 1


def test_long_document_splits_into_multiple_chunks() -> None:
    """Test that long text splits into multiple sequential chunks."""
    chunker = FixedChunker()
    version = create_test_version()
    params = ChunkingParameters(chunk_size=15, chunk_overlap=0)

    chunks = chunker.chunk(version=version, parameters=params)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert isinstance(c, DocumentChunk)
        assert c.chunk_index == i
        assert c.metadata.total_chunks == len(chunks)
        assert c.metadata.chunk_index == i


def test_configured_chunk_size_and_overlap_applied() -> None:
    """Test that custom chunk_size and chunk_overlap parameters are respected."""
    chunker = FixedChunker()
    version = create_test_version()

    params_no_overlap = ChunkingParameters(chunk_size=20, chunk_overlap=0)
    chunks_no_overlap = chunker.chunk(version=version, parameters=params_no_overlap)

    params_with_overlap = ChunkingParameters(chunk_size=20, chunk_overlap=8)
    chunks_with_overlap = chunker.chunk(version=version, parameters=params_with_overlap)

    # Overlapping chunks may produce equal or more chunks than non-overlapping
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)
    assert chunks_with_overlap[0].metadata.custom_tags["chunk_size"] == 20
    assert chunks_with_overlap[0].metadata.custom_tags["chunk_overlap"] == 8


def test_chunk_order_and_content_preservation() -> None:
    """Test that chunk ordering is preserved sequentially and content matches original text."""
    chunker = FixedChunker()
    version = create_test_version()
    params = ChunkingParameters(chunk_size=20, chunk_overlap=0)

    chunks = chunker.chunk(version=version, parameters=params)

    for i in range(len(chunks)):
        assert chunks[i].chunk_index == i
        assert chunks[i].content in version.content
        assert len(chunks[i].content) > 0


def test_provenance_preservation_with_and_without_source() -> None:
    """Test that provenance fields are properly populated on all chunks."""
    chunker = FixedChunker()
    version = create_test_version()
    source = create_test_source()
    params = ChunkingParameters(chunk_size=20, chunk_overlap=0)

    chunks_with_source = chunker.chunk(version=version, parameters=params, source=source)
    for c in chunks_with_source:
        assert c.document_id == version.document_id
        assert c.version_id == version.version_id
        assert c.metadata.source_id == source.identity.source_id
        assert c.metadata.company_id == source.identity.company_id
        assert c.metadata.workspace_id == source.identity.workspace_id
        assert c.metadata.source_url == source.location.url_or_identifier
        assert c.metadata.source_category == "documentation"

    chunks_no_source = chunker.chunk(version=version, parameters=params, source=None)
    for c in chunks_no_source:
        assert c.metadata.source_url == "https://example.com/ai-report"


def test_deterministic_repeated_execution() -> None:
    """Test that chunking the same version with same parameters produces identical chunk IDs and hashes."""
    chunker = FixedChunker()
    version = create_test_version(version_id="ver_stable_fixed_123")
    params = ChunkingParameters(chunk_size=25, chunk_overlap=5)

    run1 = chunker.chunk(version=version, parameters=params)
    run2 = chunker.chunk(version=version, parameters=params)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=True):
        assert c1.chunk_id == c2.chunk_id
        assert c1.content == c2.content
        assert c1.content_hash == c2.content_hash
        assert c1.chunk_id == derive_fixed_chunk_id("ver_stable_fixed_123", c1.chunk_index, 25, 5)


@pytest.mark.parametrize("empty_text", ["", "   ", "\t\n  \r\n"])
def test_empty_content_raises_value_error(empty_text: str) -> None:
    """Test that empty or whitespace-only content raises ValueError."""
    chunker = FixedChunker()
    version = create_test_version(content="dummy valid content")
    object.__setattr__(version, "content", empty_text)

    with pytest.raises(ValueError, match="Cannot chunk DocumentVersion with empty or whitespace-only content"):
        chunker.chunk(version=version)


def test_no_llamaindex_objects_in_domain_outputs() -> None:
    """Verify that returned objects are strictly DocumentChunk without LlamaIndex node types."""
    chunker = FixedChunker()
    version = create_test_version()

    chunks = chunker.chunk(version=version)

    for c in chunks:
        assert type(c) is DocumentChunk
        # Verify no LlamaIndex node object attributes attached
        assert not hasattr(c, "node")
        assert not hasattr(c, "text_node")
        assert isinstance(c.metadata.custom_tags, dict)


def test_markdown_structure_preserved_in_chunks() -> None:
    """Test that markdown content containing headings and bullets is preserved."""
    markdown_content = (
        "# System Architecture\n\n"
        "## Component Overview\n"
        "- Subsystem 1: Source Acquisition\n"
        "- Subsystem 2: Knowledge Pipeline\n"
        "- Subsystem 3: Retrieval Engine\n\n"
        "## Verification Invariants\n"
        "All claims must maintain direct provenance pointers."
    )
    chunker = FixedChunker()
    version = create_test_version(content=markdown_content)
    params = ChunkingParameters(chunk_size=30, chunk_overlap=5)

    chunks = chunker.chunk(version=version, parameters=params)

    combined_text = " ".join(c.content for c in chunks)
    assert "System Architecture" in combined_text
    assert "Source Acquisition" in combined_text
    assert "Knowledge Pipeline" in combined_text
