"""Unit tests for Knowledge Pipeline RecursiveChunker using LlamaIndex.

Verifies structure-aware splitting, heading context retention, recursive splitting
of oversized sections, plain text fallbacks, deterministic IDs, and strict isolation.
"""

from datetime import datetime, timezone

import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseDocumentChunker,
    ChunkingParameters,
    DocumentChunk,
    DocumentVersion,
    RecursiveChunker,
    compute_content_hash,
    derive_recursive_chunk_id,
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
    doc_id: str = "doc_rec_1",
    version_id: str = "ver_rec_1",
    content: str = (
        "# Product Overview\n\n"
        "## Pricing\n\n"
        "Pro costs $20/month.\n\n"
        "## Enterprise\n\n"
        "Enterprise pricing is custom with dedicated support."
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
        metadata=metadata or {"source_url": "https://example.com/product", "author": "Product Marketing"},
    )


def create_test_source(
    source_id: str = "src_product_page",
    company_id: str = "comp_stripe",
    workspace_id: str = "ws_primary",
    url: str = "https://example.com/product",
) -> Source:
    """Helper to construct a valid Source model."""
    return Source(
        identity=SourceIdentity(
            source_id=source_id,
            company_id=company_id,
            workspace_id=workspace_id,
            display_name="Product Page",
        ),
        classification=SourceClassification(
            category=SourceCategory.DOCUMENTATION,
            platform=PlatformType.WEB,
            content_type=ContentType.MARKDOWN,
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
# RecursiveChunker Unit Tests
# ==============================================================================


def test_chunker_is_instance_of_base_chunker() -> None:
    """Test that RecursiveChunker implements BaseDocumentChunker interface."""
    chunker = RecursiveChunker()
    assert isinstance(chunker, BaseDocumentChunker)


def test_short_document_produces_single_chunk() -> None:
    """Test that a short document without markdown headings produces exactly one chunk."""
    chunker = RecursiveChunker(default_chunk_size=1000, default_chunk_overlap=150)
    version = create_test_version(content="Short plain text note without headers.")

    chunks = chunker.chunk(version=version)

    assert len(chunks) == 1
    assert chunks[0].content == "Short plain text note without headers."
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata.total_chunks == 1


def test_multiple_headings_and_section_context_retained() -> None:
    """Test that markdown headings are preserved and ancestor heading context is prepended."""
    chunker = RecursiveChunker()
    content = (
        "# Product\n\n"
        "## Pricing\n\n"
        "Pro costs $20/month.\n\n"
        "## Enterprise\n\n"
        "Enterprise pricing is custom."
    )
    version = create_test_version(content=content)

    chunks = chunker.chunk(version=version)

    assert len(chunks) == 2
    # Verify Pricing section contains parent header # Product
    assert "# Product" in chunks[0].content
    assert "## Pricing" in chunks[0].content
    assert "Pro costs $20/month." in chunks[0].content

    # Verify Enterprise section contains parent header # Product
    assert "# Product" in chunks[1].content
    assert "## Enterprise" in chunks[1].content
    assert "Enterprise pricing is custom." in chunks[1].content


def test_section_metadata_captured_in_custom_tags() -> None:
    """Test that header_path and heading hierarchy are stored in custom_tags."""
    chunker = RecursiveChunker()
    version = create_test_version()

    chunks = chunker.chunk(version=version)

    assert chunks[0].metadata.custom_tags["strategy"] == "recursive"
    assert "header_path" in chunks[0].metadata.custom_tags
    assert "headings" in chunks[0].metadata.custom_tags


def test_long_structured_document_splits_across_sections() -> None:
    """Test that multi-section document splits into multiple ordered chunks."""
    chunker = RecursiveChunker()
    long_content = "\n\n".join([f"## Section {i}\n\nContent for section {i}." for i in range(1, 6)])
    version = create_test_version(content=long_content)

    chunks = chunker.chunk(version=version)

    assert len(chunks) == 5
    for i, c in enumerate(chunks, 1):
        assert f"## Section {i}" in c.content
        assert c.chunk_index == i - 1


def test_recursive_splitting_of_oversized_section() -> None:
    """Test that an oversized section is recursively split into sub-chunks via SentenceSplitter."""
    chunker = RecursiveChunker()
    oversized_text = "## Deep Dive\n\n" + "This is a detailed sentence describing technical mechanics. " * 20
    version = create_test_version(content=oversized_text)
    params = ChunkingParameters(chunk_size=30, chunk_overlap=5)

    chunks = chunker.chunk(version=version, parameters=params)

    assert len(chunks) > 1
    # Verify chunk indices are sequential
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.metadata.total_chunks == len(chunks)


def test_configured_chunk_size_and_overlap_applied() -> None:
    """Test that custom chunk_size and chunk_overlap are stored and applied."""
    chunker = RecursiveChunker()
    version = create_test_version()
    params = ChunkingParameters(chunk_size=40, chunk_overlap=10)

    chunks = chunker.chunk(version=version, parameters=params)

    assert chunks[0].metadata.custom_tags["chunk_size"] == 40
    assert chunks[0].metadata.custom_tags["chunk_overlap"] == 10


def test_provenance_preservation_with_and_without_source() -> None:
    """Test that provenance fields are properly populated on all chunks."""
    chunker = RecursiveChunker()
    version = create_test_version()
    source = create_test_source(
        source_id="src_custom_page",
        company_id="comp_special",
        workspace_id="ws_fintech",
        url="https://example.com/product",
    )

    chunks_with_src = chunker.chunk(version=version, source=source)
    for c in chunks_with_src:
        assert c.document_id == version.document_id
        assert c.version_id == version.version_id
        assert c.metadata.source_id == "src_custom_page"
        assert c.metadata.company_id == "comp_special"
        assert c.metadata.workspace_id == "ws_fintech"
        assert c.metadata.source_url == "https://example.com/product"
        assert c.metadata.source_category == "documentation"

    chunks_no_src = chunker.chunk(version=version, source=None)
    for c in chunks_no_src:
        assert c.metadata.source_url == "https://example.com/product"


def test_deterministic_repeated_execution() -> None:
    """Test that chunking the same version with same parameters produces identical chunk IDs and hashes."""
    chunker = RecursiveChunker()
    version = create_test_version(version_id="ver_stable_rec_456")
    params = ChunkingParameters(chunk_size=50, chunk_overlap=10)

    run1 = chunker.chunk(version=version, parameters=params)
    run2 = chunker.chunk(version=version, parameters=params)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=True):
        assert c1.chunk_id == c2.chunk_id
        assert c1.content == c2.content
        assert c1.content_hash == c2.content_hash
        assert c1.chunk_id == derive_recursive_chunk_id("ver_stable_rec_456", c1.chunk_index, 50, 10)


def test_content_without_headings_works() -> None:
    """Test that plain text without headings splits properly via sentence-aware splitting."""
    chunker = RecursiveChunker()
    plain_content = "Sentence one is here. Sentence two is here. Sentence three is here. Sentence four is here."
    version = create_test_version(content=plain_content)
    params = ChunkingParameters(chunk_size=10, chunk_overlap=2)

    chunks = chunker.chunk(version=version, parameters=params)

    assert len(chunks) > 1
    combined = " ".join(c.content for c in chunks)
    assert "Sentence one is here." in combined
    assert "Sentence four is here." in combined


@pytest.mark.parametrize("empty_text", ["", "   ", "\t\n  \r\n"])
def test_empty_content_raises_value_error(empty_text: str) -> None:
    """Test that empty or whitespace-only content raises ValueError."""
    chunker = RecursiveChunker()
    version = create_test_version(content="dummy valid content")
    object.__setattr__(version, "content", empty_text)

    with pytest.raises(ValueError, match="Cannot chunk DocumentVersion with empty or whitespace-only content"):
        chunker.chunk(version=version)


def test_no_llamaindex_types_leak_into_document_chunk() -> None:
    """Verify that returned objects are strictly DocumentChunk without LlamaIndex node types."""
    chunker = RecursiveChunker()
    version = create_test_version()

    chunks = chunker.chunk(version=version)

    for c in chunks:
        assert type(c) is DocumentChunk
        assert not hasattr(c, "node")
        assert not hasattr(c, "text_node")
        assert isinstance(c.metadata.custom_tags, dict)
