"""Unit tests for Knowledge Pipeline ParentChildChunker using LlamaIndex.

Verifies two-level parent-child hierarchy, parent_chunk_id pointers, chunk_level,
provenance preservation, deterministic IDs, sizing configuration, edge cases,
and strict LlamaIndex isolation.
"""

from datetime import datetime, timezone

import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseDocumentChunker,
    ChunkingParameters,
    DocumentChunk,
    DocumentVersion,
    ParentChildChunker,
    compute_content_hash,
    derive_parent_child_chunk_id,
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
    doc_id: str = "doc_pc_1",
    version_id: str = "ver_pc_1",
    content: str = (
        "# Technical Architecture Documentation\n\n"
        "## Vector Storage Subsystem\n"
        "We use Qdrant for dense and sparse vector indexing. "
        "Payload filtering enforces strict multi-tenant boundaries. "
        "Collections are sharded for horizontal scalability.\n\n"
        "## Hybrid Retrieval Engine\n"
        "Sparse BM25 lexical search is fused with dense vector retrieval. "
        "Reciprocal Rank Fusion (RRF) merges candidate lists deterministically. "
        "Cross-encoder reranking refines the top-K evidence excerpts.\n\n"
        "## Agentic RAG Workflow\n"
        "LangGraph orchestrates deterministic state transitions. "
        "Verification nodes validate factual grounding against raw citations."
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
        metadata=metadata or {"source_url": "https://docs.enterprise.com/arch", "tier": "enterprise"},
    )


def create_test_source(
    source_id: str = "src_tech_docs",
    company_id: str = "comp_anthropic",
    workspace_id: str = "ws_primary",
    url: str = "https://docs.enterprise.com/arch",
) -> Source:
    """Helper to construct a valid Source model."""
    return Source(
        identity=SourceIdentity(
            source_id=source_id,
            company_id=company_id,
            workspace_id=workspace_id,
            display_name="Architecture Docs",
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
# ParentChildChunker Unit Tests
# ==============================================================================


def test_chunker_is_instance_of_base_chunker() -> None:
    """Test that ParentChildChunker implements BaseDocumentChunker interface."""
    chunker = ParentChildChunker()
    assert isinstance(chunker, BaseDocumentChunker)


def test_parent_child_hierarchy_and_levels() -> None:
    """Test that output contains valid parent (level 0) and child (level 1) chunks."""
    chunker = ParentChildChunker()
    version = create_test_version()
    params = ChunkingParameters(parent_chunk_size=100, child_chunk_size=30, chunk_overlap=5)

    chunks = chunker.chunk(version=version, parameters=params)

    parents = [c for c in chunks if c.chunk_level == 0]
    children = [c for c in chunks if c.chunk_level == 1]

    assert len(parents) >= 1
    assert len(children) >= 1
    assert len(chunks) == len(parents) + len(children)

    # Verify parent attributes
    for p in parents:
        assert p.parent_chunk_id is None
        assert p.chunk_level == 0
        assert p.metadata.custom_tags["chunk_level"] == 0

    # Verify child attributes
    for c in children:
        assert c.parent_chunk_id is not None
        assert c.chunk_level == 1
        assert c.metadata.custom_tags["chunk_level"] == 1


def test_every_child_has_exactly_one_parent() -> None:
    """Invariant: Every child must reference a valid parent chunk."""
    chunker = ParentChildChunker()
    version = create_test_version()
    params = ChunkingParameters(parent_chunk_size=80, child_chunk_size=25, chunk_overlap=5)

    chunks = chunker.chunk(version=version, parameters=params)

    parent_ids = {c.chunk_id for c in chunks if c.chunk_level == 0}
    children = [c for c in chunks if c.chunk_level == 1]

    for child in children:
        assert child.parent_chunk_id in parent_ids


def test_every_parent_has_at_least_one_child() -> None:
    """Invariant: Every parent must have >= 1 associated child chunk."""
    chunker = ParentChildChunker()
    version = create_test_version()
    params = ChunkingParameters(parent_chunk_size=80, child_chunk_size=25, chunk_overlap=5)

    chunks = chunker.chunk(version=version, parameters=params)

    parents = [c for c in chunks if c.chunk_level == 0]
    children = [c for c in chunks if c.chunk_level == 1]

    for parent in parents:
        associated_children = [c for c in children if c.parent_chunk_id == parent.chunk_id]
        assert len(associated_children) >= 1


def test_short_document_produces_valid_hierarchy() -> None:
    """Test that a short document produces 1 parent and 1 child covering the text cleanly."""
    chunker = ParentChildChunker()
    version = create_test_version(content="Short single sentence note.")

    chunks = chunker.chunk(version=version)

    parents = [c for c in chunks if c.chunk_level == 0]
    children = [c for c in chunks if c.chunk_level == 1]

    assert len(parents) == 1
    assert len(children) == 1
    assert children[0].parent_chunk_id == parents[0].chunk_id
    assert parents[0].content == "Short single sentence note."
    assert children[0].content == "Short single sentence note."


def test_configured_parent_and_child_sizes_applied() -> None:
    """Test that custom parent_chunk_size and child_chunk_size are stored and applied."""
    chunker = ParentChildChunker()
    version = create_test_version()
    params = ChunkingParameters(
        parent_chunk_size=150,
        child_chunk_size=40,
        custom_params={"parent_child_overlap": 8},
    )

    chunks = chunker.chunk(version=version, parameters=params)

    for c in chunks:
        assert c.metadata.custom_tags["parent_chunk_size"] == 150
        assert c.metadata.custom_tags["child_chunk_size"] == 40
        assert c.metadata.custom_tags["chunk_overlap"] == 8
        assert c.metadata.custom_tags["strategy"] == "parent_child"


def test_provenance_preservation_with_and_without_source() -> None:
    """Test that provenance fields are properly populated on both parent and child chunks."""
    chunker = ParentChildChunker()
    version = create_test_version()
    source = create_test_source(
        source_id="src_tech_arch",
        company_id="comp_special",
        workspace_id="ws_fintech",
        url="https://docs.enterprise.com/arch",
    )

    chunks_with_src = chunker.chunk(version=version, source=source)
    for c in chunks_with_src:
        assert c.document_id == version.document_id
        assert c.version_id == version.version_id
        assert c.metadata.source_id == "src_tech_arch"
        assert c.metadata.company_id == "comp_special"
        assert c.metadata.workspace_id == "ws_fintech"
        assert c.metadata.source_url == "https://docs.enterprise.com/arch"
        assert c.metadata.source_category == "documentation"

    chunks_no_src = chunker.chunk(version=version, source=None)
    for c in chunks_no_src:
        assert c.metadata.source_url == "https://docs.enterprise.com/arch"


def test_deterministic_repeated_execution() -> None:
    """Test that chunking the same version repeatedly generates identical chunk IDs and hierarchy."""
    chunker = ParentChildChunker()
    version = create_test_version(version_id="ver_stable_pc_789")
    params = ChunkingParameters(parent_chunk_size=100, child_chunk_size=30, chunk_overlap=5)

    run1 = chunker.chunk(version=version, parameters=params)
    run2 = chunker.chunk(version=version, parameters=params)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=True):
        assert c1.chunk_id == c2.chunk_id
        assert c1.content == c2.content
        assert c1.parent_chunk_id == c2.parent_chunk_id
        assert c1.chunk_level == c2.chunk_level
        assert c1.content_hash == c2.content_hash


@pytest.mark.parametrize("empty_text", ["", "   ", "\t\n  \r\n"])
def test_empty_content_raises_value_error(empty_text: str) -> None:
    """Test that empty or whitespace-only content raises ValueError."""
    chunker = ParentChildChunker()
    version = create_test_version(content="dummy valid content")
    object.__setattr__(version, "content", empty_text)

    with pytest.raises(ValueError, match="Cannot chunk DocumentVersion with empty or whitespace-only content"):
        chunker.chunk(version=version)


def test_no_llamaindex_types_leak_into_document_chunk() -> None:
    """Verify that returned objects are strictly DocumentChunk without LlamaIndex node types."""
    chunker = ParentChildChunker()
    version = create_test_version()

    chunks = chunker.chunk(version=version)

    for c in chunks:
        assert type(c) is DocumentChunk
        assert not hasattr(c, "node")
        assert not hasattr(c, "text_node")
        assert isinstance(c.metadata.custom_tags, dict)


def test_heading_and_section_preservation_in_hierarchy() -> None:
    """Test that section text and headings remain intact in parent and child chunks."""
    chunker = ParentChildChunker()
    version = create_test_version()
    params = ChunkingParameters(parent_chunk_size=100, child_chunk_size=35, chunk_overlap=5)

    chunks = chunker.chunk(version=version, parameters=params)

    parent_texts = " ".join(c.content for c in chunks if c.chunk_level == 0)
    child_texts = " ".join(c.content for c in chunks if c.chunk_level == 1)

    assert "Vector Storage Subsystem" in parent_texts
    assert "Hybrid Retrieval Engine" in parent_texts
    assert "Vector Storage Subsystem" in child_texts or "Hybrid Retrieval Engine" in child_texts
