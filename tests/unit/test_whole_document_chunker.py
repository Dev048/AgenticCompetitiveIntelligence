"""Unit tests for Knowledge Pipeline WholeDocumentChunker.

Verifies single-chunk generation, content preservation, provenance retention,
deterministic ID generation, hierarchy defaults, validation rules, and LlamaIndex isolation.
"""

from datetime import datetime, timezone

import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseDocumentChunker,
    DocumentChunk,
    DocumentVersion,
    WholeDocumentChunker,
    compute_content_hash,
    derive_whole_document_chunk_id,
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
    doc_id: str = "doc_pricing_100",
    version_id: str = "ver_pricing_100",
    content: str = "# Pricing Page\nPro: $20/mo\nEnterprise: Custom",
    metadata: dict[str, str] | None = None,
) -> DocumentVersion:
    """Helper to construct a valid DocumentVersion."""
    return DocumentVersion(
        version_id=version_id,
        document_id=doc_id,
        content=content,
        content_hash=compute_content_hash(content),
        acquired_at=datetime.now(timezone.utc),
        metadata=metadata or {"tier": "enterprise", "source_url": "https://competitor.com/pricing"},
    )


def create_test_source(
    source_id: str = "src_pricing_100",
    company_id: str = "comp_openai",
    workspace_id: str = "ws_primary",
    url: str = "https://competitor.com/pricing",
) -> Source:
    """Helper to construct a valid Source model."""
    return Source(
        identity=SourceIdentity(
            source_id=source_id,
            company_id=company_id,
            workspace_id=workspace_id,
            display_name="Competitor Pricing",
        ),
        classification=SourceClassification(
            category=SourceCategory.PRICING,
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
# WholeDocumentChunker Unit Tests
# ==============================================================================


def test_chunker_is_instance_of_base_chunker() -> None:
    """Test that WholeDocumentChunker implements BaseDocumentChunker interface."""
    chunker = WholeDocumentChunker()
    assert isinstance(chunker, BaseDocumentChunker)


def test_produces_exactly_one_chunk_with_complete_content() -> None:
    """Test that WholeDocumentChunker produces exactly one chunk with full unmodified text."""
    chunker = WholeDocumentChunker()
    content = "# Short Pricing Note\n- Basic: $10\n- Pro: $20"
    version = create_test_version(content=content)
    source = create_test_source()

    chunks = chunker.chunk(version=version, source=source)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, DocumentChunk)
    assert chunk.content == content
    assert chunk.chunk_index == 0
    assert chunk.parent_chunk_id is None
    assert chunk.chunk_level == 0
    assert chunk.document_id == version.document_id
    assert chunk.version_id == version.version_id
    assert chunk.content_hash == version.content_hash


def test_provenance_preservation_with_source() -> None:
    """Test that provenance fields are properly populated when Source is provided."""
    chunker = WholeDocumentChunker()
    version = create_test_version()
    source = create_test_source(
        source_id="src_unique_99",
        company_id="comp_special",
        workspace_id="ws_enterprise",
        url="https://competitor.com/enterprise/pricing",
    )

    chunks = chunker.chunk(version=version, source=source)
    chunk = chunks[0]

    assert chunk.metadata.source_id == "src_unique_99"
    assert chunk.metadata.company_id == "comp_special"
    assert chunk.metadata.workspace_id == "ws_enterprise"
    assert chunk.metadata.source_url == "https://competitor.com/enterprise/pricing"
    assert chunk.metadata.source_category == "pricing"
    assert chunk.metadata.start_char == 0
    assert chunk.metadata.end_char == len(version.content)
    assert chunk.metadata.total_chunks == 1


def test_provenance_fallback_without_source() -> None:
    """Test that provenance fields fallback to version.metadata when Source is None."""
    chunker = WholeDocumentChunker()
    version = create_test_version(
        doc_id="doc_custom_42",
        metadata={
            "source_id": "src_meta_1",
            "company_id": "comp_meta_1",
            "workspace_id": "ws_meta_1",
            "source_url": "https://custom.com/page",
            "source_category": "blog",
        },
    )

    chunks = chunker.chunk(version=version, source=None)
    chunk = chunks[0]

    assert chunk.metadata.source_id == "src_meta_1"
    assert chunk.metadata.company_id == "comp_meta_1"
    assert chunk.metadata.workspace_id == "ws_meta_1"
    assert chunk.metadata.source_url == "https://custom.com/page"
    assert chunk.metadata.source_category == "blog"


def test_deterministic_chunk_id_generation() -> None:
    """Test that chunking the same DocumentVersion repeatedly generates the identical chunk_id."""
    chunker = WholeDocumentChunker()
    version = create_test_version(version_id="ver_stable_abc")

    chunks1 = chunker.chunk(version=version)
    chunks2 = chunker.chunk(version=version)

    assert chunks1[0].chunk_id == chunks2[0].chunk_id
    assert chunks1[0].chunk_id == derive_whole_document_chunk_id("ver_stable_abc")


def test_custom_tags_metadata_preservation() -> None:
    """Test that custom metadata tags on DocumentVersion are preserved in ChunkMetadata."""
    chunker = WholeDocumentChunker()
    version = create_test_version(metadata={"author": "Marketing", "release": "v2.1"})

    chunks = chunker.chunk(version=version)
    assert chunks[0].metadata.custom_tags["author"] == "Marketing"
    assert chunks[0].metadata.custom_tags["release"] == "v2.1"


@pytest.mark.parametrize("empty_text", ["", "   ", "\t\n  \r\n"])
def test_empty_content_raises_value_error(empty_text: str) -> None:
    """Test that chunking a DocumentVersion with empty content raises ValueError."""
    chunker = WholeDocumentChunker()
    # Construct DocumentVersion with dummy non-empty content then overwrite content field
    version = create_test_version(content="temporary valid content")
    object.__setattr__(version, "content", empty_text)

    with pytest.raises(ValueError, match="Cannot chunk DocumentVersion with empty or whitespace-only content"):
        chunker.chunk(version=version)


def test_no_llamaindex_dependency() -> None:
    """Verify that WholeDocumentChunker contains zero LlamaIndex imports or dependencies."""
    import competitive_intelligence.knowledge_pipeline.chunkers.whole_document_chunker as chunker_mod

    mod_file = getattr(chunker_mod, "__file__", "")
    assert mod_file != ""

    with open(mod_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "llama_index" not in code
    assert "BaseNode" not in code
    assert "TextNode" not in code
