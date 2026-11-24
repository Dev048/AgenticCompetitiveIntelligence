"""Unit tests for Knowledge Pipeline SemanticChunker using LlamaIndex and BaseEmbeddingProvider.

Verifies dependency injection of embedding providers, semantic boundary splitting,
breakpoint threshold configuration, provenance preservation, determinism,
and strict isolation from concrete embedding libraries.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseDocumentChunker,
    BaseEmbeddingProvider,
    ChunkingParameters,
    DocumentChunk,
    DocumentVersion,
    MockEmbeddingProvider,
    SemanticChunker,
    compute_content_hash,
    derive_semantic_chunk_id,
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
# Helper Factories & Spies
# ==============================================================================


class SpyEmbeddingProvider(BaseEmbeddingProvider):
    """Spy embedding provider that records calls and delegates to MockEmbeddingProvider."""

    def __init__(self, dimension: int = 16) -> None:
        self.mock = MockEmbeddingProvider(dimension=dimension)
        self.call_history: list[list[str]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.call_history.append(list(texts))
        return self.mock.embed_texts(texts)


def create_test_version(
    doc_id: str = "doc_sem_1",
    version_id: str = "ver_sem_1",
    content: str = (
        "The company was founded in 2020 to build developer intelligence tools. "
        "Their primary focus is on autonomous coding and agent verification.\n\n"
        "In Q3, enterprise revenue increased by 250% year-over-year. "
        "Sales were driven by large contracts with financial institutions.\n\n"
        "The backend infrastructure is built with Python and FastAPI. "
        "High-performance vector operations are handled by Qdrant clusters."
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
        metadata=metadata or {"source_url": "https://news.ycombinator.com/item?id=1", "topic": "business"},
    )


def create_test_source(
    source_id: str = "src_news_item",
    company_id: str = "comp_openai",
    workspace_id: str = "ws_primary",
    url: str = "https://news.ycombinator.com/item?id=1",
) -> Source:
    """Helper to construct a valid Source domain model."""
    return Source(
        identity=SourceIdentity(
            source_id=source_id,
            company_id=company_id,
            workspace_id=workspace_id,
            display_name="HN Story",
        ),
        classification=SourceClassification(
            category=SourceCategory.BLOG,
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
# SemanticChunker Unit Tests
# ==============================================================================


def test_chunker_is_instance_of_base_chunker() -> None:
    """Test that SemanticChunker implements BaseDocumentChunker interface."""
    provider = MockEmbeddingProvider()
    chunker = SemanticChunker(embedding_provider=provider)
    assert isinstance(chunker, BaseDocumentChunker)


def test_embedding_provider_dependency_injected_and_invoked() -> None:
    """Test that injected embedding provider is actively invoked during chunking."""
    spy_provider = SpyEmbeddingProvider()
    chunker = SemanticChunker(embedding_provider=spy_provider)
    version = create_test_version()

    chunks = chunker.chunk(version=version)

    assert len(chunks) >= 1
    assert len(spy_provider.call_history) >= 1
    # Verify texts were submitted to provider
    all_embedded_texts = [item for batch in spy_provider.call_history for item in batch]
    assert len(all_embedded_texts) > 0


def test_invalid_embedding_provider_rejected() -> None:
    """Test that constructing SemanticChunker without a valid provider raises TypeError."""
    with pytest.raises(TypeError, match="requires a valid BaseEmbeddingProvider"):
        SemanticChunker(embedding_provider=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="requires a valid BaseEmbeddingProvider"):
        SemanticChunker(embedding_provider="not_a_provider")  # type: ignore[arg-type]


def test_semantic_chunker_produces_valid_chunks() -> None:
    """Test that semantic chunker partitions text into ordered DocumentChunk records."""
    provider = MockEmbeddingProvider(dimension=16)
    chunker = SemanticChunker(embedding_provider=provider)
    version = create_test_version()

    chunks = chunker.chunk(version=version)

    assert len(chunks) >= 1
    for i, c in enumerate(chunks):
        assert isinstance(c, DocumentChunk)
        assert c.chunk_index == i
        assert c.document_id == version.document_id
        assert c.version_id == version.version_id
        assert c.metadata.custom_tags["strategy"] == "semantic"


def test_short_document_produces_single_chunk() -> None:
    """Test that content below chunk size with no semantic splits produces 1 chunk."""
    provider = MockEmbeddingProvider()
    chunker = SemanticChunker(embedding_provider=provider, default_chunk_size=1000)
    version = create_test_version(content="Short single sentence note.")

    chunks = chunker.chunk(version=version)

    assert len(chunks) == 1
    assert chunks[0].content == "Short single sentence note."
    assert chunks[0].chunk_index == 0


def test_configured_threshold_and_parameters_applied() -> None:
    """Test that breakpoint_percentile_threshold and chunk sizes are recorded and applied."""
    provider = MockEmbeddingProvider()
    chunker = SemanticChunker(embedding_provider=provider)
    version = create_test_version()
    params = ChunkingParameters(
        chunk_size=500,
        chunk_overlap=50,
        custom_params={
            "breakpoint_percentile_threshold": 80,
            "buffer_size": 1,
        },
    )

    chunks = chunker.chunk(version=version, parameters=params)

    for c in chunks:
        assert c.metadata.custom_tags["chunk_size"] == 500
        assert c.metadata.custom_tags["chunk_overlap"] == 50
        assert c.metadata.custom_tags["breakpoint_percentile_threshold"] == 80
        assert c.metadata.custom_tags["buffer_size"] == 1


def test_provenance_preservation_with_and_without_source() -> None:
    """Test that provenance fields are properly populated on all semantic chunks."""
    provider = MockEmbeddingProvider()
    chunker = SemanticChunker(embedding_provider=provider)
    version = create_test_version()
    source = create_test_source()

    chunks_with_src = chunker.chunk(version=version, source=source)
    for c in chunks_with_src:
        assert c.document_id == version.document_id
        assert c.version_id == version.version_id
        assert c.metadata.source_id == source.identity.source_id
        assert c.metadata.company_id == source.identity.company_id
        assert c.metadata.workspace_id == source.identity.workspace_id
        assert c.metadata.source_url == source.location.url_or_identifier
        assert c.metadata.source_category == "blog"

    chunks_no_src = chunker.chunk(version=version, source=None)
    for c in chunks_no_src:
        assert c.metadata.source_url == "https://news.ycombinator.com/item?id=1"


def test_deterministic_repeated_execution() -> None:
    """Test that same version and provider produce identical chunk IDs and hashes across runs."""
    provider = MockEmbeddingProvider()
    chunker = SemanticChunker(embedding_provider=provider)
    version = create_test_version(version_id="ver_stable_sem_999")
    params = ChunkingParameters(chunk_size=400, chunk_overlap=40)

    run1 = chunker.chunk(version=version, parameters=params)
    run2 = chunker.chunk(version=version, parameters=params)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=True):
        assert c1.chunk_id == c2.chunk_id
        assert c1.content == c2.content
        assert c1.content_hash == c2.content_hash
        assert c1.chunk_id == derive_semantic_chunk_id("ver_stable_sem_999", c1.chunk_index, 400, 40)


@pytest.mark.parametrize("empty_text", ["", "   ", "\t\n  \r\n"])
def test_empty_content_raises_value_error(empty_text: str) -> None:
    """Test that empty or whitespace-only content raises ValueError."""
    provider = MockEmbeddingProvider()
    chunker = SemanticChunker(embedding_provider=provider)
    version = create_test_version(content="dummy valid content")
    object.__setattr__(version, "content", empty_text)

    with pytest.raises(ValueError, match="Cannot chunk DocumentVersion with empty or whitespace-only content"):
        chunker.chunk(version=version)


def test_no_llamaindex_types_leak_into_document_chunk() -> None:
    """Verify that returned objects are strictly DocumentChunk without LlamaIndex node types."""
    provider = MockEmbeddingProvider()
    chunker = SemanticChunker(embedding_provider=provider)
    version = create_test_version()

    chunks = chunker.chunk(version=version)

    for c in chunks:
        assert type(c) is DocumentChunk
        assert not hasattr(c, "node")
        assert not hasattr(c, "text_node")
        assert isinstance(c.metadata.custom_tags, dict)


def test_no_concrete_embedding_model_imported() -> None:
    """Verify that semantic_chunker.py does not import concrete ML libraries."""
    import competitive_intelligence.knowledge_pipeline.chunkers.semantic_chunker as sem_mod

    mod_file = getattr(sem_mod, "__file__", "")
    assert mod_file != ""

    with open(mod_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "openai" not in code.lower() or "llama_index.embeddings.openai" not in code
    assert "sentence_transformers" not in code
    assert "torch" not in code
    assert "transformers" not in code
