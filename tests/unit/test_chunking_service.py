"""Unit tests for Knowledge Pipeline ChunkingService."""

from datetime import datetime, timezone
import pytest

from competitive_intelligence.knowledge_pipeline.chunking_service import (
    ChunkingService,
)
from competitive_intelligence.knowledge_pipeline.embeddings.mock_provider import (
    MockEmbeddingProvider,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkingParameters,
    ChunkingStrategy,
    DocumentCharacteristics,
    DocumentChunk,
    DocumentVersion,
    StructuralRichness,
)
from competitive_intelligence.knowledge_pipeline.normalizer import compute_content_hash
from competitive_intelligence.source_acquisition.models import (
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


def create_source(
    category: SourceCategory = SourceCategory.WEBSITE,
    content_type: ContentType = ContentType.HTML,
) -> Source:
    return Source(
        identity=SourceIdentity(
            source_id="src_test_1",
            company_id="comp_1",
            workspace_id="ws_1",
            display_name="Test Source",
        ),
        classification=SourceClassification(
            category=category,
            platform=PlatformType.WEB,
            content_type=content_type,
        ),
        location=SourceLocation(
            url_or_identifier="https://example.com/pricing",
            access_pattern=AccessPattern.PUBLIC_WEB,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
        ),
    )


def create_version(
    content: str,
    metadata: dict[str, object] | None = None,
) -> DocumentVersion:
    return DocumentVersion(
        version_id="ver_test_100",
        document_id="doc_test_100",
        content=content,
        title="Test Document Title",
        content_hash=compute_content_hash(content),
        acquired_at=datetime.now(timezone.utc),
        metadata=metadata or {},
    )


def test_short_content_dispatches_to_whole_document() -> None:
    """Test that documents shorter than threshold are chunked using WholeDocumentChunker."""
    service = ChunkingService()
    short_content = "Small document content about Qdrant pricing."
    version = create_version(short_content)
    source = create_source()

    chunks = service.chunk(version=version, source=source)

    assert len(chunks) == 1
    assert chunks[0].content == short_content
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_level == 0
    assert chunks[0].metadata.source_id == source.identity.source_id


def test_technical_documentation_dispatches_to_parent_child() -> None:
    """Test that high-structure technical content resolves to ParentChildChunker."""
    service = ChunkingService()
    tech_content = """# Architecture Overview
This is a comprehensive overview of the distributed vector storage engine.

## Vector Search Indexing
Detailed breakdown of HNSW graph construction and quantization parameters.
- Scalar Quantization: int8
- Product Quantization: 8x
| Parameter | Value | Description |
|---|---|---|
| ef_search | 128 | Search queue depth |
| M | 16 | Bi-directional links |

```python
def query_vector():
    return store.search()
```

## Clustering and Partitioning
Multi-tenant collection partitioning across worker shards.
""" + (" Extra context sentence for volumetric padding." * 20)

    version = create_version(
        tech_content,
        metadata={"source_category": "documentation", "content_type": "markdown"},
    )
    source = create_source(category=SourceCategory.DOCUMENTATION, content_type=ContentType.MARKDOWN)

    chunks = service.chunk(version=version, source=source)

    assert len(chunks) > 1
    # Check parent and child levels
    parent_chunks = [c for c in chunks if c.chunk_level == 0]
    child_chunks = [c for c in chunks if c.chunk_level == 1]
    assert len(parent_chunks) >= 1
    assert len(child_chunks) >= 1
    for child in child_chunks:
        assert child.parent_chunk_id is not None
        assert any(p.chunk_id == child.parent_chunk_id for p in parent_chunks)


def test_markdown_web_dispatches_to_recursive() -> None:
    """Test that web blog / pricing content dispatches to RecursiveChunker."""
    service = ChunkingService()
    long_content = """# Pricing & Plans

## Starter Tier
Cost is $49 per month including 5 seats and 10GB storage.

## Growth Tier
Cost is $199 per month including unlimited seats and 100GB storage.

## Enterprise Tier
Custom pricing with dedicated VPC and SLA guarantees.
""" + (" Volumetric padding paragraph with detailed features." * 30)

    version = create_version(long_content)
    source = create_source(category=SourceCategory.PRICING, content_type=ContentType.HTML)

    chunks = service.chunk(version=version, source=source)

    assert len(chunks) >= 1
    assert chunks[0].metadata.custom_tags.get("strategy") == ChunkingStrategy.RECURSIVE.value


def test_semantic_transitions_with_embedding_provider() -> None:
    """Test that explicit semantic transitions dispatch to SemanticChunker when embedder provided."""
    mock_embedder = MockEmbeddingProvider(dimension=8)
    service = ChunkingService(embedding_provider=mock_embedder)
    content = ("The first topic covers database architecture. " * 30) + "\n\n" + ("The second topic covers machine learning inference pipelines. " * 30)
    version = create_version(content, metadata={"has_semantic_transitions": True})

    chunks = service.chunk(version=version)

    assert len(chunks) >= 1
    assert chunks[0].metadata.custom_tags.get("strategy") == ChunkingStrategy.SEMANTIC.value


def test_semantic_transitions_without_embedding_provider_falls_back() -> None:
    """Test that semantic strategy safely falls back to RecursiveChunker when no embedder provided."""
    service = ChunkingService(embedding_provider=None)
    content = ("The first topic covers database architecture. " * 30) + "\n\n" + ("The second topic covers machine learning inference pipelines. " * 30)
    version = create_version(content, metadata={"has_semantic_transitions": True})

    chunks = service.chunk(version=version)

    assert len(chunks) >= 1
    assert chunks[0].metadata.custom_tags.get("strategy") in (
        ChunkingStrategy.RECURSIVE.value,
        ChunkingStrategy.FIXED.value,
    )


def test_empty_content_raises_value_error() -> None:
    """Test that empty or whitespace content raises ValueError."""
    service = ChunkingService()
    # 1. Pydantic validation on DocumentVersion instantiation
    with pytest.raises(ValueError, match="Document content cannot be empty"):
        create_version("")
    with pytest.raises(ValueError, match="Document content cannot be empty"):
        create_version("   \n\t  ")

    # 2. ChunkingService validation if an existing version object has content cleared
    valid_ver = create_version("Valid initial content")
    object.__setattr__(valid_ver, "content", "")
    with pytest.raises(ValueError, match="Cannot chunk DocumentVersion with empty"):
        service.chunk(valid_ver)


def test_custom_parameters_override_policy() -> None:
    """Test that caller-provided parameters override policy defaults."""
    service = ChunkingService()
    content = "Paragraph 1 with text content.\n\nParagraph 2 with more text.\n\nParagraph 3." * 50
    version = create_version(content)
    custom_params = ChunkingParameters(chunk_size=300, chunk_overlap=30)

    chunks = service.chunk(version=version, parameters=custom_params)

    assert len(chunks) > 1
    assert chunks[0].metadata.custom_tags.get("chunk_size") == 300
