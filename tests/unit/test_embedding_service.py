"""Unit tests for Knowledge Pipeline EmbeddingService.

Verifies chunk eligibility filtering (child-only parent-child policy), batching,
vector-to-chunk mapping, model metadata propagation, shape validation,
non-mutation of inputs, async execution, and strict isolation from concrete ML libraries.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseEmbeddingProvider,
    ChunkMetadata,
    DocumentChunk,
    EmbeddingRecord,
    EmbeddingService,
    MockEmbeddingProvider,
    compute_content_hash,
)


# ==============================================================================
# Helpers & Mocks
# ==============================================================================


class SpyBatchEmbeddingProvider(BaseEmbeddingProvider):
    """Spy provider tracking batches and text inputs while delegating to MockEmbeddingProvider."""

    def __init__(
        self,
        dimension: int = 8,
        model_name: str = "spy-model",
        model_version: str | None = "2.1.0",
    ) -> None:
        self._mock = MockEmbeddingProvider(
            dimension=dimension,
            model_name=model_name,
            model_version=model_version,
        )
        self.recorded_batches: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._mock.model_name

    @property
    def model_version(self) -> str | None:
        return self._mock.model_version

    @property
    def dimensions(self) -> int:
        return self._mock.dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.recorded_batches.append(list(texts))
        return self._mock.embed_texts(texts)


class FaultyCountEmbeddingProvider(BaseEmbeddingProvider):
    """Faulty provider returning wrong count of vectors."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        # Returns fewer vectors than requested
        return [[0.1] * 8 for _ in range(max(0, len(texts) - 1))]


class FaultyDimensionEmbeddingProvider(BaseEmbeddingProvider):
    """Faulty provider returning vectors with wrong dimensionality."""

    @property
    def dimensions(self) -> int:
        return 8

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        # Returns vectors with length 4 instead of 8
        return [[0.1] * 4 for _ in texts]


def create_test_chunk(
    chunk_id: str,
    content: str,
    chunk_index: int = 0,
    chunk_level: int = 0,
    parent_chunk_id: str | None = None,
    doc_id: str = "doc_test_1",
    version_id: str = "ver_test_1",
) -> DocumentChunk:
    """Helper to construct a valid DocumentChunk."""
    metadata = ChunkMetadata(
        source_id="src_test_1",
        document_id=doc_id,
        version_id=version_id,
        company_id="comp_test",
        workspace_id="ws_test",
        source_url="https://example.com/item",
        source_category="blog",
        collected_at=datetime.now(timezone.utc),
        start_char=0,
        end_char=len(content),
        chunk_index=chunk_index,
        total_chunks=1,
        language="en",
    )
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        version_id=version_id,
        content=content,
        chunk_index=chunk_index,
        metadata=metadata,
        content_hash=compute_content_hash(content),
        parent_chunk_id=parent_chunk_id,
        chunk_level=chunk_level,
        created_at=datetime.now(timezone.utc),
    )


# ==============================================================================
# Unit Tests
# ==============================================================================


def test_empty_input_returns_empty_list_and_never_calls_provider() -> None:
    """Test that passing an empty chunk list returns [] without invoking the provider."""
    spy = SpyBatchEmbeddingProvider()
    service = EmbeddingService(provider=spy)

    records = service.generate_embeddings([])

    assert records == []
    assert len(spy.recorded_batches) == 0


def test_single_chunk_embedding() -> None:
    """Test embedding generation for a single standalone DocumentChunk."""
    provider = MockEmbeddingProvider(dimension=12, model_name="test-model", model_version="1.0")
    service = EmbeddingService(provider=provider)
    chunk = create_test_chunk(chunk_id="chk_1", content="Competitor announced Q3 earnings.")

    records = service.generate_embeddings([chunk])

    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, EmbeddingRecord)
    assert rec.chunk_id == "chk_1"
    assert rec.document_id == "doc_test_1"
    assert rec.version_id == "ver_test_1"
    assert rec.model_name == "test-model"
    assert rec.model_version == "1.0"
    assert rec.dimensions == 12
    assert len(rec.vector) == 12
    assert rec.metadata["source_id"] == "src_test_1"
    assert rec.metadata["chunk_index"] == 0


def test_batching_splits_chunks_and_preserves_order() -> None:
    """Test that a large list of chunks is correctly batched according to batch_size."""
    spy = SpyBatchEmbeddingProvider(dimension=8)
    service = EmbeddingService(provider=spy, default_batch_size=3)

    chunks = [
        create_test_chunk(chunk_id=f"chk_{i}", content=f"Sentence number {i}", chunk_index=i)
        for i in range(7)
    ]

    records = service.generate_embeddings(chunks)

    # 7 chunks with batch_size=3 should produce 3 batches: 3, 3, 1
    assert len(spy.recorded_batches) == 3
    assert len(spy.recorded_batches[0]) == 3
    assert len(spy.recorded_batches[1]) == 3
    assert len(spy.recorded_batches[2]) == 1

    # Output records should match 7 input chunks in exact order
    assert len(records) == 7
    for i, rec in enumerate(records):
        assert rec.chunk_id == f"chk_{i}"
        assert rec.metadata["chunk_index"] == i


def test_batch_size_override() -> None:
    """Test overriding batch_size in generate_embeddings call."""
    spy = SpyBatchEmbeddingProvider(dimension=8)
    service = EmbeddingService(provider=spy, default_batch_size=10)

    chunks = [
        create_test_chunk(chunk_id=f"chk_{i}", content=f"Content {i}", chunk_index=i)
        for i in range(5)
    ]

    records = service.generate_embeddings(chunks, batch_size=2)

    # 5 chunks with batch_size=2 should produce 3 batches: 2, 2, 1
    assert len(spy.recorded_batches) == 3
    assert len(spy.recorded_batches[0]) == 2
    assert len(spy.recorded_batches[1]) == 2
    assert len(spy.recorded_batches[2]) == 1
    assert len(records) == 5


def test_invalid_batch_size_raises_value_error() -> None:
    """Test that non-positive batch_size raises ValueError."""
    provider = MockEmbeddingProvider()
    with pytest.raises(ValueError, match="default_batch_size must be a positive integer"):
        EmbeddingService(provider=provider, default_batch_size=0)

    service = EmbeddingService(provider=provider)
    chunk = create_test_chunk(chunk_id="chk_1", content="Some text")
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        service.generate_embeddings([chunk], batch_size=-1)


def test_invalid_provider_constructor_raises_type_error() -> None:
    """Test that constructing EmbeddingService with invalid provider raises TypeError."""
    with pytest.raises(TypeError, match="requires a valid BaseEmbeddingProvider instance"):
        EmbeddingService(provider=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="requires a valid BaseEmbeddingProvider"):
        EmbeddingService(provider="invalid_string")  # type: ignore[arg-type]


def test_provider_output_count_mismatch_raises_value_error() -> None:
    """Test that provider returning fewer/more vectors than requested raises ValueError."""
    faulty_provider = FaultyCountEmbeddingProvider()
    service = EmbeddingService(provider=faulty_provider)

    chunks = [
        create_test_chunk(chunk_id="chk_1", content="Text 1"),
        create_test_chunk(chunk_id="chk_2", content="Text 2"),
    ]

    with pytest.raises(ValueError, match="Provider output count mismatch"):
        service.generate_embeddings(chunks)


def test_vector_dimension_mismatch_raises_value_error() -> None:
    """Test that vectors not matching declared provider dimensions raise ValueError."""
    faulty_provider = FaultyDimensionEmbeddingProvider()
    service = EmbeddingService(provider=faulty_provider)

    chunk = create_test_chunk(chunk_id="chk_1", content="Text 1")

    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        service.generate_embeddings([chunk])


def test_parent_child_eligibility_policy_child_only_default() -> None:
    """Test that parent chunks with children are skipped by default while children are embedded."""
    provider = MockEmbeddingProvider(dimension=8)
    service = EmbeddingService(provider=provider)

    parent_chunk = create_test_chunk(
        chunk_id="parent_chk_1",
        content="Parent chunk containing entire section overview.",
        chunk_index=0,
        chunk_level=0,
        parent_chunk_id=None,
    )
    child_chunk_1 = create_test_chunk(
        chunk_id="child_chk_1",
        content="Child chunk paragraph 1.",
        chunk_index=1,
        chunk_level=1,
        parent_chunk_id="parent_chk_1",
    )
    child_chunk_2 = create_test_chunk(
        chunk_id="child_chk_2",
        content="Child chunk paragraph 2.",
        chunk_index=2,
        chunk_level=1,
        parent_chunk_id="parent_chk_1",
    )

    chunks = [parent_chunk, child_chunk_1, child_chunk_2]

    # Default policy: embed only children (parent is skipped)
    records = service.generate_embeddings(chunks)

    assert len(records) == 2
    embedded_ids = [r.chunk_id for r in records]
    assert "parent_chk_1" not in embedded_ids
    assert "child_chk_1" in embedded_ids
    assert "child_chk_2" in embedded_ids


def test_standalone_level_0_chunk_is_embedded() -> None:
    """Test that standalone level-0 chunks without child references are embedded."""
    provider = MockEmbeddingProvider(dimension=8)
    service = EmbeddingService(provider=provider)

    standalone_chunk_1 = create_test_chunk(
        chunk_id="standalone_1",
        content="Standalone document chunk A.",
        chunk_index=0,
        chunk_level=0,
        parent_chunk_id=None,
    )
    standalone_chunk_2 = create_test_chunk(
        chunk_id="standalone_2",
        content="Standalone document chunk B.",
        chunk_index=1,
        chunk_level=0,
        parent_chunk_id=None,
    )

    records = service.generate_embeddings([standalone_chunk_1, standalone_chunk_2])

    assert len(records) == 2
    assert records[0].chunk_id == "standalone_1"
    assert records[1].chunk_id == "standalone_2"


def test_embed_parents_with_children_override() -> None:
    """Test overriding parent-child policy to embed all chunks including parents."""
    provider = MockEmbeddingProvider(dimension=8)
    service = EmbeddingService(provider=provider, embed_parents_with_children=True)

    parent_chunk = create_test_chunk(
        chunk_id="parent_chk_1",
        content="Parent chunk overview.",
        chunk_index=0,
        chunk_level=0,
    )
    child_chunk = create_test_chunk(
        chunk_id="child_chk_1",
        content="Child chunk.",
        chunk_index=1,
        chunk_level=1,
        parent_chunk_id="parent_chk_1",
    )

    records = service.generate_embeddings([parent_chunk, child_chunk])

    assert len(records) == 2
    embedded_ids = [r.chunk_id for r in records]
    assert "parent_chk_1" in embedded_ids
    assert "child_chk_1" in embedded_ids


def test_input_chunks_are_never_mutated() -> None:
    """Test that input DocumentChunk objects remain completely unmodified."""
    provider = MockEmbeddingProvider(dimension=8)
    service = EmbeddingService(provider=provider)

    chunk = create_test_chunk(chunk_id="chk_immut", content="Original content text")
    original_dict = chunk.model_dump()

    _ = service.generate_embeddings([chunk])

    assert chunk.model_dump() == original_dict


@pytest.mark.anyio
async def test_async_generate_embeddings() -> None:
    """Test asynchronous agenerate_embeddings method."""
    spy = SpyBatchEmbeddingProvider(dimension=8)
    service = EmbeddingService(provider=spy, default_batch_size=2)

    chunks = [
        create_test_chunk(chunk_id=f"chk_async_{i}", content=f"Async content {i}", chunk_index=i)
        for i in range(3)
    ]

    records = await service.agenerate_embeddings(chunks)

    assert len(records) == 3
    assert len(spy.recorded_batches) == 2  # 2 batches: size 2, size 1
    assert records[0].chunk_id == "chk_async_0"
    assert records[1].chunk_id == "chk_async_1"
    assert records[2].chunk_id == "chk_async_2"


def test_no_concrete_ml_libraries_imported() -> None:
    """Verify that embedding_service.py does not import concrete ML or database libraries."""
    import competitive_intelligence.knowledge_pipeline.embeddings.embedding_service as srv_mod

    mod_file = getattr(srv_mod, "__file__", "")
    assert mod_file != ""

    with open(mod_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "openai" not in code
    assert "sentence_transformers" not in code
    assert "torch" not in code
    assert "transformers" not in code
    assert "qdrant" not in code
