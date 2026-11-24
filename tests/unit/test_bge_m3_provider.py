"""Unit tests for BGEM3EmbeddingProvider in Knowledge Pipeline subsystem.

Uses dependency injection of fake SentenceTransformers models to verify interface compliance,
metadata exposure, batching, query encoding, shape validation, error handling,
and strict dependency isolation without downloading 2.2GB model weights.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseEmbeddingProvider,
    BGEM3EmbeddingProvider,
    EmbeddingProvider,
    EmbeddingProviderError,
)


# ==============================================================================
# Mock / Fake SentenceTransformer Model
# ==============================================================================


class FakeSentenceTransformerModel:
    """Mock model simulating SentenceTransformer for fast, offline unit testing."""

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension
        self.call_history: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray[Any, Any]:
        self.call_history.append({
            "sentences": sentences,
            "batch_size": batch_size,
            "device": device,
            "normalize_embeddings": normalize_embeddings,
        })
        # Generate deterministic pseudo-vectors of requested dimension
        vectors: list[list[float]] = []
        for text in sentences:
            seed = sum(ord(c) for c in text) if text else 1
            vec = [(seed + i * 7) % 100 / 100.0 for i in range(self.dimension)]
            if normalize_embeddings:
                norm = sum(x * x for x in vec) ** 0.5 or 1.0
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


class FaultyDimensionModel:
    """Fake model returning wrong vector dimensionality."""

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray[Any, Any]:
        # Returns 512 dimensions instead of 1024
        return np.zeros((len(sentences), 512), dtype=np.float32)


class FaultyCountModel:
    """Fake model returning count mismatch."""

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray[Any, Any]:
        return np.zeros((len(sentences) - 1, 1024), dtype=np.float32)


class CrashingModel:
    """Fake model raising runtime exception during inference."""

    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray[Any, Any]:
        raise RuntimeError("CUDA device out of memory or compute fault")


# ==============================================================================
# Unit Tests
# ==============================================================================


def test_provider_implements_interface() -> None:
    """Test that BGEM3EmbeddingProvider implements BaseEmbeddingProvider."""
    fake_model = FakeSentenceTransformerModel()
    provider = BGEM3EmbeddingProvider(model=fake_model)

    assert isinstance(provider, BaseEmbeddingProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_name == "BAAI/bge-m3"
    assert provider.model_version == "1.0.0"
    assert provider.dimensions == 1024
    assert provider.device == "cpu"
    assert provider.batch_size == 32
    assert provider.normalize_embeddings is True


def test_custom_configuration() -> None:
    """Test provider with custom device, batch size, and version."""
    fake_model = FakeSentenceTransformerModel()
    provider = BGEM3EmbeddingProvider(
        model_name="custom/bge-m3",
        device="cuda:0",
        batch_size=16,
        normalize_embeddings=False,
        model_version="2.0.0",
        model=fake_model,
    )

    assert provider.model_name == "custom/bge-m3"
    assert provider.device == "cuda:0"
    assert provider.batch_size == 16
    assert provider.normalize_embeddings is False
    assert provider.model_version == "2.0.0"


def test_batch_embedding_shape_and_values() -> None:
    """Test embedding a sequence of texts returns 1024-dim float vectors."""
    fake_model = FakeSentenceTransformerModel(dimension=1024)
    provider = BGEM3EmbeddingProvider(model=fake_model)

    texts = [
        "Competitor launched new enterprise tier.",
        "Pricing increased from $20 to $30.",
        "Technical documentation for API architecture.",
    ]

    embeddings = provider.embed_texts(texts)

    assert len(embeddings) == len(texts)
    for vec in embeddings:
        assert isinstance(vec, list)
        assert len(vec) == 1024
        assert all(isinstance(x, float) for x in vec)

    # Verify model call parameters
    assert len(fake_model.call_history) == 1
    call = fake_model.call_history[0]
    assert call["sentences"] == texts
    assert call["batch_size"] == 32
    assert call["device"] == "cpu"
    assert call["normalize_embeddings"] is True


def test_embed_alias_matches_embed_texts() -> None:
    """Test that embed() alias produces identical results to embed_texts()."""
    fake_model = FakeSentenceTransformerModel()
    provider = BGEM3EmbeddingProvider(model=fake_model)
    texts = ["Sample document chunk 1", "Sample document chunk 2"]

    res1 = provider.embed_texts(texts)
    res2 = provider.embed(texts)

    assert res1 == res2


def test_embed_query_returns_single_vector() -> None:
    """Test that embed_query returns a single 1024-dim float vector."""
    fake_model = FakeSentenceTransformerModel()
    provider = BGEM3EmbeddingProvider(model=fake_model)

    vec = provider.embed_query("enterprise pricing models")

    assert isinstance(vec, list)
    assert len(vec) == 1024
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.parametrize("empty_query", ["", "   ", "\t\n  \r\n"])
def test_empty_query_raises_value_error(empty_query: str) -> None:
    """Test that empty or whitespace-only query string raises ValueError."""
    fake_model = FakeSentenceTransformerModel()
    provider = BGEM3EmbeddingProvider(model=fake_model)

    with pytest.raises(ValueError, match="Query text cannot be empty or whitespace-only"):
        provider.embed_query(empty_query)


def test_empty_batch_returns_empty_list_and_never_calls_model() -> None:
    """Test that empty batch input returns [] without invoking model.encode()."""
    fake_model = FakeSentenceTransformerModel()
    provider = BGEM3EmbeddingProvider(model=fake_model)

    assert provider.embed_texts([]) == []
    assert provider.embed([]) == []
    assert len(fake_model.call_history) == 0


@pytest.mark.anyio
async def test_async_embedding_methods() -> None:
    """Test aembed_texts and aembed_query async methods."""
    fake_model = FakeSentenceTransformerModel()
    provider = BGEM3EmbeddingProvider(model=fake_model)

    batch_res = await provider.aembed_texts(["Async sentence 1", "Async sentence 2"])
    query_res = await provider.aembed_query("Async search query")

    assert len(batch_res) == 2
    assert len(batch_res[0]) == 1024
    assert len(query_res) == 1024


def test_invalid_dimension_raises_provider_error() -> None:
    """Test that model returning wrong vector dimensionality raises EmbeddingProviderError."""
    faulty_model = FaultyDimensionModel()
    provider = BGEM3EmbeddingProvider(model=faulty_model)

    with pytest.raises(EmbeddingProviderError, match="Vector dimension mismatch"):
        provider.embed_texts(["Sample text"])


def test_count_mismatch_raises_provider_error() -> None:
    """Test that model returning vector count mismatch raises EmbeddingProviderError."""
    faulty_model = FaultyCountModel()
    provider = BGEM3EmbeddingProvider(model=faulty_model)

    with pytest.raises(EmbeddingProviderError, match="Model returned 1 vectors, expected 2"):
        provider.embed_texts(["Text A", "Text B"])


def test_model_runtime_error_wrapped_in_provider_error() -> None:
    """Test that runtime exceptions in model.encode() are cleanly wrapped in EmbeddingProviderError."""
    crashing_model = CrashingModel()
    provider = BGEM3EmbeddingProvider(model=crashing_model)

    with pytest.raises(EmbeddingProviderError, match="Model encoding failed for batch"):
        provider.embed_texts(["Crashing text"])


@pytest.mark.parametrize("invalid_batch_size", [0, -1, -10])
def test_invalid_batch_size_raises_value_error(invalid_batch_size: int) -> None:
    """Test that non-positive batch size raises ValueError."""
    fake_model = FakeSentenceTransformerModel()
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        BGEM3EmbeddingProvider(batch_size=invalid_batch_size, model=fake_model)


@pytest.mark.parametrize("invalid_model_name", ["", "   "])
def test_invalid_model_name_raises_value_error(invalid_model_name: str) -> None:
    """Test that empty model name raises ValueError."""
    fake_model = FakeSentenceTransformerModel()
    with pytest.raises(ValueError, match="model_name cannot be empty or whitespace-only"):
        BGEM3EmbeddingProvider(model_name=invalid_model_name, model=fake_model)


def test_model_instance_reused_across_calls() -> None:
    """Test that the injected model instance is reused across multiple embedding calls."""
    fake_model = FakeSentenceTransformerModel()
    provider = BGEM3EmbeddingProvider(model=fake_model)

    _ = provider.embed_texts(["First call"])
    _ = provider.embed_texts(["Second call"])
    _ = provider.embed_query("Query call")

    assert len(fake_model.call_history) == 3
    assert provider._model is fake_model


def test_domain_interfaces_remain_free_of_sentence_transformers_imports() -> None:
    """Verify that domain interfaces and models do not import sentence_transformers or torch."""
    import competitive_intelligence.knowledge_pipeline.interfaces as iface_mod
    import competitive_intelligence.knowledge_pipeline.models as model_mod

    for mod in (iface_mod, model_mod):
        mod_file = getattr(mod, "__file__", "")
        assert mod_file != ""
        with open(mod_file, "r", encoding="utf-8") as f:
            code = f.read()

        assert "sentence_transformers" not in code
        assert "SentenceTransformer" not in code
        assert "torch" not in code
