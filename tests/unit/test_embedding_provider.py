"""Unit tests for Knowledge Pipeline EmbeddingProvider abstraction and MockEmbeddingProvider.

Verifies provider-agnostic interface compliance, batch and single query embedding,
deterministic pseudo-vector generation, edge cases, and strict library isolation.
"""

from collections.abc import Sequence

import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseEmbeddingProvider,
    EmbeddingProvider,
    MockEmbeddingProvider,
)


# ==============================================================================
# Interface Compliance & Mock Provider Tests
# ==============================================================================


def test_mock_provider_implements_interface() -> None:
    """Test that MockEmbeddingProvider implements BaseEmbeddingProvider."""
    provider = MockEmbeddingProvider()
    assert isinstance(provider, BaseEmbeddingProvider)
    assert isinstance(provider, EmbeddingProvider)
    assert EmbeddingProvider is BaseEmbeddingProvider


def test_batch_embedding_shape_and_dimension() -> None:
    """Test that embedding a batch of N texts returns N vectors of length dimension."""
    provider = MockEmbeddingProvider(dimension=16)
    texts = [
        "Competitor launched new enterprise tier.",
        "Pricing increased from $20 to $30.",
        "New SOC2 compliance certification added.",
    ]

    embeddings = provider.embed_texts(texts)

    assert len(embeddings) == len(texts)
    for vec in embeddings:
        assert isinstance(vec, list)
        assert len(vec) == 16
        assert all(isinstance(val, float) for val in vec)
        # Verify unit normalization (length approximately 1.0)
        norm = sum(x * x for x in vec) ** 0.5
        assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_alias_matches_embed_texts() -> None:
    """Test that embed() convenience alias produces identical results to embed_texts()."""
    provider = MockEmbeddingProvider(dimension=8)
    texts = ["Feature X was deprecated.", "API rate limit updated."]

    res1 = provider.embed_texts(texts)
    res2 = provider.embed(texts)

    assert res1 == res2


def test_embed_query_returns_single_vector() -> None:
    """Test embed_query returns a single float vector of correct dimension."""
    provider = MockEmbeddingProvider(dimension=12)
    query = "enterprise pricing"

    vec = provider.embed_query(query)

    assert isinstance(vec, list)
    assert len(vec) == 12
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.anyio
async def test_async_embedding_methods() -> None:
    """Test aembed_texts and aembed_query async variants."""
    provider = MockEmbeddingProvider(dimension=8)
    texts = ["Async test sentence one.", "Async test sentence two."]

    batch_vecs = await provider.aembed_texts(texts)
    query_vec = await provider.aembed_query("Async query")

    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 8
    assert len(query_vec) == 8


def test_empty_input_returns_empty_list() -> None:
    """Test that passing an empty list of texts returns an empty list."""
    provider = MockEmbeddingProvider()

    assert provider.embed_texts([]) == []
    assert provider.embed([]) == []


def test_deterministic_embedding_generation() -> None:
    """Test that same text produces identical embedding vector across calls."""
    provider = MockEmbeddingProvider(dimension=8)
    text = "Deterministic vector generation test."

    vec1 = provider.embed_query(text)
    vec2 = provider.embed_query(text)

    assert vec1 == vec2


def test_distinct_inputs_produce_distinct_vectors() -> None:
    """Test that different input strings produce different embedding vectors."""
    provider = MockEmbeddingProvider(dimension=8)

    vec1 = provider.embed_query("Security and compliance")
    vec2 = provider.embed_query("Pricing and billing models")

    assert vec1 != vec2


@pytest.mark.parametrize("invalid_dim", [0, -1, -10])
def test_invalid_dimension_raises_value_error(invalid_dim: int) -> None:
    """Test that non-positive dimension raises ValueError."""
    with pytest.raises(ValueError, match="Embedding dimension must be positive"):
        MockEmbeddingProvider(dimension=invalid_dim)


class CustomFakeProvider(BaseEmbeddingProvider):
    """Custom fake provider for testing domain-level dependency injection."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_custom_provider_can_implement_interface() -> None:
    """Test that any custom class implementing BaseEmbeddingProvider functions seamlessly."""
    custom = CustomFakeProvider()
    res = custom.embed(["Sample A", "Sample B"])

    assert len(res) == 2
    assert res[0] == [1.0, 0.0]
    assert res[1] == [1.0, 0.0]


def test_no_concrete_embedding_libraries_imported() -> None:
    """Verify that domain interfaces and embeddings module do not import heavy ML libraries."""
    import competitive_intelligence.knowledge_pipeline.embeddings.mock_provider as mock_mod
    import competitive_intelligence.knowledge_pipeline.interfaces as iface_mod

    for mod in (iface_mod, mock_mod):
        mod_file = getattr(mod, "__file__", "")
        assert mod_file != ""
        with open(mod_file, "r", encoding="utf-8") as f:
            code = f.read()

        assert "openai" not in code
        assert "sentence_transformers" not in code
        assert "torch" not in code
        assert "transformers" not in code
