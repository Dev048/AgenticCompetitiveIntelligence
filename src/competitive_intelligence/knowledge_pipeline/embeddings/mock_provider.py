"""Deterministic mock embedding provider for testing and validation."""

from collections.abc import Sequence
import hashlib

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseEmbeddingProvider,
)


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic mock embedding provider producing normalized pseudo-vectors.

    Generates reproducible unit-length vectors of fixed dimension based on input
    text hashes without downloading model weights or making external API calls.
    """

    def __init__(
        self,
        dimension: int = 8,
        model_name: str = "mock-embedder",
        model_version: str | None = "1.0.0",
    ) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._dimension = dimension
        self._model_name = model_name
        self._model_version = model_version

    @property
    def model_name(self) -> str:
        """Name of the mock embedding model."""
        return self._model_name

    @property
    def model_version(self) -> str | None:
        """Version of the mock embedding model."""
        return self._model_version

    @property
    def dimensions(self) -> int:
        """Vector dimensions produced by this mock provider."""
        return self._dimension

    @property
    def dimension(self) -> int:
        """Alias for dimensions for backwards compatibility."""
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute deterministic mock embedding vectors for a sequence of texts."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for text in texts:
            # Deterministic seed using SHA-256 hash
            raw_hash = hashlib.sha256(text.encode("utf-8")).digest()
            # Generate dimension values from hash bytes
            vec: list[float] = []
            for i in range(self.dimension):
                byte_val = raw_hash[i % len(raw_hash)]
                # Map to range [-1.0, 1.0]
                val = (byte_val / 127.5) - 1.0
                vec.append(val)

            # Normalize to unit vector
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            unit_vec = [round(x / norm, 6) for x in vec]
            embeddings.append(unit_vec)

        return embeddings
