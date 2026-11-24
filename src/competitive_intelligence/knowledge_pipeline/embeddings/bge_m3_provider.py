"""BGE-M3 concrete embedding provider for Knowledge Pipeline subsystem.

Production provider implementing BaseEmbeddingProvider using SentenceTransformers with BAAI/bge-m3.
Generates 1024-dimensional dense L2-normalized embeddings on CPU or GPU.

Architecture Note:
- BAAI/bge-m3 provides 1024-dimensional dense representations, multi-lingual support,
  and up to 8192 token input context windows.
- Local CPU inference is supported for development and functional validation (~540ms/query, ~1s/chunk).
- Production GPU deployment can use the exact same provider with device="cuda".
- Unit tests use model injection (model=fake_model) to avoid downloading 2.2GB model weights.
"""

from collections.abc import Sequence
from typing import Any

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseEmbeddingProvider,
)

DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MODEL_VERSION = "1.0.0"
BGE_M3_DENSE_DIMENSIONS = 1024


class EmbeddingProviderError(Exception):
    """Exception raised for failures in embedding provider initialization or encoding."""


class BGEM3EmbeddingProvider(BaseEmbeddingProvider):
    """Concrete embedding provider for BAAI/bge-m3 using SentenceTransformers.

    Responsibilities:
    - Loads and maintains a single instance of the BAAI/bge-m3 model.
    - Encodes batches of text into 1024-dimensional dense float vectors.
    - Applies L2 normalization by default for cosine/dot-product retrieval.
    - Supports configurable compute devices (CPU, CUDA, MPS).
    - Preserves provider independence without leaking ML framework types outside.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize_embeddings: bool = True,
        model_version: str | None = DEFAULT_MODEL_VERSION,
        model: Any = None,
    ) -> None:
        """Initialize BGEM3EmbeddingProvider.

        Args:
            model_name: HuggingFace model identifier (default: 'BAAI/bge-m3').
            device: Compute device ('cpu', 'cuda', 'mps').
            batch_size: Default batch size for chunk encoding.
            normalize_embeddings: Whether to L2-normalize vectors for retrieval.
            model_version: Optional version identifier for model provenance.
            model: Optional pre-instantiated or mock model instance for testing.

        Raises:
            ValueError: If batch_size is non-positive.
            EmbeddingProviderError: If model initialization fails.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty or whitespace-only")

        self._model_name = model_name.strip()
        self._device = device.strip() if device else "cpu"
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._model_version = model_version

        if model is not None:
            self._model = model
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name, device=self._device)
            except Exception as exc:
                raise EmbeddingProviderError(
                    f"Failed to load embedding model '{self._model_name}' on device '{self._device}': {exc}"
                ) from exc

    @property
    def model_name(self) -> str:
        """Name of the underlying embedding model."""
        return self._model_name

    @property
    def model_version(self) -> str | None:
        """Version identifier of the embedding model."""
        return self._model_version

    @property
    def dimensions(self) -> int:
        """Dimensionality of the produced dense embeddings (1024 for BGE-M3)."""
        return BGE_M3_DENSE_DIMENSIONS

    @property
    def device(self) -> str:
        """Current compute device ('cpu', 'cuda', etc.)."""
        return self._device

    @property
    def batch_size(self) -> int:
        """Configured batch size for inference."""
        return self._batch_size

    @property
    def normalize_embeddings(self) -> bool:
        """Whether vector normalization is enabled."""
        return self._normalize_embeddings

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute dense vector embeddings for a sequence of texts.

        Args:
            texts: Sequence of string texts to embed.

        Returns:
            A list of float vectors, exactly one per input text with length 1024.

        Raises:
            EmbeddingProviderError: If model encoding fails or returns invalid dimensions.
        """
        if not texts:
            return []

        text_list = list(texts)

        try:
            raw_embeddings = self._model.encode(
                text_list,
                batch_size=self._batch_size,
                device=self._device,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Model encoding failed for batch of {len(text_list)} texts: {exc}"
            ) from exc

        if len(raw_embeddings) != len(text_list):
            raise EmbeddingProviderError(
                f"Model returned {len(raw_embeddings)} vectors, expected {len(text_list)}"
            )

        embeddings: list[list[float]] = []
        for i, vec in enumerate(raw_embeddings):
            if vec is None:
                raise EmbeddingProviderError(f"Model returned None vector at index {i}")

            float_vec = [float(x) for x in vec]
            if len(float_vec) != self.dimensions:
                raise EmbeddingProviderError(
                    f"Vector dimension mismatch at index {i}: expected {self.dimensions}, got {len(float_vec)}"
                )
            embeddings.append(float_vec)

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Compute dense vector embedding for a single search query.

        Args:
            text: Query string.

        Returns:
            A 1024-dimensional float vector.

        Raises:
            ValueError: If query text is empty or whitespace-only.
            EmbeddingProviderError: If encoding fails.
        """
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty or whitespace-only")

        results = self.embed_texts([text.strip()])
        if not results:
            raise EmbeddingProviderError("Model returned empty result for query")
        return results[0]

    async def aembed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Asynchronously compute dense vector embeddings for a sequence of texts."""
        return self.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously compute dense vector embedding for a single search query."""
        return self.embed_query(text)
