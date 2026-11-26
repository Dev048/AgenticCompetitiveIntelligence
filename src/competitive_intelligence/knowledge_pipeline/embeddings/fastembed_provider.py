"""Lightweight ONNX embedding provider (fastembed) for the free-tier demo and development.

Runs BAAI/bge-small-en-v1.5 (384 dimensions) through onnxruntime: no torch and no
sentence-transformers, so the whole demo process fits in a 512 MB container. Production
and staging must use BGE-M3; settings validation rejects this provider there.

``fastembed`` is imported lazily so that importing this module (and the API wiring that
references it) never pulls in onnxruntime for deployments that do not use it.
"""

from collections.abc import Iterable, Sequence
from typing import Any

from competitive_intelligence.knowledge_pipeline.interfaces import BaseEmbeddingProvider

DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedProviderError(Exception):
    """Raised when the fastembed model cannot be loaded or fails to embed."""


def fastembed_dimensions(model_name: str) -> int:
    """Vector size of a fastembed model, read from fastembed's model registry (no download)."""
    try:
        from fastembed import TextEmbedding
    except ImportError as err:  # pragma: no cover - exercised only without the dependency
        raise FastEmbedProviderError("fastembed is not installed (pip install fastembed).") from err
    for description in TextEmbedding.list_supported_models():
        if description["model"] == model_name:
            return int(description["dim"])
    raise FastEmbedProviderError(f"Unknown fastembed model: {model_name}")


class FastEmbedEmbeddingProvider(BaseEmbeddingProvider):
    """Dense, L2-normalized embeddings from a fastembed ONNX model on CPU."""

    def __init__(
        self,
        model_name: str = DEFAULT_FASTEMBED_MODEL,
        threads: int | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as err:
            raise FastEmbedProviderError(
                "fastembed is not installed (pip install fastembed)."
            ) from err
        self._model_name = model_name
        self._batch_size = batch_size
        self._dimensions = fastembed_dimensions(model_name)
        try:
            # Model files come from FASTEMBED_CACHE_PATH (baked into the demo image); with
            # HF_HUB_OFFLINE=1 nothing is downloaded at runtime.
            self._model: Any = TextEmbedding(model_name=model_name, threads=threads)
        except Exception as err:
            raise FastEmbedProviderError(f"Failed to load fastembed model {model_name}") from err

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._to_lists(self._model.passage_embed(list(texts), batch_size=self._batch_size))

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty or whitespace-only")
        vectors = self._to_lists(self._model.query_embed(text.strip()))
        if not vectors:
            raise FastEmbedProviderError("Model returned empty result for query")
        return vectors[0]

    @staticmethod
    def _to_lists(vectors: Iterable[Any]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in vectors]
