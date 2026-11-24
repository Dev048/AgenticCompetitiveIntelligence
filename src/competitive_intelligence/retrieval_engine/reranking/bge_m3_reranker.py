"""Concrete implementation of BaseReranker using BAAI/bge-reranker-v2-m3."""

import logging
from typing import Any, Sequence

from competitive_intelligence.retrieval_engine.interfaces import BaseReranker
from competitive_intelligence.retrieval_engine.models import RetrievalResult
from competitive_intelligence.retrieval_engine.reranking.errors import BGERerankerError

logger = logging.getLogger(__name__)


class BGEM3Reranker(BaseReranker):
    """Reranker adapter using BAAI/bge-reranker-v2-m3 via SentenceTransformers."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 32,
        cross_encoder: Any = None,
    ) -> None:
        """Initialize the BGE M3 reranker.

        Args:
            model_name: The HF model identifier.
            device: Device to load the model on (cpu, cuda, mps).
            batch_size: Number of pairs to score at once.
            cross_encoder: Optional injected model instance (for testing).
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = cross_encoder

    def _get_model(self) -> Any:
        """Lazily load the SentenceTransformers CrossEncoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading BGE reranker model %s on %s", self.model_name, self.device)
                self._model = CrossEncoder(self.model_name, device=self.device, max_length=512)
            except ImportError as e:
                raise BGERerankerError("Failed to import sentence_transformers.") from e
            except Exception as e:
                raise BGERerankerError(f"Failed to initialize model {self.model_name}: {e}") from e
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: The search query text.
            candidates: Sequence of retrieved candidate results.
            top_k: Optional limit on the number of results to return.

        Returns:
            A new list of RetrievalResult objects sorted descending by reranker_score.
            Empty queries or empty candidate lists are handled gracefully.

        Raises:
            ValueError: If the query is completely empty.
            BGERerankerError: If model scoring fails.
        """
        query_stripped = query.strip()
        if not query_stripped:
            raise ValueError("Query string cannot be empty or whitespace.")

        if not candidates:
            return []

        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")

        model = self._get_model()

        # Build query-candidate pairs for scoring
        pairs = [(query_stripped, c.text) for c in candidates]

        try:
            # model.predict returns a numpy array or a list of scores
            scores = model.predict(pairs, batch_size=self.batch_size)
        except Exception as e:
            raise BGERerankerError(f"Failed to score candidates: {e}") from e

        # Construct new RetrievalResult objects to preserve immutability
        reranked_results: list[RetrievalResult] = []
        for idx, (candidate, score) in enumerate(zip(candidates, scores)):
            # Create a shallow copy of the candidate data
            new_candidate = candidate.model_copy()
            
            # Record original position (1-indexed)
            new_candidate.original_rank = idx + 1
            
            # Add reranking score (float conversion ensures type safety from numpy)
            new_candidate.reranker_score = float(score)
            new_candidate.reranker_model = self.model_name
            
            # Note: original candidate.score is strictly preserved unmodified
            
            reranked_results.append(new_candidate)

        # Sort descending by reranker_score, tie-breaking on chunk_id for determinism
        reranked_results.sort(key=lambda x: (x.reranker_score or 0.0, x.chunk_id), reverse=True)

        # Apply top_k slicing if requested
        if top_k is not None:
            return reranked_results[:top_k]
        
        return reranked_results
