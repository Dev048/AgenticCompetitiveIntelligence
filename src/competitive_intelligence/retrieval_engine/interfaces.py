"""Abstract interfaces for Retrieval Engine subsystem.

NOTE: These interfaces are provisional placeholders for initial architectural scaffolding,
not finalized domain contracts. They will be refined during Subsystem 3 detailed design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from competitive_intelligence.retrieval_engine.models import (
    RetrievalEvaluationMetrics,
    RetrievalQuery,
    RetrievalResult,
    ScoredDocumentChunk,
)

if TYPE_CHECKING:
    from competitive_intelligence.retrieval_engine.routing.models import RoutingDecision


class BaseRetrievalRouter(ABC):
    """Abstract interface for determining query retrieval strategy and budget."""

    @abstractmethod
    def route(self, query: str | RetrievalQuery) -> RoutingDecision:
        """Analyze query and return a deterministic RoutingDecision."""
        pass


class BaseRetriever(ABC):
    """Abstract interface for a specific retrieval strategy (dense, BM25, etc.)."""

    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Execute single-strategy retrieval."""
        pass


class BaseRankFusion(ABC):
    """Abstract interface for combining multiple ranked result sets (e.g. RRF)."""

    @abstractmethod
    def fuse(
        self,
        ranked_lists: list[list[ScoredDocumentChunk]],
        top_k: int = 20,
    ) -> list[ScoredDocumentChunk]:
        """Combine and recalculate ranks from multiple retriever outputs."""
        pass


class BaseReranker(ABC):
    """Abstract interface for cross-encoder or LLM-based rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidate chunks by relevance to query."""
        pass


class BaseRetrievalEngine(ABC):
    """Abstract orchestrator interface for full hybrid search + rerank pipeline."""

    @abstractmethod
    async def execute_retrieval(
        self,
        query: RetrievalQuery,
        rerank_top_n: int = 10,
    ) -> RetrievalResult:
        """Run dense/lexical search -> fuse ranks -> rerank -> return top scored chunks."""
        pass


class BaseRetrievalEvaluator(ABC):
    """Abstract interface for computing IR metrics against a benchmark dataset."""

    @abstractmethod
    async def evaluate(
        self,
        test_dataset_path: str,
        retriever: BaseRetriever,
        k_values: list[int] | None = None,
    ) -> RetrievalEvaluationMetrics:
        """Benchmark Recall@K, MRR, and NDCG."""
        pass
