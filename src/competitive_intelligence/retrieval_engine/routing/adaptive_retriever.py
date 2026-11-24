"""Adaptive Retriever execution coordinator for Adaptive Retrieval.

Coordinates query routing via BaseRetrievalRouter and delegates execution to the
appropriate downstream retrieval pipeline based on the deterministic RoutingDecision.
"""

import time

from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.core.metrics import track_retrieval
from competitive_intelligence.retrieval_engine.interfaces import (
    BaseRetrievalRouter,
    BaseRetriever,
)

logger = get_logger(__name__)
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
)
from competitive_intelligence.retrieval_engine.reranking.reranked_retriever import (
    RerankedRetriever,
)
from competitive_intelligence.retrieval_engine.routing.errors import (
    AdaptiveRetrievalError,
)
from competitive_intelligence.retrieval_engine.routing.models import (
    RoutingDecision,
)


class AdaptiveRetriever(BaseRetriever):
    """Execution coordinator that routes queries and delegates to appropriate retriever pipelines.

    Responsibilities:
    - Implements the standard application-level BaseRetriever contract (RetrievalQuery -> list[RetrievalResult]).
    - Invokes the injected BaseRetrievalRouter to obtain a deterministic RoutingDecision.
    - Resolves the target retriever based on decision.strategy and decision.use_reranker.
    - Enforces decision budget constraints (candidate_pool_size, final_top_k) during execution.
    - Preserves input query immutability and user filter constraints.
    - Enriches output RetrievalResult metadata with routing provenance.
    - Encapsulates routing and execution errors in AdaptiveRetrievalError.
    """

    def __init__(
        self,
        router: BaseRetrievalRouter,
        bm25_retriever: BaseRetriever | None = None,
        semantic_retriever: BaseRetriever | None = None,
        hybrid_retriever: BaseRetriever | None = None,
        reranked_retriever: BaseRetriever | None = None,
    ) -> None:
        """Initialize the AdaptiveRetriever with injected router and candidate retrievers.

        Args:
            router: Deterministic or rule-based query router.
            bm25_retriever: Retriever pipeline for SPARSE_BM25 strategy.
            semantic_retriever: Retriever pipeline for DENSE_VECTOR strategy.
            hybrid_retriever: Retriever pipeline for HYBRID_RRF strategy (un-reranked).
            reranked_retriever: Retriever pipeline for HYBRID_RRF strategy with reranking.
        """
        self._router = router
        self._bm25_retriever = bm25_retriever
        self._semantic_retriever = semantic_retriever
        self._hybrid_retriever = hybrid_retriever
        self._reranked_retriever = reranked_retriever
        self._last_decision: RoutingDecision | None = None

    @property
    def router(self) -> BaseRetrievalRouter:
        """Configured query router."""
        return self._router

    @property
    def bm25_retriever(self) -> BaseRetriever | None:
        """Configured BM25 retriever."""
        return self._bm25_retriever

    @property
    def semantic_retriever(self) -> BaseRetriever | None:
        """Configured Semantic vector retriever."""
        return self._semantic_retriever

    @property
    def hybrid_retriever(self) -> BaseRetriever | None:
        """Configured un-reranked Hybrid retriever."""
        return self._hybrid_retriever

    @property
    def reranked_retriever(self) -> BaseRetriever | None:
        """Configured Reranked retriever."""
        return self._reranked_retriever

    @property
    def last_decision(self) -> RoutingDecision | None:
        """The most recent RoutingDecision made during retrieval."""
        return self._last_decision

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Route query and execute the selected retrieval pipeline.

        Args:
            query: Input RetrievalQuery containing text, filters, and bounds.

        Returns:
            List of RetrievalResult objects enriched with routing provenance metadata.

        Raises:
            AdaptiveRetrievalError: If routing fails, target retriever is unconfigured, or retrieval fails.
        """
        if not isinstance(query, RetrievalQuery):
            raise TypeError(f"query must be an instance of RetrievalQuery, got {type(query)}")

        t0 = time.perf_counter()

        # 1. Obtain deterministic RoutingDecision from router
        try:
            decision = self._router.route(query)
            self._last_decision = decision
        except Exception as e:
            track_retrieval(
                route="unknown",
                mode="adaptive",
                status="failure",
                duration_seconds=time.perf_counter() - t0,
            )
            raise AdaptiveRetrievalError(
                f"Query routing failed for query '{query.query_text}': {e}"
            ) from e

        # 2. Resolve target retriever pipeline
        target_retriever = self._resolve_retriever(decision)

        # 3. Construct downstream query derived from decision without mutating input query
        downstream_query = query.model_copy(
            update={
                "strategy": decision.strategy,
                "top_k": decision.final_top_k,
            }
        )

        # 4. Execute downstream retrieval with candidate pool budget propagation
        try:
            if (
                decision.use_reranker
                and isinstance(target_retriever, RerankedRetriever)
                and target_retriever.candidate_pool_size != decision.candidate_pool_size
            ):
                # Dynamically scope RerankedRetriever with decision candidate_pool_size
                scoped_reranked = RerankedRetriever(
                    upstream_retriever=target_retriever.upstream_retriever,
                    reranker=target_retriever.reranker,
                    candidate_pool_size=decision.candidate_pool_size,
                )
                results = await scoped_reranked.retrieve(downstream_query)
            else:
                results = await target_retriever.retrieve(downstream_query)
        except Exception as e:
            track_retrieval(
                route=decision.strategy.value,
                mode="adaptive",
                status="failure",
                duration_seconds=time.perf_counter() - t0,
                reranked=decision.use_reranker,
            )
            raise AdaptiveRetrievalError(
                f"Downstream retrieval failed for strategy '{decision.strategy.value}' "
                f"(rule: '{decision.rule_name}'): {e}"
            ) from e

        duration_s = time.perf_counter() - t0
        track_retrieval(
            route=decision.strategy.value,
            mode="adaptive",
            status="success",
            duration_seconds=duration_s,
            chunk_count=len(results),
            reranked=decision.use_reranker,
        )

        logger.info(
            "Adaptive retrieval executed",
            strategy=decision.strategy.value,
            rule=decision.rule_name,
            use_reranker=decision.use_reranker,
            chunk_count=len(results),
            duration_ms=round(duration_s * 1000.0, 2),
        )

        # 5. Enrich result metadata with routing provenance
        for result in results:
            result.metadata["routing_rule"] = decision.rule_name
            result.metadata["routing_strategy"] = decision.strategy.value
            result.metadata["routing_use_reranker"] = decision.use_reranker
            result.metadata["routing_candidate_pool_size"] = decision.candidate_pool_size
            result.metadata["routing_final_top_k"] = decision.final_top_k

        return results


    async def aretrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Asynchronous retrieval alias matching BaseRetriever async convention."""
        return await self.retrieve(query)

    def _resolve_retriever(self, decision: RoutingDecision) -> BaseRetriever:
        """Resolve the appropriate configured retriever for the RoutingDecision."""
        if decision.strategy == RetrievalStrategy.SPARSE_BM25:
            if self._bm25_retriever is None:
                raise AdaptiveRetrievalError(
                    "Missing retriever configuration: No BM25Retriever configured for SPARSE_BM25"
                )
            return self._bm25_retriever

        if decision.strategy == RetrievalStrategy.DENSE_VECTOR:
            if self._semantic_retriever is None:
                raise AdaptiveRetrievalError(
                    "Missing retriever configuration: No SemanticRetriever configured for DENSE_VECTOR"
                )
            return self._semantic_retriever

        if decision.strategy in (RetrievalStrategy.HYBRID_RRF, RetrievalStrategy.HYBRID_WEIGHTED):
            if decision.use_reranker:
                if self._reranked_retriever is None:
                    raise AdaptiveRetrievalError(
                        f"Missing retriever configuration: No RerankedRetriever configured for {decision.strategy.value} with reranking"
                    )
                return self._reranked_retriever
            else:
                if self._hybrid_retriever is None:
                    raise AdaptiveRetrievalError(
                        f"Missing retriever configuration: No HybridRetriever configured for {decision.strategy.value} without reranking"
                    )
                return self._hybrid_retriever

        raise AdaptiveRetrievalError(
            f"Unsupported retrieval strategy in RoutingDecision: {decision.strategy}"
        )
