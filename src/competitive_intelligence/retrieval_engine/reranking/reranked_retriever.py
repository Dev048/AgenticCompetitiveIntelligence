"""Composable Reranked Retriever.

This component wraps an upstream candidate BaseRetriever and a downstream BaseReranker.
It orchestrates the pipeline, applying optional final top_k truncation after reranking,
while strictly preserving all retrieval and fusion metadata produced by upstream components.
"""

from competitive_intelligence.retrieval_engine.interfaces import BaseReranker, BaseRetriever
from competitive_intelligence.retrieval_engine.models import RetrievalQuery, RetrievalResult
from competitive_intelligence.retrieval_engine.reranking.errors import RerankedRetrievalError


class RerankedRetriever(BaseRetriever):
    """Retrieval component composing candidate generation with cross-encoder reranking.

    Responsibilities:
    - Implements the standard application-level BaseRetriever contract (RetrievalQuery -> list[RetrievalResult]).
    - Calls the upstream retriever using the typed RetrievalQuery payload.
    - If candidates are found, passes them to the downstream reranker.
    - Enforces final top_k truncation if provided in the RetrievalQuery.
    - Wraps underlying failures in RerankedRetrievalError.
    """

    def __init__(self, upstream_retriever: BaseRetriever, reranker: BaseReranker, candidate_pool_size: int = 50) -> None:
        """Initialize the composed RerankedRetriever.

        Args:
            upstream_retriever: Upstream candidate generator (e.g., Semantic, BM25, Hybrid).
            reranker: Downstream cross-encoder scorer.
            candidate_pool_size: The number of candidates to request from the upstream retriever.
        """
        self._upstream_retriever = upstream_retriever
        self._reranker = reranker
        self._candidate_pool_size = candidate_pool_size

    @property
    def upstream_retriever(self) -> BaseRetriever:
        """The upstream candidate generator."""
        return self._upstream_retriever

    @property
    def reranker(self) -> BaseReranker:
        """The downstream reranker."""
        return self._reranker

    @property
    def candidate_pool_size(self) -> int:
        """The number of candidates requested from the upstream retriever."""
        return self._candidate_pool_size

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Execute the two-stage retrieve and rerank pipeline.

        Args:
            query: The standardized query payload containing query_text, filters, and top_k parameters.

        Returns:
            A list of RetrievalResult objects scored and ordered by the reranker.

        Raises:
            RerankedRetrievalError: If candidate retrieval or reranking fails.
        """
        try:
            # 1. Candidate Generation
            # We rewrite the query to request the full candidate_pool_size from the upstream retriever
            upstream_query = query.model_copy(update={"top_k": self._candidate_pool_size})
            candidates = await self._upstream_retriever.retrieve(upstream_query)
        except Exception as e:
            raise RerankedRetrievalError(f"Upstream retrieval failed for query '{query.query_text}': {e}") from e

        if not candidates:
            return []

        try:
            # 2. Reranking
            # Pass the final top_k required by the query to the reranker.
            reranked_results = await self._reranker.rerank(
                query=query.query_text,
                candidates=candidates,
                top_k=query.top_k,
            )
            return reranked_results
        except Exception as e:
            raise RerankedRetrievalError(f"Reranking failed for query '{query.query_text}': {e}") from e
