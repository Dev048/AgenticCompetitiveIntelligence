"""Hybrid retriever combining dense semantic and BM25 lexical search via Reciprocal Rank Fusion (RRF).

Implements candidate pooling, 1-indexed RRF ranking, deterministic tie-breaking,
and rich fusion metadata provenance tracking without raw score comparison or normalization.
"""

import asyncio
from collections.abc import Sequence
from typing import Any

from competitive_intelligence.knowledge_pipeline.models import (
    SearchConfiguration,
    VectorSearchFilter,
    VectorSearchResult,
)
from competitive_intelligence.retrieval_engine.dense.semantic_retriever import (
    RetrievalError,
    SemanticRetriever,
)
from competitive_intelligence.retrieval_engine.lexical.bm25_retriever import (
    BM25Retriever,
)
from competitive_intelligence.retrieval_engine.models import (
    HybridSearchConfiguration,
    RetrievalResult,
)


class HybridRetrievalError(RetrievalError):
    """Exception raised when hybrid retrieval or rank fusion fails."""


class HybridRetriever:
    """Hybrid retrieval component orchestrating SemanticRetriever and BM25Retriever with RRF.

    Responsibilities:
    - Queries SemanticRetriever and BM25Retriever using the same query string and filters.
    - Manages separate candidate pool sizes (e.g. 5x final_top_k) for both retrievers.
    - Computes Reciprocal Rank Fusion (RRF) scores using 1-indexed ranks:
        RRF(d) = sum(1 / (rrf_k + rank(d)))
    - Consolidates duplicates appearing in both candidate sets into a single result with combined score.
    - Embeds detailed fusion provenance ('semantic_rank', 'bm25_rank', 'rrf_score') into metadata.
    - Sorts descending by RRF score with deterministic tie-breaking on chunk_id.
    - Truncates output to final_top_k.
    - Fails fast on retriever errors without silent degradation.
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        bm25_retriever: BM25Retriever,
        default_config: HybridSearchConfiguration | None = None,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            semantic_retriever: Dense semantic retriever instance.
            bm25_retriever: BM25 lexical retriever instance.
            default_config: Optional default HybridSearchConfiguration.
        """
        self._semantic_retriever = semantic_retriever
        self._bm25_retriever = bm25_retriever
        self._default_config = default_config or HybridSearchConfiguration()

    @property
    def semantic_retriever(self) -> SemanticRetriever:
        """Underlying dense semantic retriever."""
        return self._semantic_retriever

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """Underlying BM25 lexical retriever."""
        return self._bm25_retriever

    @property
    def default_config(self) -> HybridSearchConfiguration:
        """Configured default hybrid parameters."""
        return self._default_config

    def retrieve(
        self,
        query: str,
        config: HybridSearchConfiguration | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[RetrievalResult]:
        """Execute synchronous hybrid retrieval with Reciprocal Rank Fusion.

        Args:
            query: Non-empty search query string.
            config: Optional configuration overriding default_config.
            filters: Optional application-level filters passed to both retrievers.

        Returns:
            A list of RetrievalResult items ranked in descending RRF score order.

        Raises:
            ValueError: If query is empty or whitespace-only.
            HybridRetrievalError: If either retriever fails.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only")

        clean_query = query.strip()
        effective_cfg = config or self._default_config

        semantic_search_cfg = SearchConfiguration(top_k=effective_cfg.effective_semantic_k)

        try:
            semantic_candidates = self._semantic_retriever.retrieve(
                query=clean_query,
                search_config=semantic_search_cfg,
                filters=filters,
            )
        except Exception as exc:
            raise HybridRetrievalError(
                f"Semantic retrieval failed during hybrid search for query '{clean_query}': {exc}"
            ) from exc

        try:
            lexical_candidates = self._bm25_retriever.retrieve(
                query=clean_query,
                top_k=effective_cfg.effective_lexical_k,
                filters=filters,
            )
        except Exception as exc:
            raise HybridRetrievalError(
                f"BM25 retrieval failed during hybrid search for query '{clean_query}': {exc}"
            ) from exc

        return self._fuse_results(
            semantic_candidates=semantic_candidates,
            lexical_candidates=lexical_candidates,
            config=effective_cfg,
        )

    async def aretrieve(
        self,
        query: str,
        config: HybridSearchConfiguration | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[RetrievalResult]:
        """Execute concurrent asynchronous hybrid retrieval with Reciprocal Rank Fusion.

        Args:
            query: Non-empty search query string.
            config: Optional configuration overriding default_config.
            filters: Optional application-level filters passed to both retrievers.

        Returns:
            A list of RetrievalResult items ranked in descending RRF score order.

        Raises:
            ValueError: If query is empty or whitespace-only.
            HybridRetrievalError: If either retriever fails.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only")

        clean_query = query.strip()
        effective_cfg = config or self._default_config

        semantic_search_cfg = SearchConfiguration(top_k=effective_cfg.effective_semantic_k)

        sem_task = self._semantic_retriever.aretrieve(
            query=clean_query,
            search_config=semantic_search_cfg,
            filters=filters,
        )
        lex_task = self._bm25_retriever.aretrieve(
            query=clean_query,
            top_k=effective_cfg.effective_lexical_k,
            filters=filters,
        )

        try:
            semantic_candidates, lexical_candidates = await asyncio.gather(
                sem_task, lex_task
            )
        except Exception as exc:
            raise HybridRetrievalError(
                f"Async hybrid retrieval failed for query '{clean_query}': {exc}"
            ) from exc

        return self._fuse_results(
            semantic_candidates=semantic_candidates,
            lexical_candidates=lexical_candidates,
            config=effective_cfg,
        )

    def _fuse_results(
        self,
        semantic_candidates: Sequence[VectorSearchResult | RetrievalResult],
        lexical_candidates: Sequence[RetrievalResult],
        config: HybridSearchConfiguration,
    ) -> list[RetrievalResult]:
        """Perform Reciprocal Rank Fusion (RRF) over semantic and lexical candidate lists.

        Formula:
            RRF(d) = sum(1 / (rrf_k + rank(d))) for 1-indexed ranks.
        """
        rrf_k = config.rrf_k
        rrf_scores: dict[str, float] = {}
        semantic_provenance: dict[str, tuple[int, float]] = {}
        bm25_provenance: dict[str, tuple[int, float]] = {}
        chunk_templates: dict[str, Any] = {}

        # 1. Process semantic candidates (1-indexed ranks)
        for rank, item in enumerate(semantic_candidates, start=1):
            cid = item.chunk_id
            score_contribution = 1.0 / (rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contribution
            semantic_provenance[cid] = (rank, float(item.score))
            chunk_templates[cid] = item

        # 2. Process lexical candidates (1-indexed ranks)
        for rank, item in enumerate(lexical_candidates, start=1):
            cid = item.chunk_id
            score_contribution = 1.0 / (rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contribution
            bm25_provenance[cid] = (rank, float(item.score))
            if cid not in chunk_templates:
                chunk_templates[cid] = item

        if not rrf_scores:
            return []

        # 3. Construct unified RetrievalResult models with fusion metadata
        fused_items: list[RetrievalResult] = []
        for cid, rrf_score in rrf_scores.items():
            base = chunk_templates[cid]
            meta = dict(base.metadata)

            sem_info = semantic_provenance.get(cid)
            bm25_info = bm25_provenance.get(cid)

            meta["semantic_rank"] = sem_info[0] if sem_info else None
            meta["semantic_score"] = sem_info[1] if sem_info else None
            meta["bm25_rank"] = bm25_info[0] if bm25_info else None
            meta["bm25_score"] = bm25_info[1] if bm25_info else None
            meta["rrf_score"] = rrf_score
            meta["fusion_method"] = "rrf"

            fused_items.append(
                RetrievalResult(
                    chunk_id=cid,
                    score=rrf_score,
                    text=base.text,
                    document_id=base.document_id,
                    version_id=base.version_id,
                    source_id=base.source_id,
                    company_id=base.company_id,
                    workspace_id=base.workspace_id,
                    source_category=base.source_category,
                    content_type=base.content_type,
                    canonical_url=base.canonical_url,
                    chunk_index=base.chunk_index,
                    parent_chunk_id=base.parent_chunk_id,
                    chunk_level=base.chunk_level,
                    metadata=meta,
                )
            )

        # 4. Deterministic sorting: descending by RRF score, secondary ascending by chunk_id
        fused_items.sort(key=lambda r: (-r.score, r.chunk_id))

        # 5. Truncate to the requested pool size
        if config.final_top_k is not None:
            return fused_items[: config.final_top_k]
        return fused_items[: config.fusion_candidate_k]
