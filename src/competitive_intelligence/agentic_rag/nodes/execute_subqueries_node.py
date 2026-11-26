"""Retrieval execution node for running sub-queries concurrently via AdaptiveRetriever."""

import asyncio
from typing import Any

from competitive_intelligence.agentic_rag.filters import (
    BaseCompanyResolver,
    CompanyRef,
    resolve_company_refs,
    sanitize_source_categories,
)
from competitive_intelligence.agentic_rag.graph.state import AgenticGraphState
from competitive_intelligence.agentic_rag.models import SubQueryPlan
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
)

logger = get_logger(__name__)


class ExecuteSubqueriesNode:
    """Node that executes sub-queries concurrently through the injected BaseRetriever."""

    def __init__(
        self,
        retriever: BaseRetriever,
        budget_tracker: LLMBudgetTracker | None = None,
        company_resolver: BaseCompanyResolver | None = None,
    ) -> None:
        self.retriever = retriever
        self.budget_tracker = budget_tracker
        self.company_resolver = company_resolver

    async def __call__(self, state: AgenticGraphState) -> dict[str, Any]:
        """Execute concurrent retrieval for current sub-queries."""
        current_sub_queries: list[SubQueryPlan] = state.get("current_sub_queries", [])
        if not current_sub_queries:
            # Fallback to original query if no subqueries provided
            current_sub_queries = [
                SubQueryPlan(
                    sub_query=state.get("original_query", ""),
                    rationale="Direct query fallback",
                )
            ]

        workspace_id = state.get("workspace_id", "")
        base_filter = state.get("metadata_filter")
        if base_filter is None:
            comp = state.get("company_id")
            base_filter = MetadataFilter(
                workspace_id=workspace_id,
                company_ids=[comp] if comp is not None else None,
            )
        else:
            # Ensure workspace_id is strictly preserved
            base_filter = base_filter.model_copy(update={"workspace_id": workspace_id})

        top_k = state.get("top_k", 10)
        hop_count = state.get("hop_count", 1)

        # Track retrieval hop in budget tracker
        if self.budget_tracker:
            try:
                self.budget_tracker.record_retrieval_hop()
            except Exception as exc:
                logger.warning("Budget limit reached during retrieval hop", error=str(exc))
                return {
                    "termination_reason": f"Budget limit reached on hop: {exc}",
                }

        known_companies: list[CompanyRef] | None = None
        if self.company_resolver is not None and any(
            sq.target_company_ids for sq in current_sub_queries
        ):
            try:
                known_companies = await self.company_resolver.known_companies(workspace_id)
            except Exception as err:
                logger.warning("Company resolution unavailable", error=str(err))
                known_companies = []

        async def _retrieve(sub_q: SubQueryPlan, flt: MetadataFilter) -> list[RetrievalResult]:
            query_payload = RetrievalQuery(
                query_text=sub_q.sub_query,
                metadata_filter=flt,
                top_k=top_k,
                strategy=sub_q.strategy_hint or RetrievalStrategy.HYBRID_RRF,
            )
            try:
                return await self.retriever.retrieve(query_payload)
            except Exception as err:
                logger.warning(
                    "Subquery retrieval failed; continuing with remaining subqueries",
                    sub_query=sub_q.sub_query,
                    error=str(err),
                )
                return []

        async def _run_single_subquery(sub_q: SubQueryPlan) -> tuple[SubQueryPlan, list[RetrievalResult]]:
            # The workspace filter (and any caller-supplied filter) is always kept. Filters
            # proposed by the LLM are untrusted: company references are resolved to stored
            # IDs and unknown categories are dropped, deterministically.
            sub_filter = base_filter.model_copy(deep=True)
            used_llm_filters = False
            if sub_q.target_company_ids and not base_filter.company_ids:
                company_ids = (
                    resolve_company_refs(sub_q.target_company_ids, known_companies)
                    if known_companies is not None
                    else list(sub_q.target_company_ids)
                )
                if company_ids:
                    sub_filter.company_ids = company_ids
                    used_llm_filters = True
            if sub_q.target_source_categories and not base_filter.source_categories:
                categories = sanitize_source_categories(sub_q.target_source_categories)
                if categories:
                    sub_filter.source_categories = categories
                    used_llm_filters = True

            results = await _retrieve(sub_q, sub_filter)
            filter_relaxed = False
            if not results and used_llm_filters:
                # A proposed filter can still be too narrow (for example a category the
                # documents were not indexed under). Retry once with only the trusted base
                # filter instead of spending a corrective hop on an empty result.
                logger.info(
                    "Sub-query filter matched nothing; retrying with base filter",
                    sub_query=sub_q.sub_query,
                    company_ids=sub_filter.company_ids,
                    source_categories=sub_filter.source_categories,
                )
                results = await _retrieve(sub_q, base_filter.model_copy(deep=True))
                filter_relaxed = True

            # Attach provenance to retrieved results
            for res in results:
                res.metadata["agentic_hop"] = hop_count
                res.metadata["agentic_sub_query"] = sub_q.sub_query
                res.metadata["agentic_retrieval_reason"] = sub_q.rationale
                if filter_relaxed:
                    res.metadata["agentic_filter_relaxed"] = True

            return sub_q, results

        # Run all approved subqueries concurrently
        tasks = [_run_single_subquery(sq) for sq in current_sub_queries]
        subquery_outputs = await asyncio.gather(*tasks)

        # Merge and deduplicate with previously accumulated results by chunk_id
        existing_results: list[RetrievalResult] = state.get("accumulated_results", [])
        chunk_map: dict[str, RetrievalResult] = {r.chunk_id: r for r in existing_results}

        new_chunks_count = 0
        for sub_q, results in subquery_outputs:
            for r in results:
                if r.chunk_id in chunk_map:
                    # If chunk exists, update score if higher and preserve multi-query metadata
                    existing = chunk_map[r.chunk_id]
                    if r.score > existing.score:
                        chunk_map[r.chunk_id] = r
                else:
                    chunk_map[r.chunk_id] = r
                    new_chunks_count += 1

        # Sort combined results descending by primary score
        accumulated = sorted(
            chunk_map.values(),
            key=lambda x: (x.reranker_score if x.reranker_score is not None else x.score),
            reverse=True,
        )

        logger.info(
            "Executed sub-queries",
            sub_queries_count=len(current_sub_queries),
            hop=hop_count,
            new_chunks_retrieved=new_chunks_count,
            total_accumulated=len(accumulated),
        )

        return {
            "accumulated_results": accumulated,
        }
