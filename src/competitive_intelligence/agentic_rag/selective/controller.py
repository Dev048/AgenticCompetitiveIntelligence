"""Selective Agentic RAG Escalation Controller.

Composes AdaptiveRetriever and AgenticRetriever with a deterministic escalation policy
to execute cheap baseline retrieval for simple queries while selectively escalating
to bounded Agentic retrieval only when evidence gaps or query complexity justify it.
"""

import time
from typing import Any

from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.agentic_rag.selective.models import (
    EscalationAssessment,
    ResultSelectionStrategy,
    SelectiveExecutionTrace,
    SelectiveRetrieverConfig,
)
from competitive_intelligence.agentic_rag.selective.policy import (
    DeterministicEscalationPolicy,
)
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
)

logger = get_logger(__name__)


class SelectiveAgenticRetriever(BaseRetriever):
    """Selective Escalation Controller for Agentic RAG.

    Invariants:
    - Implements standard BaseRetriever contract (RetrievalQuery -> list[RetrievalResult]).
    - Directly compatible with RetrievalStage and existing production pipelines.
    - Default path is cheap deterministic AdaptiveRetriever (1-2 ms, 0 LLM calls).
    - Escalates to AgenticRetriever only when DeterministicEscalationPolicy signals necessity.
    - Preserves tenant/workspace isolation and metadata filters.
    - Implements PRESERVE_BETTER result selection to avoid subquery rank-dilution.
    - Guarantees complete provenance tracking (initial strategy, escalation status, reasons, final source).
    - Atomic fallback on any agentic execution or budget failure.
    """

    def __init__(
        self,
        adaptive_retriever: BaseRetriever,
        agentic_retriever: BaseRetriever,
        policy: DeterministicEscalationPolicy | None = None,
        config: SelectiveRetrieverConfig | None = None,
    ) -> None:
        """Initialize SelectiveAgenticRetriever.

        Args:
            adaptive_retriever: Deterministic baseline retriever (e.g., AdaptiveRetriever).
            agentic_retriever: Bounded agentic retriever controller (e.g., AgenticRetriever).
            policy: Deterministic escalation policy (defaults to DeterministicEscalationPolicy()).
            config: Configuration for escalation and result selection.
        """
        if adaptive_retriever is self or agentic_retriever is self:
            raise ValueError("SelectiveAgenticRetriever cannot wrap itself (circular dependency).")

        self.adaptive_retriever = adaptive_retriever
        self.agentic_retriever = agentic_retriever
        self.config = config or SelectiveRetrieverConfig()
        self.policy = policy or DeterministicEscalationPolicy(self.config.policy_config)

        self._last_trace: SelectiveExecutionTrace | None = None
        self._last_assessment: EscalationAssessment | None = None

    @property
    def last_trace(self) -> SelectiveExecutionTrace | None:
        """Audit trace of the most recent retrieval execution."""
        return self._last_trace

    @property
    def last_assessment(self) -> EscalationAssessment | None:
        """Escalation assessment of the most recent retrieval execution."""
        return self._last_assessment

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Execute selective retrieval with deterministic escalation assessment.

        Args:
            query: Input RetrievalQuery containing text, metadata filters, and top_k.

        Returns:
            List of RetrievalResult objects enriched with selective provenance.
        """
        if not isinstance(query, RetrievalQuery):
            raise TypeError(f"query must be an instance of RetrievalQuery, got {type(query)}")

        start_time = time.perf_counter()

        # 1. Check if selective escalation is disabled via config
        if not self.config.enable_selective_escalation:
            results = await self.adaptive_retriever.retrieve(query)
            for res in results:
                res.metadata["selective_escalated"] = False
                res.metadata["selective_final_source"] = "adaptive_disabled"
            return results[:query.top_k]

        # 2. Step 1: Execute cheap baseline retrieval via AdaptiveRetriever
        initial_results = await self.adaptive_retriever.retrieve(query)

        # 3. Step 2: Deterministic Sufficiency & Complexity Assessment
        assessment = self.policy.assess(query=query, results=initial_results)
        self._last_assessment = assessment

        # 4. Step 3: Fast-Path — No Escalation Required
        if not assessment.should_escalate:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self._enrich_provenance(
                results=initial_results,
                escalated=False,
                assessment=assessment,
                final_source="adaptive",
                llm_calls=0,
                retrieval_hops=1,
            )

            self._last_trace = SelectiveExecutionTrace(
                query_text=query.query_text,
                escalated=False,
                assessment=assessment,
                initial_retrieval_count=len(initial_results),
                agentic_retrieval_count=0,
                final_result_count=len(initial_results[:query.top_k]),
                final_source="adaptive",
                latency_ms=latency_ms,
                llm_calls=0,
                retrieval_hops=1,
            )

            return initial_results[:query.top_k]

        # 5. Step 4: Escalation Path — Invoke AgenticRetriever
        logger.info(
            "Selective escalation triggered for query",
            query=query.query_text,
            reasons=assessment.reasons,
            signals=[s.value for s in assessment.signals_detected],
        )

        agentic_results: list[RetrievalResult] = []
        llm_calls = 0
        retrieval_hops = 1

        try:
            agentic_results = await self.agentic_retriever.retrieve(query)

            # Extract usage metrics if available from AgenticRetriever
            if isinstance(self.agentic_retriever, AgenticRetriever):
                if self.agentic_retriever.last_budget_usage:
                    llm_calls = getattr(self.agentic_retriever.last_budget_usage, "llm_calls", 0)
                if self.agentic_retriever.last_execution_state:
                    retrieval_hops = self.agentic_retriever.last_execution_state.hops_executed

        except Exception as exc:
            logger.warning(
                "Escalated agentic retrieval failed; falling back to initial adaptive evidence",
                query=query.query_text,
                error=str(exc),
            )
            # Safe atomic fallback to initial results
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self._enrich_provenance(
                results=initial_results,
                escalated=True,
                assessment=assessment,
                final_source="adaptive_fallback",
                llm_calls=0,
                retrieval_hops=1,
            )
            for res in initial_results:
                res.metadata["selective_fallback"] = True
                res.metadata["selective_fallback_reason"] = f"{type(exc).__name__}: {exc}"

            self._last_trace = SelectiveExecutionTrace(
                query_text=query.query_text,
                escalated=True,
                assessment=assessment,
                initial_retrieval_count=len(initial_results),
                agentic_retrieval_count=0,
                final_result_count=len(initial_results[:query.top_k]),
                final_source="adaptive_fallback",
                latency_ms=latency_ms,
                llm_calls=0,
                retrieval_hops=1,
            )
            return initial_results[:query.top_k]

        # 6. Step 5: Result Selection Policy
        final_results, final_source = self._select_results(
            query=query,
            initial_results=initial_results,
            agentic_results=agentic_results,
            strategy=self.config.selection_strategy,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self._enrich_provenance(
            results=final_results,
            escalated=True,
            assessment=assessment,
            final_source=final_source,
            llm_calls=llm_calls,
            retrieval_hops=retrieval_hops,
        )

        self._last_trace = SelectiveExecutionTrace(
            query_text=query.query_text,
            escalated=True,
            assessment=assessment,
            initial_retrieval_count=len(initial_results),
            agentic_retrieval_count=len(agentic_results),
            final_result_count=len(final_results[:query.top_k]),
            final_source=final_source,
            latency_ms=latency_ms,
            llm_calls=llm_calls,
            retrieval_hops=retrieval_hops,
        )

        return final_results[:query.top_k]

    async def aretrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Asynchronous retrieval alias conforming to BaseRetriever convention."""
        return await self.retrieve(query)

    def _select_results(
        self,
        query: RetrievalQuery,
        initial_results: list[RetrievalResult],
        agentic_results: list[RetrievalResult],
        strategy: ResultSelectionStrategy,
    ) -> tuple[list[RetrievalResult], str]:
        """Apply deterministic result selection strategy to merge/preserve evidence sets.

        Args:
            query: Original RetrievalQuery.
            initial_results: Baseline results from AdaptiveRetriever.
            agentic_results: Results from escalated AgenticRetriever.
            strategy: ResultSelectionStrategy to apply.

        Returns:
            Tuple of (final_results, final_source_label).
        """
        # If agentic produced 0 results, always preserve Adaptive
        if not agentic_results:
            return initial_results, "adaptive"

        if strategy == ResultSelectionStrategy.ADAPTIVE_ONLY:
            return initial_results, "adaptive"

        if strategy == ResultSelectionStrategy.AGENTIC_PREFERRED:
            return agentic_results, "agentic"

        if strategy == ResultSelectionStrategy.HYBRID_MERGE:
            merged = self._merge_candidate_sets(initial_results, agentic_results)
            return merged, "merged"

        # Default & Recommended: PRESERVE_BETTER
        # 1. Identify newly recovered chunks from Agentic that were missing in Adaptive
        initial_ids = {r.chunk_id for r in initial_results}
        agentic_ids = {r.chunk_id for r in agentic_results}
        novel_agentic_ids = agentic_ids - initial_ids

        # If Agentic did not discover any novel chunks and its top score is worse, retain Adaptive
        initial_top_score = float(initial_results[0].score) if initial_results else 0.0
        agentic_top_score = float(agentic_results[0].score) if agentic_results else 0.0

        if not novel_agentic_ids and agentic_top_score <= initial_top_score:
            return initial_results, "adaptive"

        # If Agentic discovered novel chunks or improved score, perform quality-preserving merge:
        # Retain high-confidence Adaptive chunks and incorporate novel Agentic chunks
        merged = self._merge_candidate_sets(initial_results, agentic_results)
        return merged, "merged" if novel_agentic_ids and initial_ids else "agentic"

    def _merge_candidate_sets(
        self,
        set_a: list[RetrievalResult],
        set_b: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Deterministically merge two result sets by chunk_id, taking max score and sorting descending."""
        merged_map: dict[str, RetrievalResult] = {}

        # Process set_a (Adaptive)
        for res in set_a:
            merged_map[res.chunk_id] = res

        # Process set_b (Agentic) - update if higher score or novel
        for res in set_b:
            if res.chunk_id not in merged_map:
                merged_map[res.chunk_id] = res
            else:
                existing = merged_map[res.chunk_id]
                if res.score > existing.score:
                    merged_map[res.chunk_id] = res

        sorted_results = sorted(merged_map.values(), key=lambda r: r.score, reverse=True)
        return sorted_results

    def _enrich_provenance(
        self,
        results: list[RetrievalResult],
        escalated: bool,
        assessment: EscalationAssessment,
        final_source: str,
        llm_calls: int,
        retrieval_hops: int,
    ) -> None:
        """Enrich every result's metadata with complete controller provenance."""
        for res in results:
            res.metadata["selective_initial_strategy"] = "adaptive"
            res.metadata["selective_escalated"] = escalated
            res.metadata["selective_reasons"] = assessment.reasons
            res.metadata["selective_signals"] = [s.value for s in assessment.signals_detected]
            res.metadata["selective_final_source"] = final_source
            res.metadata["selective_llm_calls"] = llm_calls
            res.metadata["selective_retrieval_hops"] = retrieval_hops
