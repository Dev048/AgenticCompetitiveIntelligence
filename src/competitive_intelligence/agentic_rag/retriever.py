"""AgenticRetriever adapter implementing BaseRetriever interface for LangGraph orchestration."""

import asyncio
from typing import Any
from uuid import uuid4

import time

from competitive_intelligence.agentic_rag.filters import BaseCompanyResolver
from competitive_intelligence.agentic_rag.graph.workflow import AgenticRAGWorkflow
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.models import AgenticRAGState
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.config import get_settings
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.core.metrics import track_retrieval
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
)

logger = get_logger(__name__)


class AgenticRetriever(BaseRetriever):
    """Retrieval controller orchestrating multi-hop retrieval and evidence verification via LangGraph.

    Invariants:
    - Implements the standard BaseRetriever contract (RetrievalQuery -> list[RetrievalResult]).
    - Directly compatible with RetrievalStage and downstream pipelines without modification.
    - Operates purely as a RETRIEVAL controller; does NOT generate final answers.
    - Delegates underlying search execution to injected executor_retriever (e.g., AdaptiveRetriever).
    - Strictly preserves workspace_id and metadata filter tenant isolation.
    - Enforces deterministic budget guardrails (max_hops, max_llm_calls, token budget, timeout).
    - Degrades gracefully to atomic deterministic fallback if any agentic step or budget limit fails.
    """

    def __init__(
        self,
        executor_retriever: BaseRetriever,
        llm_client: BaseLLMClient | None = None,
        max_hops: int | None = None,
        max_llm_calls: int | None = None,
        task_token_budget: int | None = None,
        timeout_seconds: float = 30.0,
        company_resolver: BaseCompanyResolver | None = None,
    ) -> None:
        """Initialize AgenticRetriever.

        Args:
            executor_retriever: Downstream retriever (e.g. AdaptiveRetriever) executing concrete search.
            llm_client: LLM client abstraction for query analysis and evidence verification.
            max_hops: Maximum retrieval hops allowed (defaults to settings.max_retrieval_hops).
            max_llm_calls: Maximum LLM calls allowed per query (defaults to settings.max_llm_calls_per_task).
            task_token_budget: Maximum tokens allowed per query (defaults to settings.task_token_budget).
            timeout_seconds: Wall-clock timeout for the entire agentic retrieval execution.
        """
        settings = get_settings()
        if executor_retriever is self or isinstance(executor_retriever, AgenticRetriever):
            raise ValueError(
                "AgenticRetriever cannot use itself or another AgenticRetriever as executor_retriever (circular dependency)."
            )
        self.executor_retriever = executor_retriever
        self.llm_client = llm_client or MockLLMClient()
        self.max_hops = max_hops if max_hops is not None else settings.max_retrieval_hops
        self.max_llm_calls = max_llm_calls if max_llm_calls is not None else settings.max_llm_calls_per_task
        self.task_token_budget = task_token_budget if task_token_budget is not None else settings.task_token_budget
        self.timeout_seconds = timeout_seconds
        self.company_resolver = company_resolver

        self._last_execution_state: AgenticRAGState | None = None
        self._last_budget_usage: Any = None

    @property
    def last_execution_state(self) -> AgenticRAGState | None:
        """State of the most recent agentic workflow execution."""
        return self._last_execution_state

    @property
    def last_budget_usage(self) -> Any:
        """Budget usage statistics of the most recent execution."""
        return self._last_budget_usage

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Execute bounded agentic retrieval with automatic atomic fallback.

        Args:
            query: Input RetrievalQuery containing query text, metadata filters, and top_k.

        Returns:
            List of RetrievalResult objects enriched with agentic provenance.
        """
        if not isinstance(query, RetrievalQuery):
            raise TypeError(f"query must be an instance of RetrievalQuery, got {type(query)}")

        t0 = time.perf_counter()
        task_id = str(uuid4())
        budget_tracker = LLMBudgetTracker(
            task_id=task_id,
            max_llm_calls=self.max_llm_calls,
            max_retrieval_hops=self.max_hops,
            max_tokens=self.task_token_budget,
        )

        try:
            workflow = AgenticRAGWorkflow(
                retriever=self.executor_retriever,
                llm_client=self.llm_client,
                budget_tracker=budget_tracker,
                company_resolver=self.company_resolver,
            )

            state = await asyncio.wait_for(
                workflow.run(query=query, max_hops=self.max_hops),
                timeout=self.timeout_seconds,
            )

            self._last_execution_state = state
            self._last_budget_usage = budget_tracker.usage

            results = state.accumulated_evidence

            # If agentic execution succeeded and produced evidence
            if results:
                # Attach latest verification provenance
                status_str = "unverified"
                if state.verifications:
                    status_str = state.verifications[-1].decision.value

                for res in results:
                    res.metadata.setdefault("agentic_verification_status", status_str)
                    res.metadata.setdefault("agentic_hops_total", state.hops_executed)

                duration_s = time.perf_counter() - t0
                track_retrieval(
                    route="agentic_multihop",
                    mode="agentic",
                    status="success",
                    duration_seconds=duration_s,
                    chunk_count=len(results),
                    escalated=True,
                )

                # Return top_k highest scoring chunks
                return results[:query.top_k]

            # If agentic workflow returned 0 results cleanly, attempt atomic direct fallback
            logger.info("Agentic workflow produced 0 results; executing direct retrieval fallback", query=query.query_text)
            track_retrieval(
                route="agentic_fallback",
                mode="agentic",
                status="success",
                duration_seconds=time.perf_counter() - t0,
                escalated=True,
            )
            return await self._execute_atomic_fallback(
                query=query,
                reason="Zero evidence accumulated in agentic workflow",
            )

        except Exception as exc:
            logger.warning(
                "Agentic retrieval failed; triggering atomic fallback to executor retriever",
                query=query.query_text,
                error=str(exc),
            )
            track_retrieval(
                route="agentic_error_fallback",
                mode="agentic",
                status="failure",
                duration_seconds=time.perf_counter() - t0,
                escalated=True,
            )
            self._last_budget_usage = budget_tracker.usage
            return await self._execute_atomic_fallback(
                query=query,
                reason=f"{type(exc).__name__}: {exc}",
            )


    async def aretrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        """Asynchronous retrieval alias conforming to BaseRetriever convention."""
        return await self.retrieve(query)

    async def _execute_atomic_fallback(
        self,
        query: RetrievalQuery,
        reason: str,
    ) -> list[RetrievalResult]:
        """Execute clean deterministic fallback via executor_retriever with explicit fallback provenance."""
        fallback_results = await self.executor_retriever.retrieve(query)

        for res in fallback_results:
            res.metadata["agentic_fallback"] = True
            res.metadata["agentic_fallback_reason"] = reason

        return fallback_results
