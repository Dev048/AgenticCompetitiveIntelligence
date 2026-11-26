"""Evidence query use case: workspace-scoped retrieval with an auditable agentic trace."""

import time
from typing import Literal

from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.models import AgenticRAGState
from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.agentic_rag.selective.controller import (
    SelectiveAgenticRetriever,
)
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)

QueryMode = Literal["auto", "agentic"]


class EvidenceQueryTrace(BaseModel):
    """How the evidence was gathered (deterministic policy decisions + agentic steps)."""

    mode: QueryMode
    escalated: bool
    escalation_reasons: list[str] = Field(default_factory=list)
    escalation_signals: list[str] = Field(default_factory=list)
    final_source: str
    llm_calls: int = 0
    retrieval_hops: int = 1
    latency_ms: float = 0.0
    agentic_state: AgenticRAGState | None = None
    degraded: bool = False
    degraded_reason: str | None = None


class EvidenceQueryOutcome(BaseModel):
    evidence: list[RetrievalResult]
    trace: EvidenceQueryTrace


class EvidenceQueryService:
    """Runs a question against one workspace's evidence.

    The workspace filter is always applied, so a query can never read another tenant's
    chunks. Generates no free-text answer: it returns grounded evidence plus the trace.
    """

    def __init__(self, retriever: SelectiveAgenticRetriever) -> None:
        self._retriever = retriever

    async def query(
        self,
        workspace_id: str,
        query_text: str,
        top_k: int,
        mode: QueryMode,
        company_ids: list[str] | None = None,
    ) -> EvidenceQueryOutcome:
        retrieval_query = RetrievalQuery(
            query_text=query_text,
            metadata_filter=MetadataFilter(workspace_id=workspace_id, company_ids=company_ids),
            top_k=top_k,
        )

        started = time.perf_counter()
        if mode == "agentic":
            agentic = self._retriever.agentic_retriever
            results = await agentic.retrieve(retrieval_query)
            trace = self._agentic_trace(agentic, results, started)
        else:
            results = await self._retriever.retrieve(retrieval_query)
            trace = self._selective_trace(results, started)

        raw_reason = _fallback_reason(results)
        if raw_reason is not None:
            trace.degraded = True
            trace.degraded_reason = describe_degradation(raw_reason)

        # Defense in depth: never return chunks outside the requested workspace.
        scoped = [r for r in results if r.workspace_id in (None, workspace_id)]
        return EvidenceQueryOutcome(evidence=scoped[:top_k], trace=trace)

    def _selective_trace(
        self, results: list[RetrievalResult], started: float
    ) -> EvidenceQueryTrace:
        sel = self._retriever.last_trace
        agentic_state: AgenticRAGState | None = None
        if sel is not None and sel.escalated and isinstance(
            self._retriever.agentic_retriever, AgenticRetriever
        ):
            agentic_state = self._retriever.agentic_retriever.last_execution_state
        return EvidenceQueryTrace(
            mode="auto",
            escalated=bool(sel and sel.escalated),
            escalation_reasons=list(sel.assessment.reasons) if sel else [],
            escalation_signals=[s.value for s in sel.assessment.signals_detected] if sel else [],
            final_source=sel.final_source if sel else "adaptive",
            llm_calls=sel.llm_calls if sel else 0,
            retrieval_hops=sel.retrieval_hops if sel else 1,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            agentic_state=agentic_state,
        )

    @staticmethod
    def _agentic_trace(
        agentic: object, results: list[RetrievalResult], started: float
    ) -> EvidenceQueryTrace:
        state: AgenticRAGState | None = None
        llm_calls = 0
        if isinstance(agentic, AgenticRetriever):
            state = agentic.last_execution_state
            usage = agentic.last_budget_usage
            llm_calls = int(getattr(usage, "llm_calls", 0) or 0)
        fell_back = any(r.metadata.get("agentic_fallback") for r in results)
        return EvidenceQueryTrace(
            mode="agentic",
            escalated=True,
            escalation_reasons=["Agentic mode requested explicitly"],
            final_source="agentic_fallback" if fell_back else "agentic",
            llm_calls=llm_calls,
            retrieval_hops=state.hops_executed if state else 1,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            agentic_state=None if fell_back else state,
        )


def _fallback_reason(results: list[RetrievalResult]) -> str | None:
    for result in results:
        for key in ("agentic_fallback_reason", "selective_fallback_reason"):
            reason = result.metadata.get(key)
            if reason:
                return str(reason)
    return None


def describe_degradation(raw_reason: str) -> str:
    """User-facing explanation of why the agentic step was skipped."""
    lowered = raw_reason.lower()
    if "ratelimit" in lowered or "rate limit" in lowered or "cooling down" in lowered:
        return (
            "The LLM provider's free-tier rate limit was reached, so the agentic steps were "
            "skipped and deterministic hybrid retrieval answered instead."
        )
    if "busy" in lowered:
        return "The LLM is busy with other queries; deterministic retrieval answered instead."
    if "budget" in lowered:
        return "The per-query LLM budget ran out; deterministic retrieval answered instead."
    if "timeout" in lowered:
        return "The agentic steps timed out; deterministic retrieval answered instead."
    return "The agentic steps failed; deterministic retrieval answered instead."
