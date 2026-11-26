"""Evidence query (retrieval + bounded agentic RAG) API schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvidenceQueryRequest(BaseModel):
    """Ask a question against a workspace's indexed evidence."""

    query_text: str = Field(min_length=3, max_length=500)
    company_ids: list[str] | None = Field(default=None, max_length=10)
    top_k: int = Field(default=5, ge=1, le=10)
    mode: Literal["auto", "agentic"] = Field(
        default="auto",
        description=(
            "'auto' runs cheap deterministic retrieval and escalates to the agentic graph "
            "only when the escalation policy detects an evidence gap; 'agentic' always runs "
            "the bounded agentic graph (query analysis, sub-queries, verification)."
        ),
    )


class EvidenceItemResponse(BaseModel):
    """One cited evidence chunk with full provenance pointers."""

    chunk_id: str
    score: float
    text: str
    document_id: str
    version_id: str
    source_id: str | None = None
    company_id: str | None = None
    canonical_url: str | None = None
    source_category: str | None = None


class SubQueryResponse(BaseModel):
    sub_query: str
    rationale: str = ""
    target_company_ids: list[str] = Field(default_factory=list)


class VerificationStepResponse(BaseModel):
    decision: str
    is_sufficient: bool
    reasoning: str
    missing_aspects: list[str] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)


class AgenticTraceResponse(BaseModel):
    """Audit trace of how the evidence was gathered."""

    mode: str
    escalated: bool
    escalation_reasons: list[str] = Field(default_factory=list)
    escalation_signals: list[str] = Field(default_factory=list)
    final_source: str
    llm_calls: int = 0
    retrieval_hops: int = 1
    latency_ms: float = 0.0
    query_decision: str | None = None
    query_reasoning: str | None = None
    sub_queries: list[SubQueryResponse] = Field(default_factory=list)
    verifications: list[VerificationStepResponse] = Field(default_factory=list)
    termination_reason: str | None = None
    is_fallback: bool = False
    degraded: bool = Field(
        default=False,
        description="True when the agentic steps were skipped (rate limit, budget, timeout).",
    )
    degraded_reason: str | None = None


class EvidenceQueryResponse(BaseModel):
    query_text: str
    workspace_id: str
    evidence: list[EvidenceItemResponse]
    trace: AgenticTraceResponse
