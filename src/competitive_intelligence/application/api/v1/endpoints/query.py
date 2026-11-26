"""Evidence query endpoint: workspace-scoped retrieval with an agentic RAG trace."""

from typing import Annotated

from fastapi import APIRouter, Depends

from competitive_intelligence.application.api.demo_guard import enforce_query_rate_limit
from competitive_intelligence.application.api.deps import (
    get_current_workspace,
    get_evidence_query_service,
)
from competitive_intelligence.application.api.v1.schemas.query import (
    AgenticTraceResponse,
    EvidenceItemResponse,
    EvidenceQueryRequest,
    EvidenceQueryResponse,
    SubQueryResponse,
    VerificationStepResponse,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.evidence_query_service import (
    EvidenceQueryService,
)

router = APIRouter(prefix="/workspaces/{workspace_id}/query", tags=["Evidence Query"])

_MAX_EVIDENCE_TEXT_CHARS = 1200


@router.post("", response_model=EvidenceQueryResponse)
async def query_evidence(
    workspace_id: str,
    payload: EvidenceQueryRequest,
    _rate_limit: Annotated[None, Depends(enforce_query_rate_limit)],
    _ws: Annotated[Workspace, Depends(get_current_workspace)],
    service: Annotated[EvidenceQueryService, Depends(get_evidence_query_service)],
) -> EvidenceQueryResponse:
    """Retrieve grounded evidence for a question and return how it was gathered.

    Returns cited chunks (with source/document/version/chunk provenance) plus the
    escalation decision and, when the agentic graph ran, its query analysis,
    sub-queries, verification verdicts and detected contradictions.
    """
    outcome = await service.query(
        workspace_id=workspace_id,
        query_text=payload.query_text,
        top_k=payload.top_k,
        mode=payload.mode,
        company_ids=payload.company_ids,
    )
    t = outcome.trace
    state = t.agentic_state
    analysis = state.query_analysis if state else None
    return EvidenceQueryResponse(
        query_text=payload.query_text,
        workspace_id=workspace_id,
        evidence=[
            EvidenceItemResponse(
                chunk_id=r.chunk_id,
                score=r.score,
                text=r.text[:_MAX_EVIDENCE_TEXT_CHARS],
                document_id=r.document_id,
                version_id=r.version_id,
                source_id=r.source_id,
                company_id=r.company_id,
                canonical_url=r.canonical_url,
                source_category=r.source_category,
            )
            for r in outcome.evidence
        ],
        trace=AgenticTraceResponse(
            mode=t.mode,
            escalated=t.escalated,
            escalation_reasons=t.escalation_reasons,
            escalation_signals=t.escalation_signals,
            final_source=t.final_source,
            llm_calls=t.llm_calls,
            retrieval_hops=t.retrieval_hops,
            latency_ms=round(t.latency_ms, 1),
            query_decision=analysis.decision.value if analysis else None,
            query_reasoning=analysis.reasoning if analysis else None,
            sub_queries=[
                SubQueryResponse(
                    sub_query=sq.sub_query,
                    rationale=sq.rationale,
                    target_company_ids=sq.target_company_ids,
                )
                for sq in (analysis.sub_queries if analysis else [])
            ],
            verifications=[
                VerificationStepResponse(
                    decision=v.decision.value,
                    is_sufficient=v.is_sufficient,
                    reasoning=v.reasoning,
                    missing_aspects=v.missing_aspects,
                    contradictions=[c.model_dump() for c in v.detected_contradictions],
                    cited_chunk_ids=v.cited_chunk_ids,
                )
                for v in (state.verifications if state else [])
            ],
            termination_reason=state.termination_reason if state else None,
            is_fallback=state.is_fallback if state else t.final_source.endswith("fallback"),
            degraded=t.degraded,
            degraded_reason=t.degraded_reason,
        ),
    )
