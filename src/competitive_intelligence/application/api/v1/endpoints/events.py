"""Intelligence Events API endpoints."""

from fastapi import APIRouter, Depends, Query

from competitive_intelligence.application.api.deps import (
    get_current_workspace,
    get_intelligence_service,
)
from competitive_intelligence.application.api.v1.schemas.intelligence import (
    EvidenceReferenceResponse,
    IntelligenceEventResponse,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.intelligence_service import (
    IntelligenceService,
)

router = APIRouter(prefix="/workspaces/{workspace_id}/events", tags=["Intelligence Events"])


@router.get("", response_model=list[IntelligenceEventResponse])
async def list_events(
    workspace_id: str,
    company_id: str | None = Query(default=None, description="Filter by competitor company ID"),
    limit: int = Query(default=100, ge=1, le=500),
    _ws: Workspace = Depends(get_current_workspace),
    service: IntelligenceService = Depends(get_intelligence_service),
) -> list[IntelligenceEventResponse]:
    """List intelligence events in workspace."""
    events = await service.list_events(workspace_id=workspace_id, company_id=company_id, limit=limit)
    return [
        IntelligenceEventResponse(
            event_id=e.event_id,
            workspace_id=e.workspace_id,
            company_id=e.company_id,
            event_type=e.event_type.value,
            title=e.title,
            summary=e.summary,
            subject=e.subject,
            previous_state=e.previous_state,
            new_state=e.new_state,
            event_time=e.event_time.model_dump(mode="json"),
            detected_at=e.detected_at,
            confidence=e.confidence.model_dump(mode="json"),
            significance=e.significance.model_dump(mode="json"),
            primary_perspective=e.primary_perspective.value,
            sequence_id=e.sequence_id,
            supporting_evidence=[
                EvidenceReferenceResponse(
                    evidence_id=ev.evidence_id,
                    source_id=ev.source_id,
                    document_id=ev.document_id,
                    version_id=ev.version_id,
                    chunk_id=ev.chunk_id,
                    source_url=ev.source_url,
                    source_category=ev.source_category,
                    source_perspective=ev.source_perspective.value,
                    evidence_role=ev.role.value,
                    extracted_text=ev.extracted_text,
                    retrieval_score=ev.retrieval_score,
                    acquired_at=ev.acquired_at,
                )
                for ev in e.evidence
            ],
            created_at=e.detected_at,
        )
        for e in events
    ]


@router.get("/{event_id}", response_model=IntelligenceEventResponse)
async def get_event(
    workspace_id: str,
    event_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: IntelligenceService = Depends(get_intelligence_service),
) -> IntelligenceEventResponse:
    """Get intelligence event details with supporting evidence."""
    e = await service.get_event(workspace_id=workspace_id, event_id=event_id)
    return IntelligenceEventResponse(
        event_id=e.event_id,
        workspace_id=e.workspace_id,
        company_id=e.company_id,
        event_type=e.event_type.value,
        title=e.title,
        summary=e.summary,
        subject=e.subject,
        previous_state=e.previous_state,
        new_state=e.new_state,
        event_time=e.event_time.model_dump(mode="json"),
        detected_at=e.detected_at,
        confidence=e.confidence.model_dump(mode="json"),
        significance=e.significance.model_dump(mode="json"),
        primary_perspective=e.primary_perspective.value,
        sequence_id=e.sequence_id,
        supporting_evidence=[
            EvidenceReferenceResponse(
                evidence_id=ev.evidence_id,
                source_id=ev.source_id,
                document_id=ev.document_id,
                version_id=ev.version_id,
                chunk_id=ev.chunk_id,
                source_url=ev.source_url,
                source_category=ev.source_category,
                source_perspective=ev.source_perspective.value,
                evidence_role=ev.role.value,
                extracted_text=ev.extracted_text,
                retrieval_score=ev.retrieval_score,
                acquired_at=ev.acquired_at,
            )
            for ev in e.evidence
        ],
        created_at=e.detected_at,
    )
