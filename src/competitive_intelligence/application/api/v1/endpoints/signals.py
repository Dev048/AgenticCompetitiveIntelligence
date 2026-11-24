"""Inferred Signals API endpoints."""

from fastapi import APIRouter, Depends, Query

from competitive_intelligence.application.api.deps import (
    get_current_workspace,
    get_intelligence_service,
)
from competitive_intelligence.application.api.v1.schemas.intelligence import (
    InferredSignalResponse,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.intelligence_service import (
    IntelligenceService,
)

router = APIRouter(prefix="/workspaces/{workspace_id}/signals", tags=["Inferred Signals"])


@router.get("", response_model=list[InferredSignalResponse])
async def list_signals(
    workspace_id: str,
    company_id: str | None = Query(default=None, description="Filter by competitor company ID"),
    limit: int = Query(default=100, ge=1, le=500),
    _ws: Workspace = Depends(get_current_workspace),
    service: IntelligenceService = Depends(get_intelligence_service),
) -> list[InferredSignalResponse]:
    """List inferred signals in workspace."""
    signals = await service.list_signals(workspace_id=workspace_id, company_id=company_id, limit=limit)
    return [
        InferredSignalResponse(
            signal_id=s.signal_id,
            workspace_id=s.workspace_id,
            company_id=s.company_id,
            signal_type=s.signal_type.value,
            subject=s.subject,
            title=s.title,
            summary=s.summary,
            first_observed_at=s.first_observed_at,
            last_observed_at=s.last_observed_at,
            inferred_at=s.inferred_at,
            confidence=s.confidence.model_dump(mode="json"),
            status=s.status.value,
            supporting_event_ids=s.supporting_event_ids,
            created_at=s.inferred_at,
        )
        for s in signals
    ]


@router.get("/{signal_id}", response_model=InferredSignalResponse)
async def get_signal(
    workspace_id: str,
    signal_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: IntelligenceService = Depends(get_intelligence_service),
) -> InferredSignalResponse:
    """Get inferred signal details with supporting event associations."""
    s = await service.get_signal(workspace_id=workspace_id, signal_id=signal_id)
    return InferredSignalResponse(
        signal_id=s.signal_id,
        workspace_id=s.workspace_id,
        company_id=s.company_id,
        signal_type=s.signal_type.value,
        subject=s.subject,
        title=s.title,
        summary=s.summary,
        first_observed_at=s.first_observed_at,
        last_observed_at=s.last_observed_at,
        inferred_at=s.inferred_at,
        confidence=s.confidence.model_dump(mode="json"),
        status=s.status.value,
        supporting_event_ids=s.supporting_event_ids,
        created_at=s.inferred_at,
    )
