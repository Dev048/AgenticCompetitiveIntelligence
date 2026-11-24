"""Strategic Insights API endpoints."""

from fastapi import APIRouter, Depends, Query

from competitive_intelligence.application.api.deps import (
    get_current_workspace,
    get_intelligence_service,
)
from competitive_intelligence.application.api.v1.schemas.intelligence import (
    InsightResponse,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.intelligence_service import (
    IntelligenceService,
)

router = APIRouter(prefix="/workspaces/{workspace_id}/insights", tags=["Strategic Insights"])


@router.get("", response_model=list[InsightResponse])
async def list_insights(
    workspace_id: str,
    company_id: str | None = Query(default=None, description="Filter by competitor company ID"),
    limit: int = Query(default=100, ge=1, le=500),
    _ws: Workspace = Depends(get_current_workspace),
    service: IntelligenceService = Depends(get_intelligence_service),
) -> list[InsightResponse]:
    """List strategic insights in workspace."""
    insights = await service.list_insights(workspace_id=workspace_id, company_id=company_id, limit=limit)
    return [
        InsightResponse(
            insight_id=ins.insight_id,
            workspace_id=ins.workspace_id,
            company_id=ins.company_id,
            insight_type=ins.insight_type.value,
            title=ins.title,
            summary=ins.summary,
            statement=ins.statement,
            factual_confidence=ins.factual_confidence.model_dump(mode="json"),
            interpretive_confidence=ins.interpretive_confidence.model_dump(mode="json"),
            significance=ins.significance.model_dump(mode="json"),
            temporal_scope=ins.temporal_scope.model_dump(mode="json"),
            generated_at=ins.generated_at,
            status=ins.status.value,
            supporting_signal_ids=ins.supporting_signal_ids,
            supporting_event_ids=ins.supporting_event_ids,
            created_at=ins.generated_at,
        )
        for ins in insights
    ]


@router.get("/{insight_id}", response_model=InsightResponse)
async def get_insight(
    workspace_id: str,
    insight_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: IntelligenceService = Depends(get_intelligence_service),
) -> InsightResponse:
    """Get strategic insight details with linked supporting signals and events."""
    ins = await service.get_insight(workspace_id=workspace_id, insight_id=insight_id)
    return InsightResponse(
        insight_id=ins.insight_id,
        workspace_id=ins.workspace_id,
        company_id=ins.company_id,
        insight_type=ins.insight_type.value,
        title=ins.title,
        summary=ins.summary,
        statement=ins.statement,
        factual_confidence=ins.factual_confidence.model_dump(mode="json"),
        interpretive_confidence=ins.interpretive_confidence.model_dump(mode="json"),
        significance=ins.significance.model_dump(mode="json"),
        temporal_scope=ins.temporal_scope.model_dump(mode="json"),
        generated_at=ins.generated_at,
        status=ins.status.value,
        supporting_signal_ids=ins.supporting_signal_ids,
        supporting_event_ids=ins.supporting_event_ids,
        created_at=ins.generated_at,
    )
