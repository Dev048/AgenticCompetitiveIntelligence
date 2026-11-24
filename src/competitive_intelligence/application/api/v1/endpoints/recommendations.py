"""Actionable Recommendations and 10-Layer Provenance API endpoints."""

from fastapi import APIRouter, Depends, Query

from competitive_intelligence.application.api.deps import (
    get_current_workspace,
    get_decision_service,
)
from competitive_intelligence.application.api.v1.schemas.decision import (
    ProvenanceTreeResponse,
    RecommendationResponse,
    RecommendationStatusUpdateRequest,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.decision_service import DecisionService
from competitive_intelligence.core.errors import ValidationError
from competitive_intelligence.decision_support.models import RecommendationStatus

router = APIRouter(prefix="/workspaces/{workspace_id}/recommendations", tags=["Recommendations"])


@router.get("", response_model=list[RecommendationResponse])
async def list_recommendations(
    workspace_id: str,
    context_id: str | None = Query(default=None, description="Optional decision context ID filter"),
    _ws: Workspace = Depends(get_current_workspace),
    service: DecisionService = Depends(get_decision_service),
) -> list[RecommendationResponse]:
    """List recommendations in workspace (optionally filtered by context)."""
    if context_id is not None:
        recommendations = await service.list_recommendations_for_context(
            workspace_id=workspace_id, context_id=context_id
        )
    else:
        recommendations = await service.list_recommendations_by_workspace(
            workspace_id=workspace_id
        )
    return [
        RecommendationResponse(
            recommendation_id=r.recommendation_id,
            decision_id=r.decision_id,
            workspace_id=r.workspace_id,
            selected_option_id=r.selected_option_id,
            urgency=r.urgency.value,
            status=r.status.value,
            title=r.title,
            summary=r.summary,
            rationale=r.rationale,
            evaluated_alternatives=[a.model_dump(mode="json") for a in r.evaluated_alternatives],
            expected_benefits=r.expected_benefits,
            tradeoffs=r.tradeoffs,
            critical_assumptions=r.critical_assumptions,
            unresolved_uncertainties=r.unresolved_uncertainties,
            confidence=r.confidence.model_dump(mode="json"),
            supporting_insight_ids=r.supporting_insight_ids,
            supporting_event_ids=r.supporting_event_ids,
            generated_at=r.generated_at,
        )
        for r in recommendations
    ]


@router.get("/{recommendation_id}", response_model=RecommendationResponse)
async def get_recommendation(
    workspace_id: str,
    recommendation_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: DecisionService = Depends(get_decision_service),
) -> RecommendationResponse:
    """Get recommendation details."""
    r = await service.get_recommendation(
        workspace_id=workspace_id, recommendation_id=recommendation_id
    )
    return RecommendationResponse(
        recommendation_id=r.recommendation_id,
        decision_id=r.decision_id,
        workspace_id=r.workspace_id,
        selected_option_id=r.selected_option_id,
        urgency=r.urgency.value,
        status=r.status.value,
        title=r.title,
        summary=r.summary,
        rationale=r.rationale,
        evaluated_alternatives=[a.model_dump(mode="json") for a in r.evaluated_alternatives],
        expected_benefits=r.expected_benefits,
        tradeoffs=r.tradeoffs,
        critical_assumptions=r.critical_assumptions,
        unresolved_uncertainties=r.unresolved_uncertainties,
        confidence=r.confidence.model_dump(mode="json"),
        supporting_insight_ids=r.supporting_insight_ids,
        supporting_event_ids=r.supporting_event_ids,
        generated_at=r.generated_at,
    )


@router.get(
    "/{recommendation_id}/provenance", response_model=ProvenanceTreeResponse
)
async def get_recommendation_provenance(
    workspace_id: str,
    recommendation_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: DecisionService = Depends(get_decision_service),
) -> ProvenanceTreeResponse:
    """Reconstruct complete 10-layer grounded provenance hierarchy for a recommendation:

    Recommendation -> Option -> Insight -> Signal -> Event -> Evidence -> Chunk -> Version -> Document -> Source.
    """
    prov = await service.get_recommendation_provenance(
        workspace_id=workspace_id, recommendation_id=recommendation_id
    )
    return ProvenanceTreeResponse(**prov)


@router.patch("/{recommendation_id}/status", response_model=RecommendationResponse)
async def update_recommendation_status(
    workspace_id: str,
    recommendation_id: str,
    payload: RecommendationStatusUpdateRequest,
    _ws: Workspace = Depends(get_current_workspace),
    service: DecisionService = Depends(get_decision_service),
) -> RecommendationResponse:
    """Record human review action for recommendation (ACCEPTED, REJECTED)."""
    try:
        target_status = RecommendationStatus(payload.status.lower())
    except ValueError as err:
        raise ValidationError(
            f"Invalid status '{payload.status}'. Allowed values: {[s.value for s in RecommendationStatus]}"
        ) from err

    updated = await service.update_recommendation_status(
        workspace_id=workspace_id,
        recommendation_id=recommendation_id,
        status=target_status,
    )
    return RecommendationResponse(
        recommendation_id=updated.recommendation_id,
        decision_id=updated.decision_id,
        workspace_id=updated.workspace_id,
        selected_option_id=updated.selected_option_id,
        urgency=updated.urgency.value,
        status=updated.status.value,
        title=updated.title,
        summary=updated.summary,
        rationale=updated.rationale,
        evaluated_alternatives=[a.model_dump(mode="json") for a in updated.evaluated_alternatives],
        expected_benefits=updated.expected_benefits,
        tradeoffs=updated.tradeoffs,
        critical_assumptions=updated.critical_assumptions,
        unresolved_uncertainties=updated.unresolved_uncertainties,
        confidence=updated.confidence.model_dump(mode="json"),
        supporting_insight_ids=updated.supporting_insight_ids,
        supporting_event_ids=updated.supporting_event_ids,
        generated_at=updated.generated_at,
    )
