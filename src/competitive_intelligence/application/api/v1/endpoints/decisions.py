"""Decision Contexts and Candidate Options API endpoints."""

from fastapi import APIRouter, Depends, status

from competitive_intelligence.application.api.deps import (
    get_current_workspace,
    get_decision_service,
)
from competitive_intelligence.application.api.v1.schemas.decision import (
    DecisionContextCreateRequest,
    DecisionContextResponse,
    DecisionOptionResponse,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.decision_service import DecisionService
from competitive_intelligence.decision_support.models import DecisionContext

router = APIRouter(prefix="/workspaces/{workspace_id}/decisions", tags=["Decision Contexts"])


@router.post(
    "/contexts", response_model=DecisionContextResponse, status_code=status.HTTP_201_CREATED
)
async def create_decision_context(
    workspace_id: str,
    payload: DecisionContextCreateRequest,
    _ws: Workspace = Depends(get_current_workspace),
    service: DecisionService = Depends(get_decision_service),
) -> DecisionContextResponse:
    """Create a new decision context."""
    domain_ctx = DecisionContext(
        workspace_id=workspace_id,
        title=payload.title,
        decision_domain=payload.decision_domain,
        competitor_ids=payload.competitor_ids,
        business_objectives=payload.business_objectives,
        constraints=payload.constraints,
        risk_tolerance=payload.risk_tolerance,
        time_horizon=payload.time_horizon,
        current_state_summary=payload.current_state_summary,
        deadline=payload.deadline,
        metadata=payload.metadata,
    )
    ctx = await service.create_decision_context(workspace_id=workspace_id, context=domain_ctx)
    return DecisionContextResponse(
        decision_id=ctx.decision_id,
        workspace_id=ctx.workspace_id,
        organization_id=ctx.organization_id,
        title=ctx.title,
        decision_domain=ctx.decision_domain.value,
        competitor_ids=ctx.competitor_ids,
        business_objectives=[o.model_dump(mode="json") for o in ctx.business_objectives],
        constraints=[c.model_dump(mode="json") for c in ctx.constraints],
        risk_tolerance=ctx.risk_tolerance.value,
        time_horizon=ctx.time_horizon.value,
        current_state_summary=ctx.current_state_summary,
        deadline=ctx.deadline,
        metadata=ctx.metadata,
    )


@router.get("/contexts", response_model=list[DecisionContextResponse])
async def list_decision_contexts(
    workspace_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: DecisionService = Depends(get_decision_service),
) -> list[DecisionContextResponse]:
    """List all decision contexts belonging to workspace."""
    contexts = await service.list_decision_contexts(workspace_id=workspace_id)
    return [
        DecisionContextResponse(
            decision_id=ctx.decision_id,
            workspace_id=ctx.workspace_id,
            organization_id=ctx.organization_id,
            title=ctx.title,
            decision_domain=ctx.decision_domain.value,
            competitor_ids=ctx.competitor_ids,
            business_objectives=[o.model_dump(mode="json") for o in ctx.business_objectives],
            constraints=[c.model_dump(mode="json") for c in ctx.constraints],
            risk_tolerance=ctx.risk_tolerance.value,
            time_horizon=ctx.time_horizon.value,
            current_state_summary=ctx.current_state_summary,
            deadline=ctx.deadline,
            metadata=ctx.metadata,
        )
        for ctx in contexts
    ]


@router.get("/contexts/{context_id}", response_model=DecisionContextResponse)
async def get_decision_context(
    workspace_id: str,
    context_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: DecisionService = Depends(get_decision_service),
) -> DecisionContextResponse:
    """Get decision context by ID."""
    ctx = await service.get_decision_context(workspace_id=workspace_id, context_id=context_id)
    return DecisionContextResponse(
        decision_id=ctx.decision_id,
        workspace_id=ctx.workspace_id,
        organization_id=ctx.organization_id,
        title=ctx.title,
        decision_domain=ctx.decision_domain.value,
        competitor_ids=ctx.competitor_ids,
        business_objectives=[o.model_dump(mode="json") for o in ctx.business_objectives],
        constraints=[c.model_dump(mode="json") for c in ctx.constraints],
        risk_tolerance=ctx.risk_tolerance.value,
        time_horizon=ctx.time_horizon.value,
        current_state_summary=ctx.current_state_summary,
        deadline=ctx.deadline,
        metadata=ctx.metadata,
    )


@router.get(
    "/contexts/{context_id}/options", response_model=list[DecisionOptionResponse]
)
async def list_options_for_context(
    workspace_id: str,
    context_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: DecisionService = Depends(get_decision_service),
) -> list[DecisionOptionResponse]:
    """List candidate decision options evaluated under context."""
    options = await service.list_options_for_context(
        workspace_id=workspace_id, context_id=context_id
    )
    return [
        DecisionOptionResponse(
            option_id=opt.option_id,
            decision_id=opt.decision_id,
            title=opt.title,
            description=opt.description,
            action_type=opt.action_type.value,
            expected_effects=[e.model_dump(mode="json") for e in opt.expected_effects],
            required_resources=opt.required_resources,
            estimated_cost_usd=opt.estimated_cost_usd,
            risks=[r.model_dump(mode="json") for r in opt.risks],
            tradeoffs=opt.tradeoffs,
            assumptions=opt.assumptions,
            supporting_insight_ids=opt.supporting_insight_ids,
            supporting_event_ids=opt.supporting_event_ids,
        )
        for opt in options
    ]
