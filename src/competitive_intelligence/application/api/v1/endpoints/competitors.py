"""Tracked Competitors API endpoints."""

from fastapi import APIRouter, Depends, Query, status

from competitive_intelligence.application.api.deps import (
    get_competitor_service,
    get_current_workspace,
)
from competitive_intelligence.application.api.v1.schemas.competitor import (
    CompetitorCreateRequest,
    CompetitorResponse,
    CompetitorUpdateRequest,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.competitor_service import CompetitorService

router = APIRouter(prefix="/workspaces/{workspace_id}/competitors", tags=["Competitors"])


@router.post("", response_model=CompetitorResponse, status_code=status.HTTP_201_CREATED)
async def register_competitor(
    workspace_id: str,
    payload: CompetitorCreateRequest,
    _ws: Workspace = Depends(get_current_workspace),
    service: CompetitorService = Depends(get_competitor_service),
) -> CompetitorResponse:
    """Register a new competitor for active monitoring within a workspace."""
    comp = await service.register_competitor(
        workspace_id=workspace_id,
        name=payload.name,
        domain=payload.domain,
        description=payload.description,
    )
    return CompetitorResponse(
        competitor_id=comp.competitor_id,
        workspace_id=comp.workspace_id,
        name=comp.name,
        domain=comp.domain,
        description=comp.description,
        is_active=comp.is_active,
        created_at=comp.created_at,
    )


@router.get("", response_model=list[CompetitorResponse])
async def list_competitors(
    workspace_id: str,
    is_active: bool | None = Query(default=None, description="Filter by active status"),
    _ws: Workspace = Depends(get_current_workspace),
    service: CompetitorService = Depends(get_competitor_service),
) -> list[CompetitorResponse]:
    """List tracked competitors in workspace."""
    competitors = await service.list_competitors(workspace_id=workspace_id, is_active=is_active)
    return [
        CompetitorResponse(
            competitor_id=c.competitor_id,
            workspace_id=c.workspace_id,
            name=c.name,
            domain=c.domain,
            description=c.description,
            is_active=c.is_active,
            created_at=c.created_at,
        )
        for c in competitors
    ]


@router.get("/{competitor_id}", response_model=CompetitorResponse)
async def get_competitor(
    workspace_id: str,
    competitor_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: CompetitorService = Depends(get_competitor_service),
) -> CompetitorResponse:
    """Get competitor details with workspace boundary verification."""
    c = await service.get_competitor(workspace_id=workspace_id, competitor_id=competitor_id)
    return CompetitorResponse(
        competitor_id=c.competitor_id,
        workspace_id=c.workspace_id,
        name=c.name,
        domain=c.domain,
        description=c.description,
        is_active=c.is_active,
        created_at=c.created_at,
    )


@router.patch("/{competitor_id}", response_model=CompetitorResponse)
async def update_competitor(
    workspace_id: str,
    competitor_id: str,
    payload: CompetitorUpdateRequest,
    _ws: Workspace = Depends(get_current_workspace),
    service: CompetitorService = Depends(get_competitor_service),
) -> CompetitorResponse:
    """Update competitor details."""
    c = await service.update_competitor(
        workspace_id=workspace_id,
        competitor_id=competitor_id,
        name=payload.name,
        description=payload.description,
        is_active=payload.is_active,
    )
    return CompetitorResponse(
        competitor_id=c.competitor_id,
        workspace_id=c.workspace_id,
        name=c.name,
        domain=c.domain,
        description=c.description,
        is_active=c.is_active,
        created_at=c.created_at,
    )
