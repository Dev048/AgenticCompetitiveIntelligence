"""Workspaces API endpoints."""

from fastapi import APIRouter, Depends, Query, status

from competitive_intelligence.application.api.deps import get_workspace_service
from competitive_intelligence.application.api.v1.schemas.workspace import (
    WorkspaceCreateRequest,
    WorkspaceResponse,
)
from competitive_intelligence.application.services.workspace_service import WorkspaceService

router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    payload: WorkspaceCreateRequest,
    service: WorkspaceService = Depends(get_workspace_service),
) -> WorkspaceResponse:
    """Create a new workspace."""
    ws = await service.create_workspace(name=payload.name, owner_user_id=payload.owner_user_id)
    return WorkspaceResponse(
        workspace_id=ws.workspace_id,
        name=ws.name,
        owner_user_id=ws.owner_user_id,
        created_at=ws.created_at,
    )


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: str,
    service: WorkspaceService = Depends(get_workspace_service),
) -> WorkspaceResponse:
    """Get workspace details by ID."""
    ws = await service.get_workspace(workspace_id)
    return WorkspaceResponse(
        workspace_id=ws.workspace_id,
        name=ws.name,
        owner_user_id=ws.owner_user_id,
        created_at=ws.created_at,
    )


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    user_id: str | None = Query(default=None, description="User ID to list workspaces for"),
    service: WorkspaceService = Depends(get_workspace_service),
) -> list[WorkspaceResponse]:
    """List workspaces (optionally filtered by user)."""
    workspaces = await service.list_user_workspaces(owner_user_id=user_id)
    return [
        WorkspaceResponse(
            workspace_id=ws.workspace_id,
            name=ws.name,
            owner_user_id=ws.owner_user_id,
            created_at=ws.created_at,
        )
        for ws in workspaces
    ]
