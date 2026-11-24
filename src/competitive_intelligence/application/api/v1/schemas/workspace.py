"""API schemas for Workspaces and Users."""

from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class UserResponse(BaseModel):
    """User account response DTO."""

    user_id: str
    email: str
    full_name: str | None = None
    is_active: bool
    created_at: datetime


class WorkspaceCreateRequest(BaseModel):
    """Payload for provisioning a workspace."""

    name: str = Field(min_length=1, max_length=255)
    owner_user_id: str = Field(min_length=1, max_length=36)


class WorkspaceResponse(BaseModel):
    """Workspace details response DTO."""

    workspace_id: str
    name: str
    owner_user_id: str
    created_at: datetime
