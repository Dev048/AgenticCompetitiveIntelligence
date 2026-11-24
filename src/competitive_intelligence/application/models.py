"""Domain models for Application layer and multi-tenancy boundaries.

NOTE: These models are provisional placeholders for initial architectural scaffolding,
not finalized domain contracts. They will be refined during Subsystem 6 detailed design.
"""

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class User(BaseModel):
    """User account model."""

    user_id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    full_name: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Workspace(BaseModel):
    """Workspace boundary containing competitors, sources, and intelligence artifacts."""

    workspace_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    owner_user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TrackedCompetitor(BaseModel):
    """Competitor entity configured for active intelligence tracking."""

    competitor_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str
    name: str
    domain: str
    description: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
