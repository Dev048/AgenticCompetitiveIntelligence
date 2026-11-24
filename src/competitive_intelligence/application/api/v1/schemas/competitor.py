"""API schemas for Tracked Competitors."""

from datetime import datetime
from pydantic import BaseModel, Field


class CompetitorCreateRequest(BaseModel):
    """Payload to register a competitor for active tracking."""

    name: str = Field(min_length=1, max_length=255)
    domain: str = Field(min_length=1, max_length=255)
    description: str | None = None


class CompetitorUpdateRequest(BaseModel):
    """Payload to update competitor status or metadata."""

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    is_active: bool | None = None


class CompetitorResponse(BaseModel):
    """Competitor response DTO."""

    competitor_id: str
    workspace_id: str
    name: str
    domain: str
    description: str | None = None
    is_active: bool
    created_at: datetime
