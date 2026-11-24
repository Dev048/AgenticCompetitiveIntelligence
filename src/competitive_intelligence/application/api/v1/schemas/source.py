"""API schemas for Monitored Sources."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

from competitive_intelligence.source_acquisition.models import SourceCategory


class SourceCreateRequest(BaseModel):
    """Payload to register a new source under a competitor."""

    company_id: str = Field(min_length=1, max_length=36)
    display_name: str = Field(min_length=1, max_length=255)
    url_or_identifier: str = Field(min_length=1)
    category: SourceCategory = SourceCategory.NEWS
    frequency_seconds: int = Field(default=86400, ge=60)
    priority: int = Field(default=1, ge=1, le=10)


class SourceUpdateRequest(BaseModel):
    """Payload to update monitoring configuration for a source."""

    monitoring_enabled: bool | None = None
    frequency_seconds: int | None = Field(default=None, ge=60)
    priority: int | None = Field(default=None, ge=1, le=10)


class SourceResponse(BaseModel):
    """Monitored source details response DTO."""

    source_id: str
    workspace_id: str
    company_id: str
    display_name: str
    category: str
    platform: str
    content_type: str
    url_or_identifier: str
    access_pattern: str
    monitoring_enabled: bool
    frequency_seconds: int
    priority: int
    last_fetch_status: str
    consecutive_failure_count: int = 0
    last_checked_at: datetime | None = None
    created_at: datetime


class MonitoredSourceDiagnostic(BaseModel):
    """Operational diagnostic view for a single monitored source."""

    source_id: str
    display_name: str
    company_id: str
    category: str
    platform: str
    url_or_identifier: str
    monitoring_enabled: bool
    frequency_seconds: int
    last_fetch_status: str
    consecutive_failure_count: int
    last_checked_at: datetime | None = None
    active_lease: dict[str, Any] | None = None


class MonitoringDiagnosticsResponse(BaseModel):
    """Operational health and status report for workspace continuous monitoring."""

    workspace_id: str
    total_sources: int
    healthy_sources_count: int
    failing_sources_count: int
    disabled_sources_count: int
    active_leases_count: int
    scheduler_mode: str
    scheduler_running: bool
    sources: list[MonitoredSourceDiagnostic]



class SourceTriggerRequest(BaseModel):
    """Payload to trigger on-demand acquisition and intelligence workflow for a source."""

    target_stop_stage: str = Field(
        default="insight_generation",
        description="Stop stage: knowledge_processing, retrieval, event_analysis, signal_detection, insight_generation, decision_support",
    )
    raw_document_content: str | None = Field(
        default=None,
        description="Optional manual document content; if omitted, defaults to simulated acquisition of source identifier",
    )
