"""API schemas for Operational Workflows."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

from competitive_intelligence.orchestration.models import (
    TriggerType,
    WorkflowStageType,
)


class WorkflowRunCreateRequest(BaseModel):
    """Payload to trigger an operational workflow pipeline run."""

    company_id: str = Field(min_length=1, max_length=36)
    target_stop_stage: WorkflowStageType = WorkflowStageType.INSIGHT_GENERATION
    trigger_type: TriggerType = TriggerType.MANUAL_USER_REQUEST
    raw_document_content: str | None = None
    source_url: str | None = None
    source_id: str | None = None
    document_id: str | None = None
    query_text: str | None = None
    enable_decision_support: bool = False
    decision_context: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowRunSummaryResponse(BaseModel):
    """Immediate acknowledgment response (HTTP 202 Accepted)."""

    run_id: str
    workspace_id: str
    company_id: str
    status: str
    target_stop_stage: str
    started_at: datetime
    location: str


class StageExecutionRecordResponse(BaseModel):
    """Execution details for an individual workflow stage."""

    stage: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float
    attempt_count: int
    output_ids: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] | None = None


class WorkflowRunDetailResponse(BaseModel):
    """Full status and audit telemetry for a workflow run."""

    run_id: str
    workspace_id: str
    company_id: str
    workflow_id: str
    trigger_type: str
    status: str
    current_stage: str | None = None
    target_stop_stage: str
    stage_records: dict[str, StageExecutionRecordResponse] = Field(default_factory=dict)
    produced_outputs: dict[str, Any] = Field(default_factory=dict)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime | None = None
    last_checkpointed_at: datetime | None = None
    correlation_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
