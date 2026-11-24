"""Domain models and execution state schemas for Operational Orchestration."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from competitive_intelligence.decision_support.models import DecisionContext


class WorkflowStageType(str, Enum):
    """Pipeline stages in the operational orchestration flow."""

    KNOWLEDGE_PROCESSING = "knowledge_processing"
    RETRIEVAL = "retrieval"
    EVENT_ANALYSIS = "event_analysis"
    SIGNAL_DETECTION = "signal_detection"
    INSIGHT_GENERATION = "insight_generation"
    DECISION_SUPPORT = "decision_support"


class WorkflowRunStatus(str, Enum):
    """Overall execution status of a workflow run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    PAUSED = "paused"


class StageStatus(str, Enum):
    """Execution status of an individual workflow stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TriggerType(str, Enum):
    """Mechanism initiating a workflow execution."""

    SCHEDULED_MONITORING = "scheduled_monitoring"
    SOURCE_OBSERVATION = "source_observation"
    MANUAL_USER_REQUEST = "manual_user_request"
    API_WEBHOOK = "api_webhook"
    RETRY_RESUME = "retry_resume"


class ErrorCategory(str, Enum):
    """Epistemic classification of workflow execution errors."""

    TRANSIENT = "transient"          # Network glitch, rate-limit (Retryable)
    PERMANENT = "permanent"          # Schema incompatibility, corrupt file (Non-retryable)
    VALIDATION = "validation"        # Structural grounding or constraint violation
    DEPENDENCY = "dependency"        # Missing upstream stage output
    DATA_QUALITY = "data_quality"    # Empty or unparseable source content
    TIMEOUT = "timeout"              # Stage execution budget exceeded
    CANCELLED = "cancelled"          # Explicitly interrupted by operator


class StageErrorRecord(BaseModel):
    """Detailed error record captured during stage execution."""

    stage: WorkflowStageType
    error_category: ErrorCategory
    error_message: str
    stack_trace: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attempt_number: int = 1
    is_retryable: bool = False


class StageExecutionRecord(BaseModel):
    """Audit record capturing the execution of a single workflow stage."""

    stage: WorkflowStageType
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0.0
    attempt_count: int = 0
    idempotency_key: str | None = None
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_ids: dict[str, Any] = Field(default_factory=dict)  # References to produced domain IDs
    error: StageErrorRecord | None = None

    @property
    def outputs(self) -> dict[str, Any]:
        """Convenience accessor for stage output payload dict."""
        return self.output_ids


class WorkflowProducedOutputs(BaseModel):
    """Transitive collection of domain entity IDs produced across stages."""

    document_ids: list[str] = Field(default_factory=list)
    version_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    retrieval_query_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    sequence_ids: list[str] = Field(default_factory=list)
    signal_ids: list[str] = Field(default_factory=list)
    insight_ids: list[str] = Field(default_factory=list)
    option_ids: list[str] = Field(default_factory=list)
    recommendation_id: str | None = None
    chunks_created_count: int = 0
    vector_points_upserted_count: int = 0


class WorkflowRun(BaseModel):
    """Complete, serializable execution state for an operational workflow run."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str = "competitive_intelligence_pipeline"
    workspace_id: str = "default_workspace"
    company_id: str = "default_company"
    trigger_type: TriggerType = TriggerType.MANUAL_USER_REQUEST
    status: WorkflowRunStatus = WorkflowRunStatus.PENDING

    current_stage: WorkflowStageType | None = None
    target_stop_stage: WorkflowStageType = WorkflowStageType.INSIGHT_GENERATION
    stage_records: dict[WorkflowStageType, StageExecutionRecord] = Field(default_factory=dict)

    produced_outputs: WorkflowProducedOutputs = Field(default_factory=WorkflowProducedOutputs)
    errors: list[StageErrorRecord] = Field(default_factory=list)

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    last_checkpointed_at: datetime | None = None

    correlation_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowInputs(BaseModel):
    """Typed inputs required for orchestrating a full or partial pipeline run."""

    workspace_id: str
    company_id: str
    raw_document_content: str | None = None
    source_url: str | None = None
    source_id: str | None = None
    document_url: str | None = None
    document_id: str | None = None
    query_text: str | None = None
    decision_context: DecisionContext | None = None
    enable_decision_support: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    """Summary result returned to caller upon workflow conclusion."""

    run_id: str
    status: WorkflowRunStatus
    completed_stages: list[WorkflowStageType] = Field(default_factory=list)
    failed_stages: list[WorkflowStageType] = Field(default_factory=list)
    produced_outputs: WorkflowProducedOutputs = Field(default_factory=WorkflowProducedOutputs)
    total_duration_seconds: float = 0.0
    error_summary: str | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)
