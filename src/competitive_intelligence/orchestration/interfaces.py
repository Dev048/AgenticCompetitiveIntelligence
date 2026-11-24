"""Abstract interfaces and strongly typed stage execution contexts for Operational Orchestration."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from competitive_intelligence.decision_support.models import (
    DecisionOption,
    Recommendation,
)
from competitive_intelligence.intelligence.models import (
    InferredSignal,
    Insight,
    IntelligenceEvent,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChangeResult,
    DocumentVersion,
)
from competitive_intelligence.orchestration.models import (
    StageExecutionRecord,
    WorkflowInputs,
    WorkflowProducedOutputs,
    WorkflowResult,
    WorkflowRun,
    WorkflowStageType,
)
from competitive_intelligence.orchestration.policies import ExecutionBudget
from competitive_intelligence.retrieval_engine.models import RetrievalResult


class StageExecutionContext(BaseModel):
    """Strongly typed execution context passed to an individual workflow stage."""

    run: WorkflowRun
    inputs: WorkflowInputs
    budget: ExecutionBudget

    # Strongly typed domain entities accumulated during the run
    current_version: DocumentVersion | None = None
    change_result: ChangeResult | None = None
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    events: list[IntelligenceEvent] = Field(default_factory=list)
    signals: list[InferredSignal] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    decision_options: list[DecisionOption] = Field(default_factory=list)
    recommendation: Recommendation | None = None


class BaseWorkflowStage(ABC):
    """Abstract interface defining an isolated, typed workflow pipeline stage."""

    @property
    @abstractmethod
    def stage_type(self) -> WorkflowStageType:
        """The specific stage identifier."""
        pass

    @abstractmethod
    async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
        """Execute the stage logic using existing subsystem components."""
        pass

    @abstractmethod
    def compute_idempotency_key(self, context: StageExecutionContext) -> str:
        """Generate deterministic idempotency fingerprint for the stage input."""
        pass


class BaseWorkflowStateStore(ABC):
    """Abstract contract for persisting and querying workflow execution checkpoints and metadata."""

    @abstractmethod
    async def create_run(self, run: WorkflowRun) -> None:
        """Persist a new workflow run record."""
        pass

    @abstractmethod
    async def update_run(self, run: WorkflowRun) -> None:
        """Update existing workflow run state."""
        pass

    @abstractmethod
    async def get_run(self, run_id: str) -> WorkflowRun | None:
        """Retrieve a workflow run by ID."""
        pass

    @abstractmethod
    async def get_run_by_idempotency_key(
        self,
        workspace_id: str,
        idempotency_key: str,
    ) -> WorkflowRun | None:
        """Retrieve an existing workflow run by its workspace-scoped idempotency key."""
        pass

    @abstractmethod
    async def list_runs(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 50,
    ) -> list[WorkflowRun]:
        """List workflow runs for a workspace."""
        pass

    @abstractmethod
    async def checkpoint_stage(
        self,
        run_id: str,
        stage: WorkflowStageType,
        record: StageExecutionRecord,
        outputs: WorkflowProducedOutputs,
    ) -> None:
        """Checkpoint stage execution record and updated domain output references."""
        pass


class BaseWorkflowOrchestrator(ABC):
    """Abstract interface for coordinating end-to-end pipeline execution."""

    @abstractmethod
    async def run_workflow(
        self,
        inputs: WorkflowInputs,
        target_stop_stage: WorkflowStageType = WorkflowStageType.INSIGHT_GENERATION,
        budget: ExecutionBudget | None = None,
        run_id: str | None = None,
    ) -> WorkflowResult:
        """Execute workflow from inception to target stop stage."""
        pass

    @abstractmethod
    async def resume_workflow(
        self,
        run_id: str,
        budget: ExecutionBudget | None = None,
    ) -> WorkflowResult:
        """Resume a paused or failed workflow from its last valid checkpoint."""
        pass

    @abstractmethod
    async def cancel_workflow(self, run_id: str) -> bool:
        """Cancel an in-flight workflow run."""
        pass
