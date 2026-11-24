"""Operational Orchestration subsystem package."""

from typing import Any

from competitive_intelligence.orchestration.interfaces import (
    BaseWorkflowOrchestrator,
    BaseWorkflowStage,
    BaseWorkflowStateStore,
    StageExecutionContext,
)
from competitive_intelligence.orchestration.models import (
    ErrorCategory,
    StageErrorRecord,
    StageExecutionRecord,
    StageStatus,
    TriggerType,
    WorkflowInputs,
    WorkflowProducedOutputs,
    WorkflowResult,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStageType,
)
from competitive_intelligence.orchestration.policies import (
    ExecutionBudget,
    RetryPolicy,
)

__all__ = [
    "BaseWorkflowOrchestrator",
    "BaseWorkflowStage",
    "BaseWorkflowStateStore",
    "DecisionSupportStage",
    "DeterministicWorkflowOrchestrator",
    "ErrorCategory",
    "EventAnalysisStage",
    "ExecutionBudget",
    "InMemoryWorkflowStateStore",
    "InsightGenerationStage",
    "KnowledgeProcessingStage",
    "RetrievalStage",
    "RetryPolicy",
    "SignalDetectionStage",
    "StageErrorRecord",
    "StageExecutionContext",
    "StageExecutionRecord",
    "StageStatus",
    "TriggerType",
    "WorkflowInputs",
    "WorkflowProducedOutputs",
    "WorkflowResult",
    "WorkflowRun",
    "WorkflowRunStatus",
    "WorkflowStageType",
]


def __getattr__(name: str) -> Any:
    if name == "DeterministicWorkflowOrchestrator":
        from competitive_intelligence.orchestration.orchestrator import (
            DeterministicWorkflowOrchestrator,
        )

        return DeterministicWorkflowOrchestrator
    if name in (
        "DecisionSupportStage",
        "EventAnalysisStage",
        "InsightGenerationStage",
        "KnowledgeProcessingStage",
        "RetrievalStage",
        "SignalDetectionStage",
    ):
        import competitive_intelligence.orchestration.stages as stages

        return getattr(stages, name)
    if name == "InMemoryWorkflowStateStore":
        from competitive_intelligence.orchestration.state_store import (
            InMemoryWorkflowStateStore,
        )

        return InMemoryWorkflowStateStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
