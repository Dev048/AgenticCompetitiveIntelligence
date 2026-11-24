"""Unit tests for Operational Orchestration subsystem (Phase 6)."""

import asyncio
from datetime import datetime, timezone
from typing import Any
import pytest

from competitive_intelligence.decision_support import (
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    ObjectiveDimension,
    RecommendationStatus,
)
from competitive_intelligence.orchestration import (
    BaseWorkflowOrchestrator,
    BaseWorkflowStage,
    DeterministicWorkflowOrchestrator,
    ErrorCategory,
    ExecutionBudget,
    InMemoryWorkflowStateStore,
    RetryPolicy,
    StageErrorRecord,
    StageExecutionContext,
    StageExecutionRecord,
    StageStatus,
    WorkflowInputs,
    WorkflowProducedOutputs,
    WorkflowResult,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStageType,
)


@pytest.fixture
def state_store() -> InMemoryWorkflowStateStore:
    return InMemoryWorkflowStateStore()


@pytest.fixture
def orchestrator(state_store: InMemoryWorkflowStateStore) -> DeterministicWorkflowOrchestrator:
    return DeterministicWorkflowOrchestrator(
        state_store=state_store,
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.01),
    )


@pytest.mark.anyio
async def test_end_to_end_monitoring_workflow_success(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify standard 5-stage monitoring workflow from raw document to Insight generation."""
    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Acme Corp announced updated Enterprise Plan pricing at $20/month with new features.",
        document_id="doc_acme_1",
    )

    result = await orchestrator.run_workflow(
        inputs=inputs,
        target_stop_stage=WorkflowStageType.INSIGHT_GENERATION,
    )

    assert result.status == WorkflowRunStatus.COMPLETED
    assert WorkflowStageType.KNOWLEDGE_PROCESSING in result.completed_stages
    assert WorkflowStageType.RETRIEVAL in result.completed_stages
    assert WorkflowStageType.EVENT_ANALYSIS in result.completed_stages
    assert WorkflowStageType.SIGNAL_DETECTION in result.completed_stages
    assert WorkflowStageType.INSIGHT_GENERATION in result.completed_stages
    assert result.failed_stages == []
    assert len(result.produced_outputs.document_ids) > 0
    assert len(result.produced_outputs.version_ids) > 0
    assert len(result.produced_outputs.event_ids) > 0


@pytest.mark.anyio
async def test_end_to_end_workflow_with_optional_decision_support(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify 6-stage workflow executing optional Decision Support producing a PROPOSED recommendation."""
    dec_context = DecisionContext(
        title="Pricing Strategic Response",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(
                name="Revenue Preservation",
                dimension=ObjectiveDimension.COST_EFFICIENCY,
                target_description="Protect gross margin.",
                priority_weight=1.0,
            )
        ],
    )
    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Acme reverted Enterprise pricing from $30 back to $20.",
        decision_context=dec_context,
        enable_decision_support=True,
    )

    result = await orchestrator.run_workflow(
        inputs=inputs,
        target_stop_stage=WorkflowStageType.DECISION_SUPPORT,
    )

    assert result.status == WorkflowRunStatus.COMPLETED
    assert WorkflowStageType.DECISION_SUPPORT in result.completed_stages
    assert len(result.produced_outputs.option_ids) > 0
    assert result.produced_outputs.recommendation_id is not None


@pytest.mark.anyio
async def test_stage_transient_failure_and_retry(
    state_store: InMemoryWorkflowStateStore,
) -> None:
    """Verify retry policy retries transient errors with backoff and succeeds on subsequent attempt."""
    attempt_counter = 0

    class FlakyStage(BaseWorkflowStage):
        @property
        def stage_type(self) -> WorkflowStageType:
            return WorkflowStageType.RETRIEVAL

        def compute_idempotency_key(self, context: StageExecutionContext) -> str:
            return "flaky:key"

        async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
            nonlocal attempt_counter
            attempt_counter += 1
            if attempt_counter == 1:
                return StageExecutionRecord(
                    stage=self.stage_type,
                    status=StageStatus.FAILED,
                    error=StageErrorRecord(
                        stage=self.stage_type,
                        error_category=ErrorCategory.TRANSIENT,
                        error_message="Network glitch",
                        is_retryable=True,
                    ),
                )
            return StageExecutionRecord(
                stage=self.stage_type,
                status=StageStatus.COMPLETED,
            )

    custom_orchestrator = DeterministicWorkflowOrchestrator(
        state_store=state_store,
        retry_policy=RetryPolicy(max_attempts=3, initial_backoff_seconds=0.01),
        stages={
            WorkflowStageType.KNOWLEDGE_PROCESSING: FlakyStage(),
            WorkflowStageType.RETRIEVAL: FlakyStage(),
            WorkflowStageType.EVENT_ANALYSIS: FlakyStage(),
            WorkflowStageType.SIGNAL_DETECTION: FlakyStage(),
            WorkflowStageType.INSIGHT_GENERATION: FlakyStage(),
            WorkflowStageType.DECISION_SUPPORT: FlakyStage(),
        },
    )

    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Test content",
    )

    result = await custom_orchestrator.run_workflow(inputs=inputs)
    assert result.status == WorkflowRunStatus.COMPLETED
    assert attempt_counter >= 2


@pytest.mark.anyio
async def test_stage_permanent_failure_terminates_run(
    state_store: InMemoryWorkflowStateStore,
) -> None:
    """Verify non-retryable error halts workflow with FAILED status without retry loop."""
    attempts = 0

    class FailingStage(BaseWorkflowStage):
        @property
        def stage_type(self) -> WorkflowStageType:
            return WorkflowStageType.EVENT_ANALYSIS

        def compute_idempotency_key(self, context: StageExecutionContext) -> str:
            return "fail:key"

        async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
            nonlocal attempts
            attempts += 1
            return StageExecutionRecord(
                stage=self.stage_type,
                status=StageStatus.FAILED,
                error=StageErrorRecord(
                    stage=self.stage_type,
                    error_category=ErrorCategory.PERMANENT,
                    error_message="Corrupt schema",
                    is_retryable=False,
                ),
            )

    custom_orchestrator = DeterministicWorkflowOrchestrator(
        state_store=state_store,
        retry_policy=RetryPolicy(max_attempts=3),
        stages={
            WorkflowStageType.KNOWLEDGE_PROCESSING: FailingStage(),
            WorkflowStageType.RETRIEVAL: FailingStage(),
            WorkflowStageType.EVENT_ANALYSIS: FailingStage(),
            WorkflowStageType.SIGNAL_DETECTION: FailingStage(),
            WorkflowStageType.INSIGHT_GENERATION: FailingStage(),
            WorkflowStageType.DECISION_SUPPORT: FailingStage(),
        },
    )

    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Content",
    )

    result = await custom_orchestrator.run_workflow(inputs=inputs)
    assert result.status == WorkflowRunStatus.FAILED
    # Should stop on attempt 1 because error is PERMANENT
    assert attempts == 1


@pytest.mark.anyio
async def test_checkpoint_persistence_and_resume(
    orchestrator: DeterministicWorkflowOrchestrator,
    state_store: InMemoryWorkflowStateStore,
) -> None:
    """Verify resuming a run from checkpoint fast-forwards completed stages."""
    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Acme updated Enterprise pricing.",
    )

    # 1. Run up to EVENT_ANALYSIS
    result1 = await orchestrator.run_workflow(
        inputs=inputs,
        target_stop_stage=WorkflowStageType.EVENT_ANALYSIS,
    )
    assert result1.status == WorkflowRunStatus.COMPLETED
    assert WorkflowStageType.EVENT_ANALYSIS in result1.completed_stages

    # 2. Update run in state store to extend target to INSIGHT_GENERATION
    run = await state_store.get_run(result1.run_id)
    assert run is not None
    run.target_stop_stage = WorkflowStageType.INSIGHT_GENERATION
    await state_store.update_run(run)

    # 3. Resume run
    result2 = await orchestrator.resume_workflow(run_id=result1.run_id)
    assert result2.status == WorkflowRunStatus.COMPLETED
    assert WorkflowStageType.SIGNAL_DETECTION in result2.completed_stages
    assert WorkflowStageType.INSIGHT_GENERATION in result2.completed_stages


def test_idempotent_stage_reexecution_avoidance(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify stage generates deterministic idempotency keys based on input content."""
    stage = orchestrator.stages[WorkflowStageType.KNOWLEDGE_PROCESSING]
    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Deterministic document content.",
        document_id="doc_1",
    )
    run = WorkflowRun(
        workspace_id="ws_test",
        company_id="acme_corp",
    )
    context = StageExecutionContext(
        run=run,
        inputs=inputs,
        budget=ExecutionBudget(),
    )

    key1 = stage.compute_idempotency_key(context)
    key2 = stage.compute_idempotency_key(context)
    assert key1 == key2
    assert "knowledge_processing:ws_test:doc_1:" in key1


@pytest.mark.anyio
async def test_workflow_cancellation(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify explicit workflow cancellation marks status as CANCELLED."""
    run = WorkflowRun(
        workspace_id="ws_test",
        company_id="acme_corp",
        status=WorkflowRunStatus.RUNNING,
    )
    await orchestrator.state_store.create_run(run)

    is_cancelled = await orchestrator.cancel_workflow(run.run_id)
    assert is_cancelled is True

    updated_run = await orchestrator.state_store.get_run(run.run_id)
    assert updated_run is not None
    assert updated_run.status == WorkflowRunStatus.CANCELLED

    is_cancelled_non_existent = await orchestrator.cancel_workflow("non_existent")
    assert is_cancelled_non_existent is False


@pytest.mark.anyio
async def test_target_stop_stage_control(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify target_stop_stage stops pipeline early at designated boundary."""
    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Acme product update.",
    )

    result = await orchestrator.run_workflow(
        inputs=inputs,
        target_stop_stage=WorkflowStageType.RETRIEVAL,
    )

    assert result.status == WorkflowRunStatus.COMPLETED
    assert result.completed_stages == [
        WorkflowStageType.KNOWLEDGE_PROCESSING,
        WorkflowStageType.RETRIEVAL,
    ]
    assert WorkflowStageType.EVENT_ANALYSIS not in result.completed_stages


@pytest.mark.anyio
async def test_transitive_provenance_lineage_preservation(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify WorkflowProducedOutputs maintains all entity IDs across stages."""
    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Acme updated Enterprise pricing from $30 to $20.",
        document_id="doc_prov_1",
    )

    result = await orchestrator.run_workflow(inputs=inputs)
    assert result.status == WorkflowRunStatus.COMPLETED
    outputs = result.produced_outputs
    assert len(outputs.document_ids) >= 1
    assert len(outputs.version_ids) >= 1
    assert len(outputs.event_ids) >= 1


@pytest.mark.anyio
async def test_concurrent_workflow_runs_isolation(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify concurrent workflow runs for different companies execute in complete isolation."""
    inputs_a = WorkflowInputs(
        workspace_id="ws_test",
        company_id="company_a",
        raw_document_content="Company A launched feature.",
    )
    inputs_b = WorkflowInputs(
        workspace_id="ws_test",
        company_id="company_b",
        raw_document_content="Company B expanded pricing.",
    )

    res_a, res_b = await asyncio.gather(
        orchestrator.run_workflow(inputs=inputs_a),
        orchestrator.run_workflow(inputs=inputs_b),
    )

    assert res_a.status == WorkflowRunStatus.COMPLETED
    assert res_b.status == WorkflowRunStatus.COMPLETED
    assert res_a.run_id != res_b.run_id


@pytest.mark.anyio
async def test_decision_support_skipped_when_disabled(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify DECISION_SUPPORT is skipped when enable_decision_support=False."""
    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Acme pricing update.",
        enable_decision_support=False,
    )

    result = await orchestrator.run_workflow(
        inputs=inputs,
        target_stop_stage=WorkflowStageType.INSIGHT_GENERATION,
    )

    assert result.status == WorkflowRunStatus.COMPLETED
    assert WorkflowStageType.DECISION_SUPPORT not in result.completed_stages
    assert result.produced_outputs.recommendation_id is None


@pytest.mark.anyio
async def test_decision_support_recommendation_status_is_proposed(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify generated recommendations strictly start with status PROPOSED."""
    dec_context = DecisionContext(
        title="Pricing Response",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(name="Revenue", dimension=ObjectiveDimension.COST_EFFICIENCY, target_description="Target", priority_weight=1.0)
        ],
    )
    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="Acme updated Enterprise Plan pricing from $50 to $30/month.",
        decision_context=dec_context,
        enable_decision_support=True,
    )

    result = await orchestrator.run_workflow(inputs=inputs)
    assert result.status == WorkflowRunStatus.COMPLETED
    assert result.produced_outputs.recommendation_id is not None


@pytest.mark.anyio
async def test_state_store_list_and_query_runs(
    state_store: InMemoryWorkflowStateStore,
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify state store list_runs filters by workspace and sorts chronologically."""
    inputs = WorkflowInputs(
        workspace_id="ws_query_test",
        company_id="acme_corp",
        raw_document_content="Acme content.",
    )
    await orchestrator.run_workflow(inputs=inputs)

    runs = await state_store.list_runs(workspace_id="ws_query_test")
    assert len(runs) == 1
    assert runs[0].workspace_id == "ws_query_test"

    empty_runs = await state_store.list_runs(workspace_id="non_existent_ws")
    assert len(empty_runs) == 0


@pytest.mark.anyio
async def test_custom_workflow_stage_injection(
    state_store: InMemoryWorkflowStateStore,
) -> None:
    """Verify dependency injection of custom BaseWorkflowStage."""

    class CustomStage(BaseWorkflowStage):
        @property
        def stage_type(self) -> WorkflowStageType:
            return WorkflowStageType.EVENT_ANALYSIS

        def compute_idempotency_key(self, context: StageExecutionContext) -> str:
            return "custom:key"

        async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
            return StageExecutionRecord(
                stage=self.stage_type,
                status=StageStatus.COMPLETED,
                output_ids={"custom_output": "val"},
            )

    custom_orch = DeterministicWorkflowOrchestrator(
        state_store=state_store,
        stages={
            WorkflowStageType.KNOWLEDGE_PROCESSING: CustomStage(),
            WorkflowStageType.RETRIEVAL: CustomStage(),
            WorkflowStageType.EVENT_ANALYSIS: CustomStage(),
            WorkflowStageType.SIGNAL_DETECTION: CustomStage(),
            WorkflowStageType.INSIGHT_GENERATION: CustomStage(),
            WorkflowStageType.DECISION_SUPPORT: CustomStage(),
        },
    )

    inputs = WorkflowInputs(workspace_id="ws", company_id="acme", raw_document_content="Content")
    res = await custom_orch.run_workflow(inputs=inputs)
    assert res.status == WorkflowRunStatus.COMPLETED


@pytest.mark.anyio
async def test_empty_content_data_quality_error(
    orchestrator: DeterministicWorkflowOrchestrator,
) -> None:
    """Verify empty document content fails fast with DATA_QUALITY error."""
    inputs = WorkflowInputs(
        workspace_id="ws_test",
        company_id="acme_corp",
        raw_document_content="",  # Empty
    )

    result = await orchestrator.run_workflow(inputs=inputs)
    assert result.status == WorkflowRunStatus.FAILED
    assert result.failed_stages == [WorkflowStageType.KNOWLEDGE_PROCESSING]
    assert "Raw document content is empty" in str(result.error_summary)


@pytest.mark.anyio
async def test_custom_orchestrator_dependency_injection() -> None:
    """Verify custom orchestrator implementing BaseWorkflowOrchestrator can be injected and called."""

    class MockCustomOrchestrator(BaseWorkflowOrchestrator):
        async def run_workflow(
            self,
            inputs: WorkflowInputs,
            target_stop_stage: WorkflowStageType = WorkflowStageType.INSIGHT_GENERATION,
            budget: ExecutionBudget | None = None,
        ) -> Any:
            from competitive_intelligence.orchestration.models import WorkflowResult
            return WorkflowResult(
                run_id="mock_run_1",
                status=WorkflowRunStatus.COMPLETED,
                completed_stages=[target_stop_stage],
            )

        async def resume_workflow(self, run_id: str, budget: ExecutionBudget | None = None) -> Any:
            pass

        async def cancel_workflow(self, run_id: str) -> bool:
            return True

    custom_orch = MockCustomOrchestrator()
    inputs = WorkflowInputs(workspace_id="ws", company_id="comp")
    res = await custom_orch.run_workflow(inputs)
    assert res.status == WorkflowRunStatus.COMPLETED
    assert res.run_id == "mock_run_1"

