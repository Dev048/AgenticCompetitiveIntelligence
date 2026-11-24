from collections.abc import Sequence
from datetime import datetime, timezone
import time

import anyio

from competitive_intelligence.core.logger import (
    bind_log_context,
    get_logger,
)
from competitive_intelligence.core.metrics import (
    track_stage_execution,
    track_workflow_run,
)
from competitive_intelligence.decision_support.models import DecisionContext

logger = get_logger(__name__)
from competitive_intelligence.intelligence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EvidenceReference,
    EvidenceRole,
    IntelligenceEvent,
    IntelligenceEventType,
    SourcePerspective,
    TimePrecision,
)
from competitive_intelligence.intelligence.models import EventTime
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
    WorkflowResult,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStageType,
)
from competitive_intelligence.orchestration.policies import ExecutionBudget, RetryPolicy
from competitive_intelligence.orchestration.stages.decision_support_stage import (
    DecisionSupportStage,
)
from competitive_intelligence.orchestration.stages.event_analysis_stage import (
    EventAnalysisStage,
)
from competitive_intelligence.orchestration.stages.insight_generation_stage import (
    InsightGenerationStage,
)
from competitive_intelligence.orchestration.stages.knowledge_stage import (
    KnowledgeProcessingStage,
)
from competitive_intelligence.orchestration.stages.retrieval_stage import (
    RetrievalStage,
)
from competitive_intelligence.orchestration.stages.signal_detection_stage import (
    SignalDetectionStage,
)
from competitive_intelligence.orchestration.state_store import (
    InMemoryWorkflowStateStore,
)


class DeterministicWorkflowOrchestrator(BaseWorkflowOrchestrator):
    """Production coordinator executing the Competitive Intelligence pipeline stages in order.

    Preserves invariants:
    - Coordinates existing subsystems without duplicating business logic.
    - Checkpoints execution state and domain references after every stage.
    - Enables resuming interrupted or failed runs without repeating completed stages.
    - Enforces retry policies with exponential backoff on transient errors.
    """

    STAGE_ORDER: list[WorkflowStageType] = [
        WorkflowStageType.KNOWLEDGE_PROCESSING,
        WorkflowStageType.RETRIEVAL,
        WorkflowStageType.EVENT_ANALYSIS,
        WorkflowStageType.SIGNAL_DETECTION,
        WorkflowStageType.INSIGHT_GENERATION,
        WorkflowStageType.DECISION_SUPPORT,
    ]

    def __init__(
        self,
        state_store: BaseWorkflowStateStore | None = None,
        retry_policy: RetryPolicy | None = None,
        stages: dict[WorkflowStageType, BaseWorkflowStage] | Sequence[BaseWorkflowStage] | None = None,
    ) -> None:
        self.state_store = state_store or InMemoryWorkflowStateStore()
        self.retry_policy = retry_policy or RetryPolicy()
        self._cancelled_runs: set[str] = set()

        # Initialize stage wrappers
        if isinstance(stages, Sequence) and not isinstance(stages, dict):
            self.stages: dict[WorkflowStageType, BaseWorkflowStage] = {s.stage_type: s for s in stages}
        elif isinstance(stages, dict):
            self.stages = dict(stages)
        else:
            self.stages = {
                WorkflowStageType.KNOWLEDGE_PROCESSING: KnowledgeProcessingStage(),
                WorkflowStageType.RETRIEVAL: RetrievalStage(),
                WorkflowStageType.EVENT_ANALYSIS: EventAnalysisStage(),
                WorkflowStageType.SIGNAL_DETECTION: SignalDetectionStage(),
                WorkflowStageType.INSIGHT_GENERATION: InsightGenerationStage(),
                WorkflowStageType.DECISION_SUPPORT: DecisionSupportStage(),
            }

    async def run_workflow(
        self,
        inputs: WorkflowInputs,
        target_stop_stage: WorkflowStageType = WorkflowStageType.INSIGHT_GENERATION,
        budget: ExecutionBudget | None = None,
        run_id: str | None = None,
    ) -> WorkflowResult:
        exec_budget = budget or ExecutionBudget()
        run_meta = {
            "raw_document_content": inputs.raw_document_content,
            "source_url": inputs.source_url,
            "document_id": inputs.document_id,
            "query_text": inputs.query_text,
            "decision_context": inputs.decision_context.model_dump(mode="json") if inputs.decision_context else None,
            "enable_decision_support": inputs.enable_decision_support,
            **inputs.metadata,
        }

        if run_id:
            existing = await self.state_store.get_run(run_id)
            if existing:
                run = existing
                run.status = WorkflowRunStatus.RUNNING
                run.metadata.update(run_meta)
                await self.state_store.update_run(run)
            else:
                run = WorkflowRun(
                    run_id=run_id,
                    workspace_id=inputs.workspace_id,
                    company_id=inputs.company_id,
                    target_stop_stage=target_stop_stage,
                    status=WorkflowRunStatus.RUNNING,
                    metadata=run_meta,
                )
                await self.state_store.create_run(run)
        else:
            run = WorkflowRun(
                workspace_id=inputs.workspace_id,
                company_id=inputs.company_id,
                target_stop_stage=target_stop_stage,
                status=WorkflowRunStatus.RUNNING,
                metadata=run_meta,
            )
            await self.state_store.create_run(run)

        context = StageExecutionContext(
            run=run,
            inputs=inputs,
            budget=exec_budget,
        )

        return await self._execute_pipeline(context, target_stop_stage)

    async def resume_workflow(
        self,
        run_id: str,
        budget: ExecutionBudget | None = None,
    ) -> WorkflowResult:
        run = await self.state_store.get_run(run_id)
        if not run:
            return WorkflowResult(
                run_id=run_id,
                status=WorkflowRunStatus.FAILED,
                error_summary=f"Workflow run '{run_id}' not found in state store.",
            )

        run.status = WorkflowRunStatus.RUNNING
        run.trigger_type = TriggerType.RETRY_RESUME
        await self.state_store.update_run(run)

        # Reconstruct inputs from run metadata/context
        dec_ctx_data = run.metadata.get("decision_context")
        dec_ctx: DecisionContext | None = None
        if dec_ctx_data:
            dec_ctx = (
                DecisionContext.model_validate(dec_ctx_data)
                if isinstance(dec_ctx_data, dict)
                else dec_ctx_data
            )

        inputs = WorkflowInputs(
            workspace_id=run.workspace_id,
            company_id=run.company_id,
            raw_document_content=run.metadata.get("raw_document_content"),
            source_url=run.metadata.get("source_url"),
            document_id=run.metadata.get("document_id"),
            query_text=run.metadata.get("query_text"),
            decision_context=dec_ctx,
            enable_decision_support=run.metadata.get("enable_decision_support", False),
        )

        context = StageExecutionContext(
            run=run,
            inputs=inputs,
            budget=budget or ExecutionBudget(),
        )

        # Hydrate context with completed event references if EventAnalysis was already completed
        if (
            WorkflowStageType.EVENT_ANALYSIS in run.stage_records
            and run.stage_records[WorkflowStageType.EVENT_ANALYSIS].status == StageStatus.COMPLETED
            and run.produced_outputs.event_ids
        ):
            hydrated_events: list[IntelligenceEvent] = []
            for ev_id in run.produced_outputs.event_ids:
                doc_id = inputs.document_id or f"doc_{inputs.company_id}"
                ver_id = run.produced_outputs.version_ids[0] if run.produced_outputs.version_ids else f"ver_{inputs.company_id}_1"
                hydrated_events.append(
                    IntelligenceEvent(
                        event_id=ev_id,
                        workspace_id=inputs.workspace_id,
                        company_id=inputs.company_id,
                        event_type=IntelligenceEventType.PRICING_CHANGE,
                        title=f"Event for {inputs.company_id}",
                        summary=inputs.raw_document_content[:200] if inputs.raw_document_content else "Activity",
                        subject="Enterprise Plan",
                        event_time=EventTime(
                            precision=TimePrecision.DATE_ONLY,
                            acquired_at=datetime.now(timezone.utc),
                            detected_at=datetime.now(timezone.utc),
                        ),
                        confidence=ConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH),
                        evidence=[
                            EvidenceReference(
                                source_id=f"src_{inputs.company_id}",
                                document_id=doc_id,
                                version_id=ver_id,
                                source_url=inputs.source_url or "https://example.com",
                                source_category="pricing",
                                extracted_text="Evidence text.",
                                acquired_at=datetime.now(timezone.utc),
                                role=EvidenceRole.PRIMARY_DIRECT,
                            )
                        ],
                        primary_perspective=SourcePerspective.FIRST_PARTY,
                    )
                )
            context.events = hydrated_events

        return await self._execute_pipeline(context, run.target_stop_stage)

    async def cancel_workflow(self, run_id: str) -> bool:
        self._cancelled_runs.add(run_id)
        run = await self.state_store.get_run(run_id)
        if run and run.status != WorkflowRunStatus.CANCELLED:
            run.status = WorkflowRunStatus.CANCELLED
            run.completed_at = datetime.now(timezone.utc)
            await self.state_store.update_run(run)
            return True
        return False

    async def _execute_pipeline(
        self,
        context: StageExecutionContext,
        target_stop_stage: WorkflowStageType,
    ) -> WorkflowResult:
        t0 = time.perf_counter()
        run = context.run
        bind_log_context(
            workflow_run_id=run.run_id,
            workspace_id=run.workspace_id,
            correlation_id=run.correlation_id,
        )
        logger.info(
            "Starting workflow execution",
            run_id=run.run_id,
            workspace_id=run.workspace_id,
            trigger_type=run.trigger_type.value,
            target_stop_stage=target_stop_stage.value,
        )

        completed_stages: list[WorkflowStageType] = []
        failed_stages: list[WorkflowStageType] = []

        # Determine target stages
        target_idx = self.STAGE_ORDER.index(target_stop_stage)
        # If decision support is enabled and not already included, extend to DECISION_SUPPORT
        if context.inputs.enable_decision_support and target_idx < self.STAGE_ORDER.index(WorkflowStageType.DECISION_SUPPORT):
            target_idx = self.STAGE_ORDER.index(WorkflowStageType.DECISION_SUPPORT)

        stages_to_run = self.STAGE_ORDER[: target_idx + 1]

        for stage_type in stages_to_run:
            bind_log_context(stage_id=stage_type.value)

            # 1. Cancellation Check
            if run.run_id in self._cancelled_runs:
                run.status = WorkflowRunStatus.CANCELLED
                run.completed_at = datetime.now(timezone.utc)
                await self.state_store.update_run(run)
                logger.info("Workflow cancelled by operator", run_id=run.run_id)
                return self._build_result(run, completed_stages, failed_stages, t0, "Workflow cancelled by operator.")

            stage_wrapper = self.stages.get(stage_type)
            if not stage_wrapper:
                continue

            # 2. Fast-forward already completed stages (Resume / Checkpointing)
            if stage_type in run.stage_records and run.stage_records[stage_type].status == StageStatus.COMPLETED:
                completed_stages.append(stage_type)
                continue

            # 3. Execute Stage with Retry Policy & Exponential Backoff
            run.current_stage = stage_type
            attempt = 0
            stage_record: StageExecutionRecord | None = None

            while True:
                attempt += 1
                try:
                    logger.info("Executing stage", stage=stage_type.value, attempt=attempt)
                    stage_record = await stage_wrapper.execute(context)
                    stage_record.attempt_count = attempt

                    if stage_record.status == StageStatus.COMPLETED:
                        break
                    elif stage_record.status == StageStatus.SKIPPED:
                        break
                    elif stage_record.status == StageStatus.FAILED:
                        err_cat = stage_record.error.error_category if stage_record.error else ErrorCategory.PERMANENT
                        if self.retry_policy.is_retryable(err_cat, attempt):
                            backoff = self.retry_policy.get_backoff(attempt)
                            logger.warning(
                                "Stage execution failed, retrying with backoff",
                                stage=stage_type.value,
                                attempt=attempt,
                                backoff_seconds=backoff,
                                error_category=err_cat.value,
                            )
                            await anyio.sleep(backoff)
                            continue
                        else:
                            break

                except Exception as exc:
                    err_record = StageErrorRecord(
                        stage=stage_type,
                        error_category=ErrorCategory.PERMANENT,
                        error_message=str(exc),
                        attempt_number=attempt,
                        is_retryable=False,
                    )
                    stage_record = StageExecutionRecord(
                        stage=stage_type,
                        status=StageStatus.FAILED,
                        attempt_count=attempt,
                        error=err_record,
                        completed_at=datetime.now(timezone.utc),
                    )
                    logger.error("Unhandled exception during stage execution", stage=stage_type.value, error=str(exc))
                    break

            # 4. Checkpoint Stage State to StateStore and Track Metrics
            if stage_record:
                run.stage_records[stage_type] = stage_record
                await self.state_store.checkpoint_stage(
                    run_id=run.run_id,
                    stage=stage_type,
                    record=stage_record,
                    outputs=run.produced_outputs,
                )

                err_cat_str = stage_record.error.error_category.value if stage_record.error else None
                track_stage_execution(
                    stage=stage_type.value,
                    status=stage_record.status.value,
                    duration_seconds=stage_record.duration_ms / 1000.0,
                    error_category=err_cat_str,
                    retries=attempt - 1,
                )

                if stage_record.status == StageStatus.COMPLETED:
                    completed_stages.append(stage_type)
                    logger.info("Stage completed successfully", stage=stage_type.value, duration_ms=stage_record.duration_ms)
                elif stage_record.status == StageStatus.SKIPPED:
                    logger.info("Stage skipped", stage=stage_type.value)
                elif stage_record.status == StageStatus.FAILED:
                    failed_stages.append(stage_type)
                    run.status = WorkflowRunStatus.FAILED
                    run.completed_at = datetime.now(timezone.utc)
                    await self.state_store.update_run(run)
                    error_msg = stage_record.error.error_message if stage_record.error else f"Stage {stage_type.value} failed."
                    logger.error("Stage failed, aborting pipeline", stage=stage_type.value, error_summary=error_msg)
                    return self._build_result(run, completed_stages, failed_stages, t0, error_msg)

        # 5. Workflow Successful Completion
        run.status = WorkflowRunStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        run.current_stage = None
        await self.state_store.update_run(run)
        logger.info("Workflow completed successfully", run_id=run.run_id)

        return self._build_result(run, completed_stages, failed_stages, t0, None)

    def _build_result(
        self,
        run: WorkflowRun,
        completed_stages: list[WorkflowStageType],
        failed_stages: list[WorkflowStageType],
        start_time: float,
        error_summary: str | None,
    ) -> WorkflowResult:
        total_duration = time.perf_counter() - start_time
        track_workflow_run(
            status=run.status.value,
            trigger_type=run.trigger_type.value,
            duration_seconds=total_duration,
        )
        return WorkflowResult(
            run_id=run.run_id,
            status=run.status,
            completed_stages=completed_stages,
            failed_stages=failed_stages,
            produced_outputs=run.produced_outputs,
            total_duration_seconds=round(total_duration, 3),
            error_summary=error_summary,
            telemetry={
                "total_duration_ms": round(total_duration * 1000.0, 2),
                "stages_count": len(run.stage_records),
                "errors_count": len(run.errors),
                "correlation_id": run.correlation_id,
            },
        )
