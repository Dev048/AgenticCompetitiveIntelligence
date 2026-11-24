"""Strategic Insight Generation stage wrapper."""

from datetime import datetime, timezone
import hashlib
import time

from competitive_intelligence.application.interfaces import BaseIntelligenceRepository
from competitive_intelligence.intelligence import (
    BaseStrategicAnalyzer,
    DeterministicStrategicAnalyzer,
    Insight,
)
from competitive_intelligence.orchestration.interfaces import (
    BaseWorkflowStage,
    StageExecutionContext,
)
from competitive_intelligence.orchestration.models import (
    ErrorCategory,
    StageErrorRecord,
    StageExecutionRecord,
    StageStatus,
    WorkflowStageType,
)


class InsightGenerationStage(BaseWorkflowStage):
    """Synthesizes verified, grounded strategic Insights across signals and events."""

    def __init__(
        self,
        strategic_analyzer: BaseStrategicAnalyzer | None = None,
        intelligence_repo: BaseIntelligenceRepository | None = None,
    ) -> None:
        self.strategic_analyzer = strategic_analyzer or DeterministicStrategicAnalyzer()
        self.intelligence_repo = intelligence_repo

    @property
    def stage_type(self) -> WorkflowStageType:
        return WorkflowStageType.INSIGHT_GENERATION

    def compute_idempotency_key(self, context: StageExecutionContext) -> str:
        sig_ids_str = ",".join(sorted(s.signal_id for s in context.signals))
        ev_ids_str = ",".join(sorted(e.event_id for e in context.events))
        combined = f"{sig_ids_str}:{ev_ids_str}"
        return f"{self.stage_type.value}:{context.inputs.workspace_id}:{context.inputs.company_id}:{hashlib.sha256(combined.encode('utf-8')).hexdigest()}"

    async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        record = StageExecutionRecord(
            stage=self.stage_type,
            status=StageStatus.RUNNING,
            started_at=started_at,
            idempotency_key=self.compute_idempotency_key(context),
            input_summary={
                "signals_count": len(context.signals),
                "events_count": len(context.events),
            },
        )

        try:
            insights: list[Insight] = []
            if context.events or context.signals:
                insights = await self.strategic_analyzer.generate_insights(
                    workspace_id=context.inputs.workspace_id,
                    company_id=context.inputs.company_id,
                    signals=context.signals,
                    events=context.events,
                )

            # Derive deterministic IDs to ensure idempotency across retry re-executions
            deterministic_insights: list[Insight] = []
            for ins in insights:
                sig_ids_key = ",".join(sorted(ins.supporting_signal_ids))
                ev_ids_key = ",".join(sorted(ins.supporting_event_ids))
                det_id = f"ins_{hashlib.sha256(f'{context.inputs.workspace_id}:{context.inputs.company_id}:{ins.insight_type.value}:{ins.title}:{sig_ids_key}:{ev_ids_key}'.encode('utf-8')).hexdigest()[:16]}"
                det_ins = ins.model_copy(update={"insight_id": det_id})
                deterministic_insights.append(det_ins)
            insights = deterministic_insights

            context.insights = insights

            for ins in insights:
                context.run.produced_outputs.insight_ids.append(ins.insight_id)

            # Persist downstream insights if repository is configured
            if self.intelligence_repo is not None:
                for ins in insights:
                    await self.intelligence_repo.create_insight(ins)

            record.status = StageStatus.COMPLETED
            record.output_ids = {
                "generated_insights_count": len(insights),
                "insight_ids": [ins.insight_id for ins in insights],
            }
            record.completed_at = datetime.now(timezone.utc)
            record.duration_ms = (time.perf_counter() - t0) * 1000.0
            return record

        except Exception as exc:
            record.status = StageStatus.FAILED
            record.error = StageErrorRecord(
                stage=self.stage_type,
                error_category=ErrorCategory.VALIDATION,
                error_message=str(exc),
                is_retryable=False,
            )
            record.completed_at = datetime.now(timezone.utc)
            record.duration_ms = (time.perf_counter() - t0) * 1000.0
            return record
