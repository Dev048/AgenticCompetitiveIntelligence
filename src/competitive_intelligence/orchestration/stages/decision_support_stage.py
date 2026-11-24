"""Optional Decision Support and Recommendation stage wrapper."""

from datetime import datetime, timezone
import hashlib
import time

from competitive_intelligence.application.interfaces import BaseDecisionRepository
from competitive_intelligence.decision_support import (
    BaseDecisionAnalyzer,
    DecisionOption,
    DeterministicDecisionAnalyzer,
    Recommendation,
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


class DecisionSupportStage(BaseWorkflowStage):
    """Optional stage evaluating decision options and justified recommendations from insights and context."""

    def __init__(
        self,
        decision_analyzer: BaseDecisionAnalyzer | None = None,
        decision_repo: BaseDecisionRepository | None = None,
    ) -> None:
        self.decision_analyzer = decision_analyzer or DeterministicDecisionAnalyzer()
        self.decision_repo = decision_repo

    @property
    def stage_type(self) -> WorkflowStageType:
        return WorkflowStageType.DECISION_SUPPORT

    def compute_idempotency_key(self, context: StageExecutionContext) -> str:
        dec_id = context.inputs.decision_context.decision_id if context.inputs.decision_context else "no_decision"
        ins_ids_str = ",".join(sorted(ins.insight_id for ins in context.insights))
        return f"{self.stage_type.value}:{context.inputs.workspace_id}:{dec_id}:{hashlib.sha256(ins_ids_str.encode('utf-8')).hexdigest()}"

    async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        record = StageExecutionRecord(
            stage=self.stage_type,
            status=StageStatus.RUNNING,
            started_at=started_at,
            idempotency_key=self.compute_idempotency_key(context),
            input_summary={
                "enabled": context.inputs.enable_decision_support,
                "has_decision_context": bool(context.inputs.decision_context),
            },
        )

        try:
            # 1. Skip if not explicitly enabled or missing decision context
            if not context.inputs.enable_decision_support or not context.inputs.decision_context:
                record.status = StageStatus.SKIPPED
                record.completed_at = datetime.now(timezone.utc)
                record.duration_ms = (time.perf_counter() - t0) * 1000.0
                return record

            # 2. Run Decision Analysis
            dec_ctx = context.inputs.decision_context
            if dec_ctx.workspace_id != context.inputs.workspace_id:
                dec_ctx = dec_ctx.model_copy(update={"workspace_id": context.inputs.workspace_id})

            options, recommendation = await self.decision_analyzer.analyze_and_recommend(
                context=dec_ctx,
                insights=context.insights,
                events=context.events,
            )

            # Derive deterministic IDs to ensure idempotency across retry re-executions
            deterministic_options: list[DecisionOption] = []
            option_id_map: dict[str, str] = {}
            for opt in options:
                det_opt_id = f"opt_{hashlib.sha256(f'{dec_ctx.workspace_id}:{dec_ctx.decision_id}:{opt.action_type.value}:{opt.title}'.encode('utf-8')).hexdigest()[:16]}"
                option_id_map[opt.option_id] = det_opt_id
                det_opt = opt.model_copy(update={"option_id": det_opt_id})
                deterministic_options.append(det_opt)
            options = deterministic_options

            if recommendation:
                new_rec_opt_id = option_id_map.get(recommendation.selected_option_id, recommendation.selected_option_id)
                ins_ids_key = ",".join(sorted(recommendation.supporting_insight_ids))
                det_rec_id = f"rec_{hashlib.sha256(f'{dec_ctx.workspace_id}:{dec_ctx.decision_id}:{new_rec_opt_id}:{ins_ids_key}'.encode('utf-8')).hexdigest()[:16]}"
                recommendation = recommendation.model_copy(
                    update={
                        "recommendation_id": det_rec_id,
                        "workspace_id": dec_ctx.workspace_id,
                        "selected_option_id": new_rec_opt_id,
                    }
                )

            context.decision_options = options
            context.recommendation = recommendation

            # Record produced output IDs
            for opt in options:
                context.run.produced_outputs.option_ids.append(opt.option_id)
            if recommendation:
                context.run.produced_outputs.recommendation_id = recommendation.recommendation_id

            # Persist downstream decision entities in strict dependency order
            if self.decision_repo is not None:
                await self.decision_repo.create_context(dec_ctx)
                for opt in options:
                    await self.decision_repo.create_option(opt)
                if recommendation:
                    await self.decision_repo.create_recommendation(recommendation)

            record.status = StageStatus.COMPLETED
            record.output_ids = {
                "generated_options_count": len(options),
                "has_recommendation": bool(recommendation),
                "recommendation_id": recommendation.recommendation_id if recommendation else None,
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
