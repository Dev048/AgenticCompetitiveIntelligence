"""Signal Detection and History stage wrapper."""

from datetime import datetime, timezone
import hashlib
import time

from competitive_intelligence.application.interfaces import BaseIntelligenceRepository
from competitive_intelligence.intelligence import (
    BaseEventSequenceTracker,
    BaseSignalDetector,
    DeterministicEventSequenceTracker,
    DeterministicSignalDetector,
    InferredSignal,
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


class SignalDetectionStage(BaseWorkflowStage):
    """Synthesizes high-level InferredSignals and tracks EventSequences from grounded events."""

    def __init__(
        self,
        signal_detector: BaseSignalDetector | None = None,
        sequence_tracker: BaseEventSequenceTracker | None = None,
        intelligence_repo: BaseIntelligenceRepository | None = None,
    ) -> None:
        self.signal_detector = signal_detector or DeterministicSignalDetector()
        self.sequence_tracker = sequence_tracker or DeterministicEventSequenceTracker()
        self.intelligence_repo = intelligence_repo

    @property
    def stage_type(self) -> WorkflowStageType:
        return WorkflowStageType.SIGNAL_DETECTION

    def compute_idempotency_key(self, context: StageExecutionContext) -> str:
        ev_ids_str = ",".join(sorted(e.event_id for e in context.events))
        return f"{self.stage_type.value}:{context.inputs.workspace_id}:{context.inputs.company_id}:{hashlib.sha256(ev_ids_str.encode('utf-8')).hexdigest()}"

    async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        record = StageExecutionRecord(
            stage=self.stage_type,
            status=StageStatus.RUNNING,
            started_at=started_at,
            idempotency_key=self.compute_idempotency_key(context),
            input_summary={"input_events_count": len(context.events)},
        )

        try:
            signals: list[InferredSignal] = []
            if context.events:
                signals = await self.signal_detector.detect_signals(
                    workspace_id=context.inputs.workspace_id,
                    company_id=context.inputs.company_id,
                    events=context.events,
                )

            # Derive deterministic IDs to ensure idempotency across retry re-executions
            deterministic_signals: list[InferredSignal] = []
            for sig in signals:
                ev_ids_key = ",".join(sorted(sig.supporting_event_ids))
                det_id = f"sig_{hashlib.sha256(f'{context.inputs.workspace_id}:{context.inputs.company_id}:{sig.signal_type.value}:{sig.subject}:{ev_ids_key}'.encode('utf-8')).hexdigest()[:16]}"
                det_sig = sig.model_copy(update={"signal_id": det_id})
                deterministic_signals.append(det_sig)
            signals = deterministic_signals

            context.signals = signals

            for sig in signals:
                context.run.produced_outputs.signal_ids.append(sig.signal_id)

            # Persist downstream signals if repository is configured
            if self.intelligence_repo is not None:
                for sig in signals:
                    await self.intelligence_repo.create_signal(sig)

            record.status = StageStatus.COMPLETED
            record.output_ids = {
                "detected_signals_count": len(signals),
                "signal_ids": [sig.signal_id for sig in signals],
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
