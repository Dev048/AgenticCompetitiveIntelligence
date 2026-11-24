"""Event Analysis stage wrapper."""

from datetime import datetime, timezone
import hashlib
import time

from competitive_intelligence.application.interfaces import BaseIntelligenceRepository
from competitive_intelligence.intelligence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    DocumentChangeAnalyzer,
    EvidenceReference,
    EvidenceRole,
    IntelligenceEvent,
    IntelligenceEventType,
    RetrievedEvidenceAnalyzer,
    SourcePerspective,
    TimePrecision,
)
from competitive_intelligence.intelligence.models import EventTime
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


class EventAnalysisStage(BaseWorkflowStage):
    """Transforms document changes and retrieved search evidence into grounded IntelligenceEvent objects."""

    def __init__(
        self,
        document_analyzer: DocumentChangeAnalyzer | None = None,
        evidence_analyzer: RetrievedEvidenceAnalyzer | None = None,
        intelligence_repo: BaseIntelligenceRepository | None = None,
    ) -> None:
        self.document_analyzer = document_analyzer or DocumentChangeAnalyzer()
        self.evidence_analyzer = evidence_analyzer or RetrievedEvidenceAnalyzer()
        self.intelligence_repo = intelligence_repo

    @property
    def stage_type(self) -> WorkflowStageType:
        return WorkflowStageType.EVENT_ANALYSIS

    def compute_idempotency_key(self, context: StageExecutionContext) -> str:
        version_id = context.current_version.version_id if context.current_version else "no_version"
        retrieval_count = len(context.retrieval_results)
        return f"{self.stage_type.value}:{context.inputs.workspace_id}:{context.inputs.company_id}:{version_id}:{retrieval_count}"

    async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        record = StageExecutionRecord(
            stage=self.stage_type,
            status=StageStatus.RUNNING,
            started_at=started_at,
            idempotency_key=self.compute_idempotency_key(context),
            input_summary={
                "has_change_result": bool(context.change_result),
                "retrieval_count": len(context.retrieval_results),
            },
        )

        try:
            events: list[IntelligenceEvent] = []

            # 1. Document Change Analysis
            if context.change_result and context.current_version:
                doc_events = await self.document_analyzer.analyze_document_change(
                    change=context.change_result,
                    current_version=context.current_version,
                    previous_version=None,
                    context_evidence=context.retrieval_results,
                )
                events.extend(doc_events)

            # 2. Retrieved Evidence Analysis
            if context.retrieval_results:
                q_text = context.inputs.query_text or f"Updates for {context.inputs.company_id}"
                ev_events = await self.evidence_analyzer.analyze_retrieved_evidence(
                    query=q_text,
                    company_id=context.inputs.company_id,
                    workspace_id=context.inputs.workspace_id,
                    evidence=context.retrieval_results,
                )
                events.extend(ev_events)

            # Derive deterministic IDs to ensure idempotency across retry re-executions
            deterministic_events: list[IntelligenceEvent] = []
            for ev in events:
                det_id = f"ev_{hashlib.sha256(f'{context.inputs.workspace_id}:{context.inputs.company_id}:{ev.event_type.value}:{ev.title}:{ev.subject}'.encode('utf-8')).hexdigest()[:16]}"
                det_ev = ev.model_copy(update={"event_id": det_id})
                deterministic_events.append(det_ev)
            events = deterministic_events

            context.events = events

            # Record produced output IDs
            for ev in events:
                context.run.produced_outputs.event_ids.append(ev.event_id)

            # Persist downstream events if repository is configured
            if self.intelligence_repo is not None:
                for ev in events:
                    await self.intelligence_repo.create_event(ev)

            record.status = StageStatus.COMPLETED
            record.output_ids = {
                "extracted_events_count": len(events),
                "event_ids": [ev.event_id for ev in events],
                "diagnostics": getattr(self.evidence_analyzer, "last_diagnostics", {}),
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
