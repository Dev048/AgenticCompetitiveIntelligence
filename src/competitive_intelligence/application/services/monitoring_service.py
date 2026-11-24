"""Application use-case service for continuous source monitoring."""

from datetime import datetime, timezone
import hashlib
from uuid import uuid4

from competitive_intelligence.application.interfaces import (
    BaseSourceRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.application.services.workflow_service import (
    WorkflowExecutionService,
)
from competitive_intelligence.core.errors import (
    EntityNotFoundError,
    ValidationError,
    WorkspaceIsolationError,
)
import time

from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.core.metrics import track_monitoring_job
from competitive_intelligence.monitoring.interfaces import (
    BaseMonitoringLeaseRepository,
    BaseSourceMonitoringService,
)
from competitive_intelligence.orchestration.models import (
    TriggerType,
    WorkflowInputs,
    WorkflowRun,
    WorkflowStageType,
)
from competitive_intelligence.source_acquisition.models import (
    AcquisitionResult,
    FetchStatus,
    Source,
)
from competitive_intelligence.source_acquisition.orchestrator import (
    AcquisitionOrchestrator,
)

logger = get_logger(__name__)


class SourceMonitoringService(BaseSourceMonitoringService):
    """Coordinates single-source monitoring execution, acquisition, status tracking, and pipeline dispatch."""

    def __init__(
        self,
        source_repo: BaseSourceRepository,
        lease_repo: BaseMonitoringLeaseRepository,
        workflow_service: WorkflowExecutionService,
        acquisition_orchestrator: AcquisitionOrchestrator | None = None,
        workspace_repo: BaseWorkspaceRepository | None = None,
    ) -> None:
        self._source_repo = source_repo
        self._lease_repo = lease_repo
        self._workflow_service = workflow_service
        self._acquisition_orchestrator = (
            acquisition_orchestrator or AcquisitionOrchestrator()
        )
        self._workspace_repo = workspace_repo

    async def monitor_source(
        self,
        source_id: str,
        lease_owner: str,
        correlation_id: str | None = None,
        target_stop_stage: WorkflowStageType = WorkflowStageType.INSIGHT_GENERATION,
    ) -> WorkflowRun | None:
        """Execute scheduled monitoring for a claimed source.

        1. Validates source and confirms lease ownership.
        2. Acquires content through AcquisitionOrchestrator outside DB transaction.
        3. Updates source health and fetch telemetry.
        4. On success, submits workflow with TriggerType.SCHEDULED_MONITORING.
        5. Releases lease in both success and failure paths.
        """
        corr_id = correlation_id or str(uuid4())
        t0 = time.perf_counter()

        # 1. Validate source
        source = await self._source_repo.get_by_id(source_id)
        if source is None:
            logger.error("Monitoring source not found", source_id=source_id, correlation_id=corr_id)
            await self._lease_repo.release_source_lease(source_id, lease_owner)
            raise EntityNotFoundError(f"Source '{source_id}' not found")

        workspace_id = source.identity.workspace_id
        company_id = source.identity.company_id

        # Workspace verification if repository provided
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                await self._lease_repo.release_source_lease(source_id, lease_owner)
                raise WorkspaceIsolationError(f"Workspace '{workspace_id}' not found")

        # 2. Confirm lease ownership
        lease = await self._lease_repo.get_active_lease(source_id)
        if lease is None or lease.lease_owner != lease_owner or lease.is_expired:
            logger.warning(
                "Cannot monitor source without active lease ownership",
                source_id=source_id,
                lease_owner=lease_owner,
                correlation_id=corr_id,
            )
            return None

        attempt_count = lease.attempt_count

        logger.info(
            "SOURCE_CLAIMED",
            workspace_id=workspace_id,
            company_id=company_id,
            source_id=source_id,
            lease_owner=lease_owner,
            attempt=attempt_count,
            correlation_id=corr_id,
        )

        # 3. Perform acquisition (outside any DB transaction)
        logger.info(
            "ACQUISITION_STARTED",
            source_id=source_id,
            url=source.location.url_or_identifier,
            correlation_id=corr_id,
        )

        now = datetime.now(timezone.utc)
        try:
            acq_result: AcquisitionResult = await self._acquisition_orchestrator.acquire(source)
        except Exception as exc:
            logger.error(
                "Acquisition raised unhandled exception",
                source_id=source_id,
                error=str(exc),
                correlation_id=corr_id,
            )
            acq_result = AcquisitionResult(
                status=FetchStatus.FAILURE.value,  # type: ignore[arg-type]
                error_message=str(exc),
                acquired_at=now,
            )

        # 4. Process Acquisition Outcome
        if not acq_result.is_success or not acq_result.content:
            logger.warning(
                "ACQUISITION_FAILED",
                source_id=source_id,
                status=acq_result.status.value,
                error=acq_result.error_message,
                correlation_id=corr_id,
            )

            # Update fetch failure telemetry
            await self._source_repo.update_fetch_status(
                source_id=source_id,
                last_fetch_status=acq_result.status.value,
                consecutive_failure_count=source.acquisition.consecutive_failure_count + 1,
                last_checked_at=now,
            )

            # Release lease
            await self._lease_repo.release_source_lease(source_id, lease_owner)
            logger.info("LEASE_RELEASED", source_id=source_id, lease_owner=lease_owner, correlation_id=corr_id)
            
            # Record monitoring failure metric
            track_monitoring_job(
                status="failed",
                duration_seconds=time.perf_counter() - t0,
                platform=getattr(source.location, "platform", "web"),
                category=source.classification.category.value,
                failure_reason=acq_result.error_message or "acquisition_failed",
            )
            return None

        logger.info(
            "ACQUISITION_SUCCEEDED",
            source_id=source_id,
            bytes=len(acq_result.content),
            correlation_id=corr_id,
        )

        # Update fetch success telemetry
        await self._source_repo.update_fetch_status(
            source_id=source_id,
            last_fetch_status=FetchStatus.SUCCESS.value,
            consecutive_failure_count=0,
            last_checked_at=now,
        )

        # 5. Compute deterministic idempotency key for this source observation
        content_hash = hashlib.sha256(acq_result.content.encode("utf-8")).hexdigest()[:16]
        lease_time = int(lease.lease_acquired_at.timestamp())
        idempotency_key = f"sched_mon:{source_id}:{lease_time}:{attempt_count}:{content_hash}"

        # 6. Dispatch to WorkflowExecutionService with SCHEDULED_MONITORING
        inputs = WorkflowInputs(
            workspace_id=workspace_id,
            company_id=company_id,
            raw_document_content=acq_result.content,
            source_url=source.location.url_or_identifier,
            metadata={
                "source_id": source_id,
                "source_category": source.classification.category.value,
                "monitoring_attempt": attempt_count,
                "correlation_id": corr_id,
                "idempotency_key": idempotency_key,
            },
        )

        try:
            workflow_run = await self._workflow_service.submit_workflow_run(
                workspace_id=workspace_id,
                inputs=inputs,
                target_stop_stage=target_stop_stage,
                trigger_type=TriggerType.SCHEDULED_MONITORING,
            )
            track_monitoring_job(
                status="success",
                duration_seconds=time.perf_counter() - t0,
                platform=getattr(source.location, "platform", "web"),
                category=source.classification.category.value,
            )
        except Exception as exc:
            track_monitoring_job(
                status="failed",
                duration_seconds=time.perf_counter() - t0,
                platform=getattr(source.location, "platform", "web"),
                category=source.classification.category.value,
                failure_reason="workflow_submission_failed",
            )
            raise
        finally:
            # Release lease ONLY after workflow run is durably submitted (or on submission exception)
            await self._lease_repo.release_source_lease(source_id, lease_owner)
            logger.info(
                "LEASE_RELEASED",
                source_id=source_id,
                lease_owner=lease_owner,
                correlation_id=corr_id,
            )

        logger.info(
            "WORKFLOW_SUBMITTED",
            workspace_id=workspace_id,
            company_id=company_id,
            source_id=source_id,
            workflow_run_id=workflow_run.run_id,
            trigger=TriggerType.SCHEDULED_MONITORING.value,
            correlation_id=corr_id,
            idempotency_key=idempotency_key,
        )

        return workflow_run

