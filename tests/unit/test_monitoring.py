"""Unit tests for Continuous Monitoring Subsystem.

Tests:
1. SourceMonitoringLease domain model validation and expiration logic.
2. SourceMonitoringService coordination (acquisition, telemetry, lease release, workflow dispatch).
3. MonitoringWorker heartbeat loop, lease extension, and exception containment.
4. MonitoringScheduler polling cycle, batching, concurrency limits, and clean shutdown.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch
import pytest

from competitive_intelligence.application.interfaces import (
    BaseSourceRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.monitoring_service import (
    SourceMonitoringService,
)
from competitive_intelligence.application.services.workflow_service import (
    WorkflowExecutionService,
)
from competitive_intelligence.core.errors import (
    EntityNotFoundError,
    ValidationError,
    WorkspaceIsolationError,
)
from competitive_intelligence.monitoring.interfaces import (
    BaseMonitoringLeaseRepository,
)
from competitive_intelligence.monitoring.models import SourceMonitoringLease
from competitive_intelligence.monitoring.scheduler import MonitoringScheduler
from competitive_intelligence.monitoring.worker import MonitoringWorker
from competitive_intelligence.orchestration.interfaces import (
    BaseWorkflowOrchestrator,
    BaseWorkflowStateStore,
)
from competitive_intelligence.orchestration.models import (
    TriggerType,
    WorkflowInputs,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStageType,
)
from competitive_intelligence.orchestration.state_store import (
    InMemoryWorkflowStateStore,
)
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    AcquisitionResult,
    AcquisitionStatus,
    ContentType,
    DataSensitivity,
    FetchStatus,
    LifecycleStatus,
    MonitoringPolicy,
    PlatformType,
    RateLimitConstraints,
    Source,
    SourceCategory,
    SourceClassification,
    SourceGovernance,
    SourceIdentity,
    SourceLifecycle,
    SourceLocation,
    SourceReliability,
    SourceTier,
    ToSStatus,
)
from competitive_intelligence.source_acquisition.orchestrator import (
    AcquisitionOrchestrator,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _build_sample_source(
    source_id: str = "src_test_1",
    workspace_id: str = "ws_test",
    company_id: str = "comp_test",
    monitoring_enabled: bool = True,
    frequency_seconds: int = 3600,
    consecutive_failures: int = 0,
) -> Source:
    return Source(
        identity=SourceIdentity(
            source_id=source_id,
            workspace_id=workspace_id,
            company_id=company_id,
            display_name="Test Competitor Pricing",
        ),
        classification=SourceClassification(
            category=SourceCategory.PRICING,
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier="https://example.com/pricing",
            access_pattern=AccessPattern.PUBLIC_WEB,
            requires_javascript=False,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            consecutive_failure_count=consecutive_failures,
        ),
        monitoring=MonitoringPolicy(
            enabled=monitoring_enabled,
            frequency=frequency_seconds,
            priority=1,
        ),
        reliability=SourceReliability(
            credibility_score=0.9,
            source_tier=SourceTier.PRIMARY,
        ),
        governance=SourceGovernance(
            terms_of_service_status=ToSStatus.COMPLIANT,
            robots_txt_respected=True,
            data_sensitivity=DataSensitivity.PUBLIC,
        ),
        lifecycle=SourceLifecycle(
            status=LifecycleStatus.ACTIVE,
        ),
    )


# ==============================================================================
# 1. Domain Model Tests
# ==============================================================================


def test_source_monitoring_lease_model_valid() -> None:
    now = datetime.now(timezone.utc)
    lease = SourceMonitoringLease(
        source_id="src_1",
        lease_owner="worker_node_1",
        lease_acquired_at=now,
        lease_expires_at=now + timedelta(seconds=300),
        heartbeat_at=now,
        attempt_count=1,
    )
    assert lease.source_id == "src_1"
    assert lease.lease_owner == "worker_node_1"
    assert not lease.is_expired
    assert lease.attempt_count == 1


def test_source_monitoring_lease_model_expired() -> None:
    past = datetime.now(timezone.utc) - timedelta(seconds=10)
    lease = SourceMonitoringLease(
        source_id="src_1",
        lease_owner="worker_node_1",
        lease_acquired_at=past - timedelta(seconds=300),
        lease_expires_at=past,
        heartbeat_at=past,
    )
    assert lease.is_expired


def test_source_monitoring_lease_validation_errors() -> None:
    now = datetime.now(timezone.utc)
    naive_dt = datetime.now()  # naive

    with pytest.raises(ValueError, match="Field cannot be empty"):
        SourceMonitoringLease(
            source_id="",
            lease_owner="worker",
            lease_acquired_at=now,
            lease_expires_at=now + timedelta(seconds=300),
            heartbeat_at=now,
        )

    with pytest.raises(ValueError, match="Datetime must be timezone-aware"):
        SourceMonitoringLease(
            source_id="src_1",
            lease_owner="worker",
            lease_acquired_at=naive_dt,
            lease_expires_at=now + timedelta(seconds=300),
            heartbeat_at=now,
        )


# ==============================================================================
# 2. SourceMonitoringService Tests
# ==============================================================================


@pytest.mark.anyio
async def test_monitoring_service_successful_acquisition_flow() -> None:
    source = _build_sample_source(source_id="src_success_1")
    now = datetime.now(timezone.utc)
    lease = SourceMonitoringLease(
        source_id="src_success_1",
        lease_owner="worker_1",
        lease_acquired_at=now,
        lease_expires_at=now + timedelta(seconds=300),
        heartbeat_at=now,
    )

    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_by_id.return_value = source

    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.get_active_lease.return_value = lease
    mock_lease_repo.release_source_lease.return_value = True

    mock_acq = AsyncMock(spec=AcquisitionOrchestrator)
    mock_acq.acquire.return_value = AcquisitionResult(
        status=AcquisitionStatus.SUCCESS,
        content="<html><body>Enterprise Plan: $99/mo</body></html>",
        status_code=200,
        acquired_at=now,
    )

    expected_workflow_run = WorkflowRun(
        run_id="run_sched_1",
        workspace_id=source.identity.workspace_id,
        status=WorkflowRunStatus.COMPLETED,
        trigger_type=TriggerType.SCHEDULED_MONITORING,
    )

    mock_workflow_service = AsyncMock(spec=WorkflowExecutionService)
    mock_workflow_service.submit_workflow_run.return_value = expected_workflow_run

    service = SourceMonitoringService(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        workflow_service=mock_workflow_service,
        acquisition_orchestrator=mock_acq,
    )

    result = await service.monitor_source(
        source_id="src_success_1",
        lease_owner="worker_1",
        correlation_id="corr_test_123",
    )

    assert result is not None
    assert result.run_id == "run_sched_1"
    assert result.trigger_type == TriggerType.SCHEDULED_MONITORING

    # Verify telemetry updated with success
    mock_source_repo.update_fetch_status.assert_awaited_once_with(
        source_id="src_success_1",
        last_fetch_status=FetchStatus.SUCCESS.value,
        consecutive_failure_count=0,
        last_checked_at=ANY,
    )

    # Verify lease was released
    mock_lease_repo.release_source_lease.assert_awaited_once_with(
        "src_success_1", "worker_1"
    )

    # Verify workflow submitted with correct payload & trigger
    mock_workflow_service.submit_workflow_run.assert_awaited_once()
    submit_kwargs = mock_workflow_service.submit_workflow_run.call_args.kwargs
    assert submit_kwargs["workspace_id"] == source.identity.workspace_id
    assert submit_kwargs["trigger_type"] == TriggerType.SCHEDULED_MONITORING
    inputs = submit_kwargs["inputs"]
    assert inputs.workspace_id == source.identity.workspace_id
    assert inputs.company_id == source.identity.company_id
    assert inputs.metadata["source_id"] == "src_success_1"
    assert inputs.metadata["correlation_id"] == "corr_test_123"


@pytest.mark.anyio
async def test_monitoring_service_acquisition_failure_flow() -> None:
    source = _build_sample_source(source_id="src_fail_1", consecutive_failures=2)
    now = datetime.now(timezone.utc)
    lease = SourceMonitoringLease(
        source_id="src_fail_1",
        lease_owner="worker_1",
        lease_acquired_at=now,
        lease_expires_at=now + timedelta(seconds=300),
        heartbeat_at=now,
    )

    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_by_id.return_value = source

    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.get_active_lease.return_value = lease
    mock_lease_repo.release_source_lease.return_value = True

    mock_acq = AsyncMock(spec=AcquisitionOrchestrator)
    mock_acq.acquire.return_value = AcquisitionResult(
        status=AcquisitionStatus.FAILURE,
        error_message="HTTP 403 Forbidden",
        status_code=403,
        acquired_at=now,
    )

    mock_workflow_service = AsyncMock(spec=WorkflowExecutionService)

    service = SourceMonitoringService(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        workflow_service=mock_workflow_service,
        acquisition_orchestrator=mock_acq,
    )

    result = await service.monitor_source(
        source_id="src_fail_1",
        lease_owner="worker_1",
    )

    assert result is None

    # Telemetry updated with failure count incremented
    mock_source_repo.update_fetch_status.assert_awaited_once_with(
        source_id="src_fail_1",
        last_fetch_status=FetchStatus.FAILURE.value,
        consecutive_failure_count=3,
        last_checked_at=ANY,
    )

    # Lease released
    mock_lease_repo.release_source_lease.assert_awaited_once_with(
        "src_fail_1", "worker_1"
    )

    # Workflow service MUST NOT be called on failed acquisition
    mock_workflow_service.submit_workflow_run.assert_not_called()


@pytest.mark.anyio
async def test_monitoring_service_missing_or_expired_lease() -> None:
    source = _build_sample_source(source_id="src_no_lease")
    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_by_id.return_value = source

    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.get_active_lease.return_value = None  # No active lease

    mock_workflow_service = AsyncMock(spec=WorkflowExecutionService)
    mock_acq = AsyncMock(spec=AcquisitionOrchestrator)

    service = SourceMonitoringService(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        workflow_service=mock_workflow_service,
        acquisition_orchestrator=mock_acq,
    )

    result = await service.monitor_source(
        source_id="src_no_lease",
        lease_owner="worker_1",
    )

    assert result is None
    mock_acq.acquire.assert_not_called()
    mock_workflow_service.submit_workflow_run.assert_not_called()


@pytest.mark.anyio
async def test_monitoring_service_workspace_isolation() -> None:
    source = _build_sample_source(source_id="src_ws_iso", workspace_id="ws_invalid")
    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_by_id.return_value = source

    mock_workspace_repo = AsyncMock(spec=BaseWorkspaceRepository)
    mock_workspace_repo.get_by_id.return_value = None  # Workspace not found

    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_workflow_service = AsyncMock(spec=WorkflowExecutionService)

    service = SourceMonitoringService(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        workflow_service=mock_workflow_service,
        workspace_repo=mock_workspace_repo,
    )

    with pytest.raises(WorkspaceIsolationError):
        await service.monitor_source(source_id="src_ws_iso", lease_owner="worker_1")

    # Ensure lease released on isolation error
    mock_lease_repo.release_source_lease.assert_awaited_once_with(
        "src_ws_iso", "worker_1"
    )


# ==============================================================================
# 3. MonitoringWorker Tests
# ==============================================================================


@pytest.mark.anyio
async def test_monitoring_worker_heartbeat_and_execution() -> None:
    mock_service = AsyncMock(spec=SourceMonitoringService)
    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.renew_source_lease.return_value = True

    expected_run = WorkflowRun(
        run_id="run_w_1",
        workspace_id="ws_1",
        status=WorkflowRunStatus.COMPLETED,
        trigger_type=TriggerType.SCHEDULED_MONITORING,
    )
    mock_service.monitor_source.return_value = expected_run

    worker = MonitoringWorker(
        monitoring_service=mock_service,
        lease_repo=mock_lease_repo,
        heartbeat_interval_seconds=0.05,
        lease_extension_seconds=60,
    )

    # Let execution take slightly longer than heartbeat interval to trigger renewal
    async def _delayed_monitor(*args: Any, **kwargs: Any) -> WorkflowRun:
        await asyncio.sleep(0.12)
        return expected_run

    mock_service.monitor_source.side_effect = _delayed_monitor

    result = await worker.execute_monitoring_task(
        source_id="src_worker_1",
        lease_owner="worker_1",
    )

    assert result == expected_run
    # Heartbeat loop should have triggered renewal at least once
    assert mock_lease_repo.renew_source_lease.await_count >= 1


@pytest.mark.anyio
async def test_monitoring_worker_error_containment() -> None:
    mock_service = AsyncMock(spec=SourceMonitoringService)
    mock_service.monitor_source.side_effect = RuntimeError("Worker crashed!")
    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)

    worker = MonitoringWorker(
        monitoring_service=mock_service,
        lease_repo=mock_lease_repo,
        heartbeat_interval_seconds=1.0,
    )

    result = await worker.execute_monitoring_task(
        source_id="src_crash_1",
        lease_owner="worker_1",
    )

    # Worker must catch exception, release lease and return None without propagating crash
    assert result is None
    mock_lease_repo.release_source_lease.assert_awaited_once_with(
        "src_crash_1", "worker_1"
    )


# ==============================================================================
# 4. MonitoringScheduler Tests
# ==============================================================================


@pytest.mark.anyio
async def test_scheduler_run_once_dispatch() -> None:
    source_1 = _build_sample_source(source_id="src_due_1")
    source_2 = _build_sample_source(source_id="src_due_2")

    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_due_sources.return_value = [source_1, source_2]

    now = datetime.now(timezone.utc)
    lease_1 = SourceMonitoringLease(
        source_id="src_due_1",
        lease_owner="sched_worker",
        lease_acquired_at=now,
        lease_expires_at=now + timedelta(seconds=300),
        heartbeat_at=now,
    )

    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.reclaim_expired_leases.return_value = 1
    # First source claims successfully, second is already claimed (returns None)
    mock_lease_repo.claim_due_source.side_effect = [lease_1, None]

    mock_worker = AsyncMock(spec=MonitoringWorker)
    mock_worker.execute_monitoring_task.return_value = None

    scheduler = MonitoringScheduler(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        worker=mock_worker,
        worker_id="sched_worker",
        batch_size=5,
        max_concurrency=2,
    )

    claimed_count = await scheduler.run_once()

    assert claimed_count == 1
    mock_lease_repo.reclaim_expired_leases.assert_awaited_once()
    mock_source_repo.get_due_sources.assert_awaited_once()
    assert mock_lease_repo.claim_due_source.await_count == 2

    # Await dispatched task
    await asyncio.sleep(0.05)
    mock_worker.execute_monitoring_task.assert_awaited_once_with(
        source_id="src_due_1",
        lease_owner="sched_worker",
        correlation_id=ANY,
    )


@pytest.mark.anyio
async def test_scheduler_start_stop_lifecycle() -> None:
    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_due_sources.return_value = []
    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.reclaim_expired_leases.return_value = 0
    mock_worker = AsyncMock(spec=MonitoringWorker)

    scheduler = MonitoringScheduler(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        worker=mock_worker,
        poll_interval_seconds=0.05,
    )

    assert not scheduler.is_running
    await scheduler.start()
    assert scheduler.is_running

    await asyncio.sleep(0.12)
    assert mock_source_repo.get_due_sources.await_count >= 1

    await scheduler.stop()
    assert not scheduler.is_running


@pytest.mark.anyio
async def test_scheduler_concurrency_bound() -> None:
    sources = [_build_sample_source(source_id=f"src_cc_{i}") for i in range(4)]
    now = datetime.now(timezone.utc)
    leases = [
        SourceMonitoringLease(
            source_id=f"src_cc_{i}",
            lease_owner="worker_node",
            lease_acquired_at=now,
            lease_expires_at=now + timedelta(seconds=300),
            heartbeat_at=now,
        )
        for i in range(4)
    ]

    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_due_sources.return_value = sources
    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.reclaim_expired_leases.return_value = 0
    mock_lease_repo.claim_due_source.side_effect = leases

    max_concurrent_seen = 0
    current_concurrent = 0

    async def _mock_execute_task(*args: Any, **kwargs: Any) -> None:
        nonlocal max_concurrent_seen, current_concurrent
        current_concurrent += 1
        if current_concurrent > max_concurrent_seen:
            max_concurrent_seen = current_concurrent
        await asyncio.sleep(0.08)
        current_concurrent -= 1

    mock_worker = AsyncMock(spec=MonitoringWorker)
    mock_worker.execute_monitoring_task.side_effect = _mock_execute_task

    scheduler = MonitoringScheduler(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        worker=mock_worker,
        batch_size=10,
        max_concurrency=2,
    )

    claimed = await scheduler.run_once()
    assert claimed == 4

    # Await in-flight worker tasks
    await scheduler.stop()

    assert max_concurrent_seen <= 2
    assert mock_worker.execute_monitoring_task.await_count == 4


@pytest.mark.anyio
async def test_scheduler_worker_error_does_not_crash_scheduler() -> None:
    source = _build_sample_source(source_id="src_err_1")
    now = datetime.now(timezone.utc)
    lease = SourceMonitoringLease(
        source_id="src_err_1",
        lease_owner="worker_node",
        lease_acquired_at=now,
        lease_expires_at=now + timedelta(seconds=300),
        heartbeat_at=now,
    )

    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_due_sources.return_value = [source]
    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.reclaim_expired_leases.return_value = 0
    mock_lease_repo.claim_due_source.return_value = lease

    mock_worker = AsyncMock(spec=MonitoringWorker)
    mock_worker.execute_monitoring_task.side_effect = Exception("Worker exploded!")

    scheduler = MonitoringScheduler(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        worker=mock_worker,
    )

    claimed = await scheduler.run_once()
    assert claimed == 1

    # Should not raise exception
    await scheduler.stop()


# ==============================================================================
# 5. Handoff Durability & Workflow Idempotency Tests
# ==============================================================================


@pytest.mark.anyio
async def test_monitoring_service_handoff_ordering() -> None:
    """Verify that submit_workflow_run is executed BEFORE release_source_lease."""
    source = _build_sample_source(source_id="src_order_1")
    now = datetime.now(timezone.utc)
    lease = SourceMonitoringLease(
        source_id="src_order_1",
        lease_owner="worker_1",
        lease_acquired_at=now,
        lease_expires_at=now + timedelta(seconds=300),
        heartbeat_at=now,
    )

    call_order: list[str] = []

    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_by_id.return_value = source

    async def _mock_update_status(*args: Any, **kwargs: Any) -> bool:
        call_order.append("update_fetch_status")
        return True

    mock_source_repo.update_fetch_status.side_effect = _mock_update_status

    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.get_active_lease.return_value = lease

    async def _mock_release(*args: Any, **kwargs: Any) -> bool:
        call_order.append("release_source_lease")
        return True

    mock_lease_repo.release_source_lease.side_effect = _mock_release

    mock_acq = AsyncMock(spec=AcquisitionOrchestrator)
    mock_acq.acquire.return_value = AcquisitionResult(
        status=AcquisitionStatus.SUCCESS,
        content="Pricing update $79/mo",
        status_code=200,
        acquired_at=now,
    )

    expected_run = WorkflowRun(
        run_id="run_order_1",
        workspace_id=source.identity.workspace_id,
        status=WorkflowRunStatus.PENDING,
        trigger_type=TriggerType.SCHEDULED_MONITORING,
    )

    mock_workflow_service = AsyncMock(spec=WorkflowExecutionService)

    async def _mock_submit(*args: Any, **kwargs: Any) -> WorkflowRun:
        call_order.append("submit_workflow_run")
        return expected_run

    mock_workflow_service.submit_workflow_run.side_effect = _mock_submit

    service = SourceMonitoringService(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        workflow_service=mock_workflow_service,
        acquisition_orchestrator=mock_acq,
    )

    result = await service.monitor_source(
        source_id="src_order_1",
        lease_owner="worker_1",
    )

    assert result == expected_run
    assert call_order == [
        "update_fetch_status",
        "submit_workflow_run",
        "release_source_lease",
    ]


@pytest.mark.anyio
async def test_monitoring_service_generates_deterministic_idempotency_key() -> None:
    source = _build_sample_source(source_id="src_idem_1")
    now = datetime.now(timezone.utc)
    lease = SourceMonitoringLease(
        source_id="src_idem_1",
        lease_owner="worker_1",
        lease_acquired_at=now,
        lease_expires_at=now + timedelta(seconds=300),
        heartbeat_at=now,
        attempt_count=2,
    )

    mock_source_repo = AsyncMock(spec=BaseSourceRepository)
    mock_source_repo.get_by_id.return_value = source
    mock_lease_repo = AsyncMock(spec=BaseMonitoringLeaseRepository)
    mock_lease_repo.get_active_lease.return_value = lease
    mock_lease_repo.release_source_lease.return_value = True

    mock_acq = AsyncMock(spec=AcquisitionOrchestrator)
    mock_acq.acquire.return_value = AcquisitionResult(
        status=AcquisitionStatus.SUCCESS,
        content="Same content observation",
        status_code=200,
        acquired_at=now,
    )

    captured_inputs: list[WorkflowInputs] = []
    mock_workflow_service = AsyncMock(spec=WorkflowExecutionService)

    async def _capture_submit(
        workspace_id: str,
        inputs: WorkflowInputs,
        **kwargs: Any,
    ) -> WorkflowRun:
        captured_inputs.append(inputs)
        return WorkflowRun(
            run_id="run_1",
            workspace_id=workspace_id,
            status=WorkflowRunStatus.PENDING,
        )

    mock_workflow_service.submit_workflow_run.side_effect = _capture_submit

    service = SourceMonitoringService(
        source_repo=mock_source_repo,
        lease_repo=mock_lease_repo,
        workflow_service=mock_workflow_service,
        acquisition_orchestrator=mock_acq,
    )

    await service.monitor_source(source_id="src_idem_1", lease_owner="worker_1")

    assert len(captured_inputs) == 1
    idempotency_key = captured_inputs[0].metadata.get("idempotency_key")
    assert idempotency_key is not None
    assert idempotency_key.startswith("sched_mon:src_idem_1:")
    assert ":2:" in idempotency_key


@pytest.mark.anyio
async def test_workflow_execution_service_idempotency_sequential() -> None:
    state_store = InMemoryWorkflowStateStore()
    mock_orchestrator = AsyncMock(spec=BaseWorkflowOrchestrator)
    dispatched_count = 0

    def _dispatcher(coro: Any) -> Any:
        nonlocal dispatched_count
        dispatched_count += 1
        return asyncio.create_task(coro)

    service = WorkflowExecutionService(
        state_store=state_store,
        orchestrator=mock_orchestrator,
        task_dispatcher=_dispatcher,
    )

    inputs = WorkflowInputs(
        workspace_id="ws_1",
        company_id="comp_1",
        raw_document_content="Hello world",
        metadata={"idempotency_key": "sched_mon:src_1:12345:1:abcd"},
    )

    # First submission
    run1 = await service.submit_workflow_run(
        workspace_id="ws_1",
        inputs=inputs,
        trigger_type=TriggerType.SCHEDULED_MONITORING,
    )
    assert run1.status == WorkflowRunStatus.PENDING
    assert dispatched_count == 1

    # Duplicate submission with identical idempotency_key
    run2 = await service.submit_workflow_run(
        workspace_id="ws_1",
        inputs=inputs,
        trigger_type=TriggerType.SCHEDULED_MONITORING,
    )

    # Must return exact same WorkflowRun and NOT dispatch background task again
    assert run2.run_id == run1.run_id
    assert dispatched_count == 1


@pytest.mark.anyio
async def test_workflow_execution_service_distinct_idempotency_keys() -> None:
    state_store = InMemoryWorkflowStateStore()
    mock_orchestrator = AsyncMock(spec=BaseWorkflowOrchestrator)
    dispatched_count = 0

    def _dispatcher(coro: Any) -> Any:
        nonlocal dispatched_count
        dispatched_count += 1
        return asyncio.create_task(coro)

    service = WorkflowExecutionService(
        state_store=state_store,
        orchestrator=mock_orchestrator,
        task_dispatcher=_dispatcher,
    )

    inputs1 = WorkflowInputs(
        workspace_id="ws_1",
        company_id="comp_1",
        raw_document_content="Doc 1",
        metadata={"idempotency_key": "sched_mon:src_1:100:1:hash1"},
    )
    inputs2 = WorkflowInputs(
        workspace_id="ws_1",
        company_id="comp_1",
        raw_document_content="Doc 2",
        metadata={"idempotency_key": "sched_mon:src_1:200:1:hash2"},
    )

    run1 = await service.submit_workflow_run(
        workspace_id="ws_1",
        inputs=inputs1,
    )
    run2 = await service.submit_workflow_run(
        workspace_id="ws_1",
        inputs=inputs2,
    )

    assert run1.run_id != run2.run_id
    assert dispatched_count == 2


@pytest.mark.anyio
async def test_workflow_execution_service_manual_without_idempotency() -> None:
    state_store = InMemoryWorkflowStateStore()
    mock_orchestrator = AsyncMock(spec=BaseWorkflowOrchestrator)
    dispatched_count = 0

    def _dispatcher(coro: Any) -> Any:
        nonlocal dispatched_count
        dispatched_count += 1
        return asyncio.create_task(coro)

    service = WorkflowExecutionService(
        state_store=state_store,
        orchestrator=mock_orchestrator,
        task_dispatcher=_dispatcher,
    )

    inputs = WorkflowInputs(
        workspace_id="ws_1",
        company_id="comp_1",
        raw_document_content="Manual doc",
        metadata={},  # No idempotency key
    )

    run1 = await service.submit_workflow_run(
        workspace_id="ws_1",
        inputs=inputs,
        trigger_type=TriggerType.MANUAL_USER_REQUEST,
    )
    run2 = await service.submit_workflow_run(
        workspace_id="ws_1",
        inputs=inputs,
        trigger_type=TriggerType.MANUAL_USER_REQUEST,
    )

    # Standard manual submissions without idempotency key create separate runs
    assert run1.run_id != run2.run_id
    assert dispatched_count == 2


def test_worker_cli_clean_process_import() -> None:
    """Verify worker_cli imports cleanly in a fresh process without circular dependencies."""
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "-c", "import competitive_intelligence.monitoring.worker_cli"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"worker_cli import failed with error:\n{res.stderr}"

