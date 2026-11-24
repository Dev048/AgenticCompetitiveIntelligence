"""Integration tests for Continuous Monitoring Persistence.

Tests:
1. SqlMonitoringLeaseRepository lifecycle (atomic claim, duplicate claim prevention, heartbeat renewal, release, expiration, reclaim).
2. SqlSourceRepository.get_due_sources query logic (due calculation, disabled monitoring, lifecycle filtering, active lease filtering, expired lease recovery, priority ordering).
3. SqlSourceRepository.update_fetch_status telemetry updates.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock
import pytest

from competitive_intelligence.application.models import (
    TrackedCompetitor,
    User,
    Workspace,
)
from competitive_intelligence.infrastructure.database.repositories.competitor_repo import (
    SqlCompetitorRepository,
)
from competitive_intelligence.infrastructure.database.repositories.monitoring_lease_repo import (
    SqlMonitoringLeaseRepository,
)
from competitive_intelligence.monitoring.models import SourceMonitoringLease
from competitive_intelligence.infrastructure.database.repositories.source_repo import (
    SqlSourceRepository,
)
from competitive_intelligence.infrastructure.database.repositories.workspace_repo import (
    SqlUserRepository,
    SqlWorkspaceRepository,
)
from competitive_intelligence.infrastructure.database.session import DatabaseSessionManager
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
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


from sqlalchemy.pool import StaticPool


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _get_initialized_db() -> DatabaseSessionManager:
    """Helper creating and initializing an in-memory test database."""
    manager = DatabaseSessionManager(
        database_url="sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    await manager.create_tables()
    return manager


def _build_source(
    source_id: str,
    workspace_id: str = "ws_test",
    company_id: str = "comp_test",
    monitoring_enabled: bool = True,
    frequency_seconds: int = 3600,
    priority: int = 1,
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE,
    last_checked_at: datetime | None = None,
    consecutive_failures: int = 0,
) -> Source:
    return Source(
        identity=SourceIdentity(
            source_id=source_id,
            workspace_id=workspace_id,
            company_id=company_id,
            display_name=f"Source {source_id}",
        ),
        classification=SourceClassification(
            category=SourceCategory.PRICING,
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier=f"https://example.com/source/{source_id}",
            access_pattern=AccessPattern.PUBLIC_WEB,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            consecutive_failure_count=consecutive_failures,
        ),
        monitoring=MonitoringPolicy(
            enabled=monitoring_enabled,
            frequency=frequency_seconds,
            priority=priority,
            last_checked_at=last_checked_at,
        ),
        reliability=SourceReliability(
            credibility_score=1.0,
            source_tier=SourceTier.PRIMARY,
        ),
        governance=SourceGovernance(
            terms_of_service_status=ToSStatus.COMPLIANT,
            robots_txt_respected=True,
            data_sensitivity=DataSensitivity.PUBLIC,
        ),
        lifecycle=SourceLifecycle(
            status=lifecycle_status,
        ),
    )


# ==============================================================================
# 1. SqlMonitoringLeaseRepository Tests
# ==============================================================================


@pytest.mark.anyio
async def test_lease_repository_lifecycle() -> None:
    db = await _get_initialized_db()

    async with db.session() as session:
        # Setup prerequisite parent hierarchy
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)
        src_repo = SqlSourceRepository(session)
        lease_repo = SqlMonitoringLeaseRepository(session)

        await user_repo.create(User(user_id="u1", email="u1@test.com"))
        await ws_repo.create(Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1"))
        await comp_repo.create(
            TrackedCompetitor(competitor_id="c1", workspace_id="ws1", name="C1", domain="c1.com")
        )

        source = _build_source(source_id="s1", workspace_id="ws1", company_id="c1")
        await src_repo.create(source)

        # 1. Claim initial lease
        lease = await lease_repo.claim_due_source(
            source_id="s1",
            lease_owner="worker_a",
            lease_duration_seconds=300,
        )
        assert lease is not None
        assert lease.source_id == "s1"
        assert lease.lease_owner == "worker_a"
        assert lease.attempt_count == 1
        assert not lease.is_expired

        # 2. Concurrent claim by another worker should be rejected
        conflict_lease = await lease_repo.claim_due_source(
            source_id="s1",
            lease_owner="worker_b",
            lease_duration_seconds=300,
        )
        assert conflict_lease is None

        # 3. Get active lease
        active = await lease_repo.get_active_lease("s1")
        assert active is not None
        assert active.lease_owner == "worker_a"

        # 4. Heartbeat renewal
        renewed = await lease_repo.renew_source_lease(
            source_id="s1",
            lease_owner="worker_a",
            extension_seconds=600,
        )
        assert renewed is True

        # Renewal by wrong owner fails
        wrong_renew = await lease_repo.renew_source_lease(
            source_id="s1",
            lease_owner="worker_b",
            extension_seconds=600,
        )
        assert wrong_renew is False

        # 5. Release lease
        released = await lease_repo.release_source_lease(
            source_id="s1",
            lease_owner="worker_a",
        )
        assert released is True

        # Active lease is now None
        active_after = await lease_repo.get_active_lease("s1")
        assert active_after is None


@pytest.mark.anyio
async def test_lease_expiration_and_reclaim() -> None:
    db = await _get_initialized_db()

    async with db.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)
        src_repo = SqlSourceRepository(session)
        lease_repo = SqlMonitoringLeaseRepository(session)

        await user_repo.create(User(user_id="u1", email="u1@test.com"))
        await ws_repo.create(Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1"))
        await comp_repo.create(
            TrackedCompetitor(competitor_id="c1", workspace_id="ws1", name="C1", domain="c1.com")
        )

        source = _build_source(source_id="s_exp", workspace_id="ws1", company_id="c1")
        await src_repo.create(source)

        # Claim lease with 1 second duration
        lease = await lease_repo.claim_due_source(
            source_id="s_exp",
            lease_owner="crashed_worker",
            lease_duration_seconds=1,
        )
        assert lease is not None

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Re-claim expired lease by new worker should succeed
        new_lease = await lease_repo.claim_due_source(
            source_id="s_exp",
            lease_owner="recovery_worker",
            lease_duration_seconds=300,
        )
        assert new_lease is not None
        assert new_lease.lease_owner == "recovery_worker"
        assert new_lease.attempt_count == 2


# ==============================================================================
# 2. SqlSourceRepository get_due_sources & update_fetch_status Tests
# ==============================================================================


@pytest.mark.anyio
async def test_source_repository_get_due_sources_filtering() -> None:
    db = await _get_initialized_db()

    async with db.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)
        src_repo = SqlSourceRepository(session)
        lease_repo = SqlMonitoringLeaseRepository(session)

        await user_repo.create(User(user_id="u1", email="u1@test.com"))
        await ws_repo.create(Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1"))
        await comp_repo.create(
            TrackedCompetitor(competitor_id="c1", workspace_id="ws1", name="C1", domain="c1.com")
        )

        now = datetime.now(timezone.utc)

        # 1. Source never checked -> DUE (High Priority 10)
        s_never = _build_source(
            source_id="s_never",
            workspace_id="ws1",
            company_id="c1",
            priority=10,
            last_checked_at=None,
        )
        await src_repo.create(s_never)

        # 2. Source checked 2 hours ago with 1-hour frequency -> DUE (Priority 5)
        s_due = _build_source(
            source_id="s_due",
            workspace_id="ws1",
            company_id="c1",
            frequency_seconds=3600,
            priority=5,
            last_checked_at=now - timedelta(hours=2),
        )
        await src_repo.create(s_due)

        # 3. Source checked 10 minutes ago with 1-hour frequency -> NOT DUE
        s_not_due = _build_source(
            source_id="s_not_due",
            workspace_id="ws1",
            company_id="c1",
            frequency_seconds=3600,
            priority=8,
            last_checked_at=now - timedelta(minutes=10),
        )
        await src_repo.create(s_not_due)

        # 4. Monitoring disabled -> NOT DUE
        s_disabled = _build_source(
            source_id="s_disabled",
            workspace_id="ws1",
            company_id="c1",
            monitoring_enabled=False,
            last_checked_at=None,
        )
        await src_repo.create(s_disabled)

        # 5. Lifecycle paused -> NOT DUE
        s_deprecated = _build_source(
            source_id="s_deprecated",
            workspace_id="ws1",
            company_id="c1",
            lifecycle_status=LifecycleStatus.PAUSED,
            last_checked_at=None,
        )
        await src_repo.create(s_deprecated)

        # 6. Actively leased source -> NOT DUE
        s_leased = _build_source(
            source_id="s_leased",
            workspace_id="ws1",
            company_id="c1",
            last_checked_at=None,
        )
        await src_repo.create(s_leased)
        await lease_repo.claim_due_source(
            source_id="s_leased",
            lease_owner="active_worker",
            lease_duration_seconds=300,
        )

        # Query due sources
        due_sources = await src_repo.get_due_sources(limit=10, as_of=now)
        due_ids = [s.identity.source_id for s in due_sources]

        # Verify due sources match exactly and are ordered by priority desc
        assert "s_never" in due_ids
        assert "s_due" in due_ids
        assert "s_not_due" not in due_ids
        assert "s_disabled" not in due_ids
        assert "s_deprecated" not in due_ids
        assert "s_leased" not in due_ids

        # Priority ordering: s_never (priority 10) before s_due (priority 5)
        assert due_ids[0] == "s_never"
        assert due_ids[1] == "s_due"


@pytest.mark.anyio
async def test_source_repository_update_fetch_status() -> None:
    db = await _get_initialized_db()

    async with db.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)
        src_repo = SqlSourceRepository(session)

        await user_repo.create(User(user_id="u1", email="u1@test.com"))
        await ws_repo.create(Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1"))
        await comp_repo.create(
            TrackedCompetitor(competitor_id="c1", workspace_id="ws1", name="C1", domain="c1.com")
        )

        source = _build_source(source_id="s_status", workspace_id="ws1", company_id="c1")
        await src_repo.create(source)

        check_time = datetime.now(timezone.utc)
        updated = await src_repo.update_fetch_status(
            source_id="s_status",
            last_fetch_status="failure",
            consecutive_failure_count=4,
            last_checked_at=check_time,
        )
        assert updated is True

        reloaded = await src_repo.get_by_id("s_status")
        assert reloaded is not None
        assert reloaded.acquisition.last_fetch_status == FetchStatus.FAILURE
        assert reloaded.acquisition.consecutive_failure_count == 4
        assert reloaded.monitoring.last_checked_at is not None


@pytest.mark.anyio
async def test_due_sources_expired_lease_and_workspace_isolation() -> None:
    db = await _get_initialized_db()

    async with db.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)
        src_repo = SqlSourceRepository(session)
        lease_repo = SqlMonitoringLeaseRepository(session)

        # Setup Workspace A and B
        await user_repo.create(User(user_id="u_iso", email="iso@test.com"))
        await ws_repo.create(Workspace(workspace_id="ws_a", name="WS A", owner_user_id="u_iso"))
        await ws_repo.create(Workspace(workspace_id="ws_b", name="WS B", owner_user_id="u_iso"))
        await comp_repo.create(
            TrackedCompetitor(competitor_id="c_a", workspace_id="ws_a", name="CA", domain="ca.com")
        )
        await comp_repo.create(
            TrackedCompetitor(competitor_id="c_b", workspace_id="ws_b", name="CB", domain="cb.com")
        )

        # Source in WS A
        s_a = _build_source(source_id="src_a", workspace_id="ws_a", company_id="c_a")
        await src_repo.create(s_a)

        # Source in WS B
        s_b = _build_source(source_id="src_b", workspace_id="ws_b", company_id="c_b")
        await src_repo.create(s_b)

        # Claim s_a with short lease and let it expire
        await lease_repo.claim_due_source("src_a", lease_owner="worker_old", lease_duration_seconds=1)
        await asyncio.sleep(1.1)

        # WS A filter: only returns src_a (which had expired lease)
        due_a = await src_repo.get_due_sources(workspace_id="ws_a")
        assert len(due_a) == 1
        assert due_a[0].identity.source_id == "src_a"

        # WS B filter: only returns src_b
        due_b = await src_repo.get_due_sources(workspace_id="ws_b")
        assert len(due_b) == 1
        assert due_b[0].identity.source_id == "src_b"


# ==============================================================================
# 3. Concurrent Initial Lease Claim & SQL Workflow Idempotency Tests
# ==============================================================================


from pathlib import Path


@pytest.mark.anyio
async def test_concurrent_initial_lease_claim_race(tmp_path: Path) -> None:
    """Test race condition where two workers attempt to claim a source with no existing lease row."""
    db_file = tmp_path / "test_race.db"
    db = DatabaseSessionManager(database_url=f"sqlite+aiosqlite:///{db_file}")
    await db.create_tables()

    async with db.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)
        src_repo = SqlSourceRepository(session)

        await user_repo.create(User(user_id="u_race", email="race@test.com"))
        await ws_repo.create(Workspace(workspace_id="ws_race", name="WS Race", owner_user_id="u_race"))
        await comp_repo.create(
            TrackedCompetitor(competitor_id="c_race", workspace_id="ws_race", name="C Race", domain="crace.com")
        )
        source = _build_source(source_id="src_race", workspace_id="ws_race", company_id="c_race")
        await src_repo.create(source)

    async def _worker_claim(owner: str) -> SourceMonitoringLease | None:
        async with db.session() as session:
            lease_repo = SqlMonitoringLeaseRepository(session)
            return await lease_repo.claim_due_source(
                source_id="src_race",
                lease_owner=owner,
                lease_duration_seconds=300,
            )

    gather_results = await asyncio.gather(
        _worker_claim("worker_a"),
        _worker_claim("worker_b"),
        return_exceptions=True,
    )

    result_a = gather_results[0]
    result_b = gather_results[1]

    assert not isinstance(result_a, BaseException), f"Worker A failed with {result_a}"
    assert not isinstance(result_b, BaseException), f"Worker B failed with {result_b}"

    successes: list[SourceMonitoringLease] = [r for r in (result_a, result_b) if r is not None]
    failures = [r for r in (result_a, result_b) if r is None]

    assert len(successes) == 1
    assert len(failures) == 1

    winning_lease = successes[0]
    assert winning_lease.source_id == "src_race"
    assert winning_lease.lease_owner in ("worker_a", "worker_b")

    async with db.session() as session:
        lease_repo = SqlMonitoringLeaseRepository(session)
        active = await lease_repo.get_active_lease("src_race")
        assert active is not None
        assert active.lease_owner == winning_lease.lease_owner


@pytest.mark.anyio
async def test_sql_workflow_state_store_idempotency_and_service() -> None:
    """Test SQL workflow state store idempotency key lookup and duplicate suppression."""
    from unittest.mock import AsyncMock
    from competitive_intelligence.application.services.workflow_service import (
        WorkflowExecutionService,
    )
    from competitive_intelligence.infrastructure.database.repositories.workflow_state_store import (
        SqlWorkflowStateStore,
    )
    from competitive_intelligence.orchestration.interfaces import (
        BaseWorkflowOrchestrator,
    )
    from competitive_intelligence.orchestration.models import (
        TriggerType,
        WorkflowInputs,
        WorkflowRun,
        WorkflowRunStatus,
    )

    db = await _get_initialized_db()

    async with db.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)

        await user_repo.create(User(user_id="u_idem", email="idem@test.com"))
        await ws_repo.create(Workspace(workspace_id="ws_idem", name="WS Idem", owner_user_id="u_idem"))
        await comp_repo.create(
            TrackedCompetitor(competitor_id="c_idem", workspace_id="ws_idem", name="C Idem", domain="cidem.com")
        )

    state_store = SqlWorkflowStateStore(session_manager=db)
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
        workspace_id="ws_idem",
        company_id="c_idem",
        raw_document_content="Observed doc content",
        metadata={"idempotency_key": "sched_mon:src_idem:1000:1:hash123"},
    )

    # 1. First submission creates WorkflowRun
    run1 = await service.submit_workflow_run(
        workspace_id="ws_idem",
        inputs=inputs,
        trigger_type=TriggerType.SCHEDULED_MONITORING,
    )
    assert run1.status == WorkflowRunStatus.PENDING
    assert dispatched_count == 1

    # 2. Lookup by idempotency key
    found = await state_store.get_run_by_idempotency_key("ws_idem", "sched_mon:src_idem:1000:1:hash123")
    assert found is not None
    assert found.run_id == run1.run_id

    # 3. Sequential duplicate submission returns exact same run without dispatching
    run2 = await service.submit_workflow_run(
        workspace_id="ws_idem",
        inputs=inputs,
        trigger_type=TriggerType.SCHEDULED_MONITORING,
    )
    assert run2.run_id == run1.run_id
    assert dispatched_count == 1

    # 4. Concurrent duplicate submissions with same idempotency key
    inputs_conc = WorkflowInputs(
        workspace_id="ws_idem",
        company_id="c_idem",
        raw_document_content="Concurrent doc",
        metadata={"idempotency_key": "sched_mon:src_idem:2000:1:hash456"},
    )

    results = await asyncio.gather(
        service.submit_workflow_run(
            workspace_id="ws_idem",
            inputs=inputs_conc,
            trigger_type=TriggerType.SCHEDULED_MONITORING,
        ),
        service.submit_workflow_run(
            workspace_id="ws_idem",
            inputs=inputs_conc,
            trigger_type=TriggerType.SCHEDULED_MONITORING,
        ),
        return_exceptions=True,
    )

    assert isinstance(results[0], WorkflowRun), f"Submission 1 failed: {results[0]}"
    assert isinstance(results[1], WorkflowRun), f"Submission 2 failed: {results[1]}"
    assert results[0].run_id == results[1].run_id

    # 5. Different observation key creates a separate WorkflowRun
    inputs_diff = WorkflowInputs(
        workspace_id="ws_idem",
        company_id="c_idem",
        raw_document_content="Different doc",
        metadata={"idempotency_key": "sched_mon:src_idem:3000:1:hash789"},
    )
    run_diff = await service.submit_workflow_run(
        workspace_id="ws_idem",
        inputs=inputs_diff,
        trigger_type=TriggerType.SCHEDULED_MONITORING,
    )
    assert run_diff.run_id != run1.run_id
    assert run_diff.run_id != results[0].run_id

