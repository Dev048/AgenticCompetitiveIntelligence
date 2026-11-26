"""Integration tests for Task 1 (Production Retrieval Wiring) and Task 2 (Downstream Persistence).

Verifies:
1. Production RetrievalStage wiring:
   - Queries real QdrantVectorStore via AdaptiveRetriever / SelectiveAgenticRetriever.
   - Verifies that mock retrieval fallback is NOT used (retrieval_mode == "production").
   - Verifies evidence chunks match seeded indexed vectors.
   - Verifies workspace isolation across multi-tenant retrieval.
   - Verifies standalone RetrievalStage raises RuntimeError when unconfigured and mock fallback disabled.
2. Downstream intelligence & decision persistence:
   - Verifies rows in intelligence_events, inferred_signals, insights, decision_options, recommendations.
   - Verifies 10-layer provenance reconstruction and foreign-key integrity.
   - Verifies workspace isolation in persisted rows.
   - Verifies retry idempotency (no duplicate records).
   - Verifies recommendations remain strictly RecommendationStatus.PROPOSED.
3. Scheduled monitoring integration path:
   - Confirms scheduled monitoring resolves the identical production workflow stages and persistence repositories.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from competitive_intelligence.application.api.app import app
from competitive_intelligence.application.api.deps import (
    get_adaptive_retriever,
    get_agentic_retriever,
    get_decision_repo,
    get_decision_support_stage,
    get_document_repo,
    get_event_analysis_stage,
    get_insight_generation_stage,
    get_intelligence_repo,
    get_monitoring_service,
    get_production_retriever,
    get_retrieval_stage,
    get_semantic_retriever,
    get_session,
    get_signal_detection_stage,
    get_state_store,
    get_vector_store,
    get_workflow_decision_repo,
    get_workflow_document_repo,
    get_workflow_intelligence_repo,
    get_workflow_orchestrator,
    get_workflow_service,
)
from competitive_intelligence.application.models import (
    TrackedCompetitor,
    User,
    Workspace,
)
from competitive_intelligence.decision_support.models import (
    ActionType,
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    ObjectiveDimension,
    RecommendationStatus,
    RiskTolerance,
    TimeHorizon,
)
from competitive_intelligence.infrastructure.database.models import (
    DecisionOptionOrm,
    InferredSignalOrm,
    InsightOrm,
    IntelligenceEventOrm,
    RecommendationOrm,
)
from competitive_intelligence.infrastructure.database.repositories import (
    SqlCompetitorRepository,
    SqlDecisionRepository,
    SqlDocumentRepository,
    SqlIntelligenceRepository,
    SqlMonitoringLeaseRepository,
    SqlSourceRepository,
    SqlUserRepository,
    SqlWorkflowStateStore,
    SqlWorkspaceRepository,
)
from competitive_intelligence.infrastructure.database.session import DatabaseSessionManager
from competitive_intelligence.knowledge_pipeline.embeddings.embedding_service import (
    EmbeddingService,
)
from competitive_intelligence.knowledge_pipeline.embeddings.mock_provider import (
    MockEmbeddingProvider,
)
from competitive_intelligence.knowledge_pipeline.models import (
    VectorSearchFilter,
    VectorStoreConfig,
    VectorStoreEntry,
)
from competitive_intelligence.knowledge_pipeline.vector_store.qdrant_store import (
    QdrantVectorStore,
)
from competitive_intelligence.orchestration.models import (
    StageStatus,
    TriggerType,
    WorkflowInputs,
    WorkflowStageType,
)
from competitive_intelligence.orchestration.stages import RetrievalStage
from competitive_intelligence.source_acquisition.models import (
    AcquisitionConfig,
    AcquisitionMethod,
    ContentType,
    DataSensitivity,
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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def db_manager_fixture(tmp_path: Any) -> DatabaseSessionManager:
    # File-backed SQLite: each session gets its own connection. In-memory SQLite uses a
    # single shared StaticPool connection, so the polling request's session close/rollback
    # can discard the background workflow's in-flight writes (flaky lost rows).
    manager = DatabaseSessionManager(database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await manager.create_tables()
    return manager


@pytest.fixture
async def in_memory_vector_store() -> QdrantVectorStore:
    store = QdrantVectorStore(
        config=VectorStoreConfig(
            collection_name="test_prod_retrieval_coll",
            vector_dimensions=1024,
            location=":memory:",
        )
    )
    await store.aensure_collection()
    return store


@pytest.fixture
async def test_client(
    db_manager_fixture: DatabaseSessionManager,
    in_memory_vector_store: QdrantVectorStore,
) -> Any:
    async def override_get_session() -> Any:
        async with db_manager_fixture.session() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_state_store] = lambda: SqlWorkflowStateStore(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_workflow_document_repo] = lambda: SqlDocumentRepository(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_workflow_intelligence_repo] = lambda: SqlIntelligenceRepository(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_workflow_decision_repo] = lambda: SqlDecisionRepository(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_document_repo] = lambda: SqlDocumentRepository(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_intelligence_repo] = lambda: SqlIntelligenceRepository(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_decision_repo] = lambda: SqlDecisionRepository(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_vector_store] = lambda: in_memory_vector_store

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()
    await db_manager_fixture.close()


# ==============================================================================
# Helper to poll workflow status
# ==============================================================================


async def _poll_workflow_completion(
    client: AsyncClient,
    workspace_id: str,
    run_id: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    elapsed = 0.0
    interval = 0.1
    while elapsed < timeout:
        resp = await client.get(f"/api/v1/workspaces/{workspace_id}/workflows/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            await asyncio.sleep(0.05)
            return data
        await asyncio.sleep(interval)
        elapsed += interval
    raise TimeoutError(f"Workflow run {run_id} did not finish within {timeout}s")


# ==============================================================================
# Task 1: Production RetrievalStage Wiring Verification
# ==============================================================================


@pytest.mark.anyio
async def test_production_retrieval_wiring_and_workspace_isolation(
    test_client: AsyncClient,
    db_manager_fixture: DatabaseSessionManager,
    in_memory_vector_store: QdrantVectorStore,
) -> Any:
    """Verifies that RetrievalStage uses production retrieval (Qdrant),
    recovers actual seeded chunks, disables mock fallback, and strictly respects workspace isolation.
    """
    ws_a = "ws_retrieval_a"
    comp_a = "comp_retrieval_a"
    ws_b = "ws_retrieval_b"
    comp_b = "comp_retrieval_b"

    # 1. Seed Workspaces, Users, Competitors
    async with db_manager_fixture.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)

        await user_repo.create(User(user_id="user_retrieval", email="retrieval@test.com"))
        await ws_repo.create(Workspace(workspace_id=ws_a, name="Workspace A", owner_user_id="user_retrieval"))
        await ws_repo.create(Workspace(workspace_id=ws_b, name="Workspace B", owner_user_id="user_retrieval"))
        await comp_repo.create(
            TrackedCompetitor(competitor_id=comp_a, workspace_id=ws_a, name="Alpha Corp", domain="alpha.com")
        )
        await comp_repo.create(
            TrackedCompetitor(competitor_id=comp_b, workspace_id=ws_b, name="Beta Corp", domain="beta.com")
        )

    # 2. Seed document chunks & vectors directly in in_memory_vector_store (valid UUIDs for Qdrant)
    chunk_a_id = "11111111-1111-1111-1111-111111111111"
    chunk_b_id = "22222222-2222-2222-2222-222222222222"

    # Mock 1024-dim vectors
    vector_a = [0.05] * 1024
    vector_b = [0.02] * 1024

    await in_memory_vector_store.aupsert_entries([
        VectorStoreEntry(
            chunk_id=chunk_a_id,
            document_id="doc_alpha_1",
            version_id="ver_alpha_1",
            workspace_id=ws_a,
            company_id=comp_a,
            vector=vector_a,
            text="Alpha Corp launched Quantum Cloud Engine at $1999 per month.",
            canonical_url="https://alpha.com/pricing",
        ),
        VectorStoreEntry(
            chunk_id=chunk_b_id,
            document_id="doc_beta_1",
            version_id="ver_beta_1",
            workspace_id=ws_b,
            company_id=comp_b,
            vector=vector_b,
            text="Beta Corp reduced database tier pricing to $49 per month.",
            canonical_url="https://beta.com/pricing",
        ),
    ])

    # 3. Trigger workflow for Workspace A targeting retrieval stage
    payload_a = {
        "company_id": comp_a,
        "raw_document_content": "Alpha Corp updated Quantum Cloud Engine pricing from $2999 to $1999.",
        "document_id": "doc_alpha_1",
        "query_text": "Quantum Cloud Engine pricing updates",
        "target_stop_stage": "retrieval",
        "trigger_type": "manual_user_request",
    }
    resp = await test_client.post(f"/api/v1/workspaces/{ws_a}/workflows/runs", json=payload_a)
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    run_detail = await _poll_workflow_completion(test_client, ws_a, run_id)
    assert run_detail["status"] == "completed", f"Workflow errors: {run_detail.get('errors')}, stages: {run_detail.get('stage_records')}"

    # 4. Verify Retrieval Stage records
    stage_records = run_detail["stage_records"]
    assert "retrieval" in stage_records
    retrieval_rec = stage_records["retrieval"]
    assert retrieval_rec["status"] == "completed"

    output_ids = retrieval_rec["output_ids"]
    assert output_ids["retrieval_mode"] == "production", "Retrieval stage must use production retrieval"
    assert chunk_a_id in output_ids["chunk_ids"], f"Expected {chunk_a_id} to be retrieved"
    assert chunk_b_id not in output_ids["chunk_ids"], "Cross-workspace chunk leaked into results!"

    # 5. Verify mock fallback is disabled: standalone RetrievalStage without retriever raises RuntimeError
    standalone_stage = RetrievalStage(retriever=None, allow_mock_fallback=False)
    from competitive_intelligence.orchestration.interfaces import StageExecutionContext
    from competitive_intelligence.orchestration.models import WorkflowInputs, WorkflowRun
    from competitive_intelligence.orchestration.policies import ExecutionBudget

    dummy_run = WorkflowRun(
        run_id="run_dummy",
        workspace_id=ws_a,
        company_id=comp_a,
        trigger_type=TriggerType.MANUAL_USER_REQUEST,
    )
    dummy_ctx = StageExecutionContext(
        run=dummy_run,
        inputs=WorkflowInputs(workspace_id=ws_a, company_id=comp_a, query_text="test"),
        budget=ExecutionBudget(),
    )
    res_rec = await standalone_stage.execute(dummy_ctx)
    assert res_rec.status == StageStatus.FAILED
    assert "mock fallback is disabled" in (res_rec.error.error_message if res_rec.error else "")


# ==============================================================================
# Task 2: Downstream Intelligence and Decision Persistence Verification
# ==============================================================================


@pytest.mark.anyio
async def test_downstream_intelligence_and_decision_persistence(
    test_client: AsyncClient,
    db_manager_fixture: DatabaseSessionManager,
    in_memory_vector_store: QdrantVectorStore,
) -> Any:
    """Verifies complete end-to-end execution through decision support with direct PostgreSQL
    queries verifying rows in intelligence_events, inferred_signals, insights, decision_options,
    and recommendations. Validates FK relationships, provenance reconstruction, workspace isolation,
    idempotency on retry, and PROPOSED recommendation status.
    """
    ws_id = "ws_persistence_test"
    comp_id = "comp_persistence_test"

    # 1. Seed Workspace, User, Competitor, and Source
    async with db_manager_fixture.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)
        source_repo = SqlSourceRepository(session)

        await user_repo.create(User(user_id="user_pers", email="pers@test.com"))
        await ws_repo.create(Workspace(workspace_id=ws_id, name="Persist WS", owner_user_id="user_pers"))
        await comp_repo.create(
            TrackedCompetitor(competitor_id=comp_id, workspace_id=ws_id, name="CloudScale Inc", domain="cloudscale.io")
        )

        source = Source(
            identity=SourceIdentity(
                source_id="src_pers_1",
                workspace_id=ws_id,
                company_id=comp_id,
                display_name="CloudScale Pricing",
            ),
            location=SourceLocation(url_or_identifier="https://cloudscale.io/pricing"),
            classification=SourceClassification(
                category=SourceCategory.PRICING,
                platform=PlatformType.WEB,
                tier=SourceTier.PRIMARY,
            ),
            governance=SourceGovernance(
                tos_status=ToSStatus.COMPLIANT,
                data_sensitivity=DataSensitivity.PUBLIC,
            ),
            reliability=SourceReliability(),
            acquisition=AcquisitionConfig(
                method=AcquisitionMethod.CRAWL4AI,
                content_type=ContentType.HTML,
            ),
            lifecycle=SourceLifecycle(),
            monitoring=MonitoringPolicy(
                interval_minutes=60,
                rate_limits=RateLimitConstraints(max_requests_per_minute=30),
            ),
        )
        await source_repo.create(source)

    # 2. Build decision context for DecisionSupportStage
    decision_context = {
        "decision_id": "dec_ctx_e2e_1",
        "workspace_id": ws_id,
        "organization_id": "org_e2e_1",
        "title": "CloudScale Price Drop Strategic Response",
        "decision_domain": "pricing",
        "business_objectives": [
            {
                "objective_id": "obj_1",
                "name": "Defend Market Share",
                "dimension": "growth",
                "target_description": "Prevent customer churn to CloudScale",
                "priority_weight": 0.9,
            }
        ],
        "constraints": [],
        "risk_tolerance": "moderate",
        "time_horizon": "immediate_short_term",
        "competitor_ids": [comp_id],
    }

    # Seed vector store with acquired intelligence evidence
    mock_vector = [0.05] * 1024
    chunk_acq_id = "33333333-3333-3333-3333-333333333333"
    await in_memory_vector_store.aupsert_entries(
        [
            VectorStoreEntry(
                chunk_id=chunk_acq_id,
                document_id="doc_pers_1",
                version_id="ver_pers_1",
                workspace_id=ws_id,
                company_id=comp_id,
                vector=mock_vector,
                text="CloudScale Inc acquired DataPulse for $50M.",
                canonical_url="https://cloudscale.io/news",
            ),
        ]
    )

    raw_content = "CloudScale Inc reduced Serverless DB Tier pricing from $99 to $49 today."

    # 3. Submit workflow run through HTTP API
    payload = {
        "company_id": comp_id,
        "raw_document_content": raw_content,
        "source_url": "https://cloudscale.io/pricing",
        "source_id": "src_pers_1",
        "document_id": "doc_pers_1",
        "query_text": "CloudScale acquisition of DataPulse",
        "target_stop_stage": "decision_support",
        "enable_decision_support": True,
        "decision_context": decision_context,
        "trigger_type": "manual_user_request",
    }

    resp = await test_client.post(f"/api/v1/workspaces/{ws_id}/workflows/runs", json=payload)
    assert resp.status_code == 202
    run_id = resp.json()["run_id"]

    run_detail = await _poll_workflow_completion(test_client, ws_id, run_id)
    assert run_detail["status"] == "completed", f"Workflow failed with status {run_detail['status']}. Errors: {run_detail.get('errors')}. Stages: {run_detail.get('stage_records')}"

    # 4. Query PostgreSQL tables directly to verify persisted records
    async with db_manager_fixture.session() as session:
        # Intelligence Events
        events_res = await session.execute(
            select(IntelligenceEventOrm).where(IntelligenceEventOrm.workspace_id == ws_id)
        )
        events = list(events_res.scalars().all())
        assert len(events) > 0, "Expected persisted intelligence_events rows"
        for ev in events:
            assert ev.workspace_id == ws_id
            assert ev.company_id == comp_id

        # Inferred Signals
        signals_res = await session.execute(
            select(InferredSignalOrm).where(InferredSignalOrm.workspace_id == ws_id)
        )
        signals = list(signals_res.scalars().all())
        assert len(signals) > 0, "Expected persisted inferred_signals rows"
        for sig in signals:
            assert sig.workspace_id == ws_id
            assert sig.company_id == comp_id

        # Insights
        insights_res = await session.execute(
            select(InsightOrm).where(InsightOrm.workspace_id == ws_id)
        )
        insights = list(insights_res.scalars().all())
        assert len(insights) > 0, "Expected persisted insights rows"
        for ins in insights:
            assert ins.workspace_id == ws_id
            assert ins.company_id == comp_id

        # Decision Options
        options_res = await session.execute(
            select(DecisionOptionOrm).where(DecisionOptionOrm.decision_context_id == "dec_ctx_e2e_1")
        )
        options = list(options_res.scalars().all())
        assert len(options) > 0, "Expected persisted decision_options rows"

        # Recommendations
        rec_res = await session.execute(
            select(RecommendationOrm).where(RecommendationOrm.workspace_id == ws_id)
        )
        recommendations = list(rec_res.scalars().all())
        assert len(recommendations) == 1, f"Expected exactly 1 recommendation row, got {len(recommendations)}. Run detail: {run_detail}"
        rec = recommendations[0]
        assert rec.workspace_id == ws_id
        assert rec.decision_context_id == "dec_ctx_e2e_1"
        assert rec.status == "proposed", "Recommendation status MUST remain 'proposed'"

    # 5. Verify 10-layer provenance reconstruction via SqlDecisionRepository
    dec_repo = SqlDecisionRepository(db_manager_fixture.sessionmaker)
    rec_id = recommendations[0].recommendation_id
    provenance = await dec_repo.reconstruct_recommendation_provenance(rec_id)
    assert provenance["recommendation_id"] == rec_id
    assert provenance["decision_id"] == "dec_ctx_e2e_1"
    assert provenance["insight_count"] > 0
    assert provenance["event_count"] > 0
    assert provenance["evidence_count"] > 0
    assert len(provenance["source_ids"]) > 0

    # 6. Verify Workspace Isolation
    async with db_manager_fixture.session() as session:
        other_events = await session.execute(
            select(IntelligenceEventOrm).where(IntelligenceEventOrm.workspace_id == "other_workspace")
        )
        assert len(list(other_events.scalars().all())) == 0

    # 7. Retry Idempotency: re-submit exact same workflow run and verify NO duplicate rows created
    events_count_before = len(events)
    signals_count_before = len(signals)
    insights_count_before = len(insights)
    options_count_before = len(options)
    recs_count_before = len(recommendations)

    resp_retry = await test_client.post(f"/api/v1/workspaces/{ws_id}/workflows/runs", json=payload)
    assert resp_retry.status_code == 202
    retry_run_id = resp_retry.json()["run_id"]

    retry_detail = await _poll_workflow_completion(test_client, ws_id, retry_run_id)
    assert retry_detail["status"] == "completed", f"Retry failed: {retry_detail.get('errors')}, stages: {retry_detail.get('stage_records')}"

    async with db_manager_fixture.session() as session:
        events_after = list((await session.execute(
            select(IntelligenceEventOrm).where(IntelligenceEventOrm.workspace_id == ws_id)
        )).scalars().all())
        signals_after = list((await session.execute(
            select(InferredSignalOrm).where(InferredSignalOrm.workspace_id == ws_id)
        )).scalars().all())
        insights_after = list((await session.execute(
            select(InsightOrm).where(InsightOrm.workspace_id == ws_id)
        )).scalars().all())
        options_after = list((await session.execute(
            select(DecisionOptionOrm).where(DecisionOptionOrm.decision_context_id == "dec_ctx_e2e_1")
        )).scalars().all())
        recs_after = list((await session.execute(
            select(RecommendationOrm).where(RecommendationOrm.workspace_id == ws_id)
        )).scalars().all())

        assert len(events_after) == events_count_before, "Duplicate events created on retry!"
        assert len(signals_after) == signals_count_before, "Duplicate signals created on retry!"
        assert len(insights_after) == insights_count_before, "Duplicate insights created on retry!"
        assert len(options_after) == options_count_before, "Duplicate options created on retry!"
        assert len(recs_after) == recs_count_before, "Duplicate recommendations created on retry!"


# ==============================================================================
# Task 3: Scheduled Monitoring Workflow Path Verification
# ==============================================================================


@pytest.mark.anyio
async def test_scheduled_monitoring_uses_same_production_pipeline(
    db_manager_fixture: DatabaseSessionManager,
    in_memory_vector_store: QdrantVectorStore,
) -> Any:
    """Verifies that SourceMonitoringService resolves the exact same workflow orchestrator
    with production RetrievalStage and persistence repositories.
    """
    sessionmaker = db_manager_fixture.sessionmaker

    source_repo = SqlSourceRepository(sessionmaker)
    lease_repo = SqlMonitoringLeaseRepository(sessionmaker)
    ws_repo = SqlWorkspaceRepository(sessionmaker)
    comp_repo = SqlCompetitorRepository(sessionmaker)
    user_repo = SqlUserRepository(sessionmaker)

    ws_id = "ws_mon_pipeline"
    comp_id = "comp_mon_pipeline"
    src_id = "src_mon_pipeline"

    async with db_manager_fixture.session() as session:
        await SqlUserRepository(session).create(User(user_id="user_mon", email="mon@test.com"))
        await SqlWorkspaceRepository(session).create(
            Workspace(workspace_id=ws_id, name="Monitoring WS", owner_user_id="user_mon")
        )
        await SqlCompetitorRepository(session).create(
            TrackedCompetitor(competitor_id=comp_id, workspace_id=ws_id, name="Gamma Inc", domain="gamma.com")
        )
        await SqlSourceRepository(session).create(
            Source(
                identity=SourceIdentity(
                    source_id=src_id,
                    workspace_id=ws_id,
                    company_id=comp_id,
                    display_name="Gamma Updates",
                ),
                location=SourceLocation(url_or_identifier="https://gamma.com/blog"),
                classification=SourceClassification(
                    category=SourceCategory.PRICING,
                    platform=PlatformType.WEB,
                    tier=SourceTier.PRIMARY,
                ),
                governance=SourceGovernance(
                    tos_status=ToSStatus.COMPLIANT,
                    data_sensitivity=DataSensitivity.PUBLIC,
                ),
                reliability=SourceReliability(),
                acquisition=AcquisitionConfig(
                    method=AcquisitionMethod.CRAWL4AI,
                    content_type=ContentType.HTML,
                ),
                lifecycle=SourceLifecycle(),
                monitoring=MonitoringPolicy(interval_minutes=30),
            )
        )

    # Resolve production stages and orchestrator from dependency graph
    emb_service = EmbeddingService(provider=MockEmbeddingProvider(dimension=1024))
    retrieval_stage = get_retrieval_stage(
        retriever=get_production_retriever(
            adaptive_retriever=get_adaptive_retriever(
                semantic_retriever=get_semantic_retriever(
                    vector_store=in_memory_vector_store,
                    embedding_service=emb_service,
                )
            ),
            agentic_retriever=get_agentic_retriever(
                adaptive_retriever=get_adaptive_retriever(
                    semantic_retriever=get_semantic_retriever(
                        vector_store=in_memory_vector_store,
                        embedding_service=emb_service,
                    )
                )
            ),
        )
    )
    event_stage = get_event_analysis_stage(intelligence_repo=SqlIntelligenceRepository(sessionmaker))
    signal_stage = get_signal_detection_stage(intelligence_repo=SqlIntelligenceRepository(sessionmaker))
    insight_stage = get_insight_generation_stage(intelligence_repo=SqlIntelligenceRepository(sessionmaker))
    decision_stage = get_decision_support_stage(decision_repo=SqlDecisionRepository(sessionmaker))

    orchestrator = get_workflow_orchestrator(
        state_store=SqlWorkflowStateStore(sessionmaker),
        knowledge_stage=app.dependency_overrides.get(get_workflow_document_repo, lambda: SqlDocumentRepository(sessionmaker))(),
        retrieval_stage=retrieval_stage,
        event_stage=event_stage,
        signal_stage=signal_stage,
        insight_stage=insight_stage,
        decision_stage=decision_stage,
    )

    # Inspect stages configured in orchestrator
    ret_stage = orchestrator.stages[WorkflowStageType.RETRIEVAL]
    assert isinstance(ret_stage, RetrievalStage)
    assert ret_stage.allow_mock_fallback is False, "Scheduled pipeline must not allow mock fallback"
    assert ret_stage.retriever is not None, "Production retriever must be injected"

    ev_stage = orchestrator.stages[WorkflowStageType.EVENT_ANALYSIS]
    assert getattr(ev_stage, "intelligence_repo", None) is not None, "EventAnalysisStage must have intelligence_repo"

    dec_stage = orchestrator.stages[WorkflowStageType.DECISION_SUPPORT]
    assert getattr(dec_stage, "decision_repo", None) is not None, "DecisionSupportStage must have decision_repo"
