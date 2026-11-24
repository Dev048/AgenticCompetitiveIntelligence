"""Integration tests for FastAPI HTTP API Layer (Subsystem 6).

Exercises complete REST lifecycle, workspace isolation guards, 202 Accepted asynchronous workflow
submission, 10-layer provenance retrieval, and error mapping over HTTP transport using AsyncClient.
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import pytest
from httpx import ASGITransport, AsyncClient

from competitive_intelligence.application.api.app import app
from competitive_intelligence.application.api.deps import (
    get_decision_repo,
    get_document_repo,
    get_intelligence_repo,
    get_session,
    get_state_store,
    get_vector_store,
    get_workflow_decision_repo,
    get_workflow_document_repo,
    get_workflow_intelligence_repo,
)
from competitive_intelligence.application.models import (
    TrackedCompetitor,
    User,
    Workspace,
)
from competitive_intelligence.decision_support.models import (
    ActionType,
    ConstraintType,
    DecisionConstraint,
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    DecisionOption,
    DecisionRisk,
    DecisionUrgency,
    EvaluatedAlternative,
    ObjectiveDimension,
    PredictedConsequence,
    Recommendation,
    RecommendationConfidenceAssessment,
    RecommendationStatus,
    RiskBasisType,
    RiskTolerance,
    TimeHorizon,
)
from competitive_intelligence.infrastructure.database.models import Base
from competitive_intelligence.infrastructure.database.repositories import (
    SqlCompetitorRepository,
    SqlDecisionRepository,
    SqlDocumentRepository,
    SqlIntelligenceRepository,
    SqlSourceRepository,
    SqlUserRepository,
    SqlWorkflowStateStore,
    SqlWorkspaceRepository,
)
from competitive_intelligence.infrastructure.database.session import (
    DatabaseSessionManager,
)
from competitive_intelligence.intelligence.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EvidenceReference,
    EvidenceRole,
    InferredSignal,
    Insight,
    InsightStatus,
    InsightType,
    IntelligenceEvent,
    IntelligenceEventType,
    InterpretiveConfidenceAssessment,
    SignalConfidenceAssessment,
    SignalStatus,
    SignalType,
    SignificanceAssessment,
    SignificanceLevel,
    SignificanceStatus,
    SourcePerspective,
    TemporalScope,
    TimePrecision,
)
from competitive_intelligence.intelligence.models import EventTime
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentVersion,
    VectorStoreConfig,
)
from competitive_intelligence.knowledge_pipeline.vector_store import QdrantVectorStore
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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def db_manager_fixture() -> DatabaseSessionManager:
    """Fixture providing an isolated DatabaseSessionManager for setup in tests."""
    manager = DatabaseSessionManager(database_url="sqlite+aiosqlite:///:memory:")
    await manager.create_tables()
    return manager


from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.fixture
async def test_client(
    db_manager_fixture: DatabaseSessionManager,
) -> AsyncGenerator[AsyncClient, None]:
    """Provides an AsyncClient connected to the FastAPI application with an in-memory DB override."""
    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
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

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()
    await db_manager_fixture.close()


# ==============================================================================
# 1. Health Probe Test
# ==============================================================================


@pytest.mark.anyio
async def test_health_endpoints(test_client: AsyncClient) -> None:
    """Verify health endpoints return 200 OK and valid status envelope."""
    resp = await test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "environment" in data


# ==============================================================================
# 2. Workspace & Competitor Lifecycle API Tests
# ==============================================================================


@pytest.mark.anyio
async def test_workspace_and_competitor_api_flow(test_client: AsyncClient) -> None:
    """Test full workspace creation, competitor registration, and workspace isolation over HTTP."""
    # 1. Create Workspace
    ws_payload = {"name": "SaaS Intelligence Workspace", "owner_user_id": "usr_alpha_1"}
    ws_resp = await test_client.post("/api/v1/workspaces", json=ws_payload)
    assert ws_resp.status_code == 201
    ws_data = ws_resp.json()
    ws_id = ws_data["workspace_id"]
    assert ws_data["name"] == "SaaS Intelligence Workspace"

    # 2. Get Workspace
    get_ws = await test_client.get(f"/api/v1/workspaces/{ws_id}")
    assert get_ws.status_code == 200
    assert get_ws.json()["workspace_id"] == ws_id

    # 3. List Workspaces for User
    list_ws = await test_client.get("/api/v1/workspaces?user_id=usr_alpha_1")
    assert list_ws.status_code == 200
    assert len(list_ws.json()) == 1

    # 4. Register Competitor
    comp_payload = {
        "name": "Acme SaaS",
        "domain": "acmesaas.com",
        "description": "Primary enterprise rival",
    }
    comp_resp = await test_client.post(f"/api/v1/workspaces/{ws_id}/competitors", json=comp_payload)
    assert comp_resp.status_code == 201
    comp_data = comp_resp.json()
    comp_id = comp_data["competitor_id"]
    assert comp_data["domain"] == "acmesaas.com"

    # 5. List Competitors
    list_comp = await test_client.get(f"/api/v1/workspaces/{ws_id}/competitors")
    assert list_comp.status_code == 200
    assert len(list_comp.json()) == 1

    # 6. Update Competitor
    patch_comp = await test_client.patch(
        f"/api/v1/workspaces/{ws_id}/competitors/{comp_id}",
        json={"name": "Acme SaaS Corp", "is_active": True},
    )
    assert patch_comp.status_code == 200
    assert patch_comp.json()["name"] == "Acme SaaS Corp"

    # 7. Workspace Isolation Violation Check
    iso_resp = await test_client.get(f"/api/v1/workspaces/other_workspace_999/competitors/{comp_id}")
    assert iso_resp.status_code == 404


# ==============================================================================
# 3. Monitored Sources API Tests
# ==============================================================================


@pytest.mark.anyio
async def test_sources_api_flow(test_client: AsyncClient) -> None:
    """Test source registration, listing, and monitoring policy updates over HTTP."""
    # Create workspace & competitor
    ws_resp = await test_client.post(
        "/api/v1/workspaces", json={"name": "Source Test WS", "owner_user_id": "u_src"}
    )
    ws_id = ws_resp.json()["workspace_id"]
    comp_resp = await test_client.post(
        f"/api/v1/workspaces/{ws_id}/competitors",
        json={"name": "Comp Source", "domain": "compsource.com"},
    )
    comp_id = comp_resp.json()["competitor_id"]

    # 1. Register Source
    src_payload = {
        "company_id": comp_id,
        "display_name": "Comp Source Pricing Page",
        "url_or_identifier": "https://compsource.com/pricing",
        "category": "pricing",
        "frequency_seconds": 3600,
        "priority": 2,
    }
    src_resp = await test_client.post(f"/api/v1/workspaces/{ws_id}/sources", json=src_payload)
    assert src_resp.status_code == 201
    src_data = src_resp.json()
    src_id = src_data["source_id"]
    assert src_data["category"] == "pricing"

    # 2. Get Source
    get_src = await test_client.get(f"/api/v1/workspaces/{ws_id}/sources/{src_id}")
    assert get_src.status_code == 200
    assert get_src.json()["source_id"] == src_id

    # 3. Patch Source Monitoring
    patch_src = await test_client.patch(
        f"/api/v1/workspaces/{ws_id}/sources/{src_id}",
        json={"monitoring_enabled": False, "priority": 5},
    )
    assert patch_src.status_code == 200
    assert patch_src.json()["monitoring_enabled"] is False
    assert patch_src.json()["priority"] == 5


# ==============================================================================
# 4. Document Lineage API Tests
# ==============================================================================


@pytest.mark.anyio
async def test_documents_and_version_api_flow(
    test_client: AsyncClient, db_manager_fixture: DatabaseSessionManager
) -> None:
    """Test GET documents, document versions, and version chunks."""
    now = datetime.now(timezone.utc)
    ws_resp = await test_client.post(
        "/api/v1/workspaces", json={"name": "Doc WS", "owner_user_id": "u_doc"}
    )
    ws_id = ws_resp.json()["workspace_id"]
    comp_resp = await test_client.post(
        f"/api/v1/workspaces/{ws_id}/competitors",
        json={"name": "Doc Target", "domain": "doctarget.com"},
    )
    comp_id = comp_resp.json()["competitor_id"]

    # Seed Document, Version, and Chunk directly via repository
    async with db_manager_fixture.session() as session:
        src = Source(
            identity=SourceIdentity(source_id="src_doc_1", company_id=comp_id, workspace_id=ws_id, display_name="Doc Page"),
            classification=SourceClassification(category=SourceCategory.PRICING),
            location=SourceLocation(url_or_identifier="https://doctarget.com/pricing"),
            acquisition=AcquisitionConfig(),
        )
        await SqlSourceRepository(session).create(src)

        doc = Document(
            document_id="doc_test_1",
            source_id="src_doc_1",
            company_id=comp_id,
            workspace_id=ws_id,
            canonical_url="https://doctarget.com/pricing",
        )
        await SqlDocumentRepository(session).create_document(doc)

        ver = DocumentVersion(
            version_id="ver_test_1",
            document_id="doc_test_1",
            content="Pricing details: Pro $99/mo",
            content_hash="h111",
            acquired_at=now,
        )
        await SqlDocumentRepository(session).create_version(ver)

        chunk = DocumentChunk(
            chunk_id="chk_test_1",
            document_id="doc_test_1",
            version_id="ver_test_1",
            content="Pro $99/mo",
            chunk_index=0,
            chunk_level=0,
            metadata=ChunkMetadata(
                source_id="src_doc_1",
                document_id="doc_test_1",
                version_id="ver_test_1",
                company_id=comp_id,
                workspace_id=ws_id,
                source_url="https://doctarget.com/pricing",
                source_category="pricing",
                collected_at=now,
            ),
        )
        await SqlDocumentRepository(session).create_chunk(chunk)

    # 1. GET /documents
    docs_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/documents")
    assert docs_resp.status_code == 200
    docs = docs_resp.json()
    assert len(docs) == 1
    assert docs[0]["document_id"] == "doc_test_1"

    # 2. GET /documents/{doc_id}
    doc_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/documents/doc_test_1")
    assert doc_resp.status_code == 200
    assert doc_resp.json()["canonical_url"] == "https://doctarget.com/pricing"

    # 3. GET /documents/{doc_id}/versions
    vers_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/documents/doc_test_1/versions")
    assert vers_resp.status_code == 200
    assert len(vers_resp.json()) == 1

    # 4. GET /documents/{doc_id}/versions/{ver_id}
    ver_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/documents/doc_test_1/versions/ver_test_1")
    assert ver_resp.status_code == 200
    assert ver_resp.json()["content_hash"] == "h111"

    # 5. GET /documents/{doc_id}/versions/{ver_id}/chunks
    chunks_resp = await test_client.get(
        f"/api/v1/workspaces/{ws_id}/documents/doc_test_1/versions/ver_test_1/chunks"
    )
    assert chunks_resp.status_code == 200
    assert len(chunks_resp.json()) == 1
    assert chunks_resp.json()[0]["chunk_id"] == "chk_test_1"


# ==============================================================================
# 5. Grounded Intelligence API Tests (Events, Signals, Insights)
# ==============================================================================


@pytest.mark.anyio
async def test_intelligence_events_signals_insights_api_flow(
    test_client: AsyncClient, db_manager_fixture: DatabaseSessionManager
) -> None:
    """Test GET events, signals, and insights endpoints."""
    now = datetime.now(timezone.utc)
    ws_resp = await test_client.post(
        "/api/v1/workspaces", json={"name": "Intel WS", "owner_user_id": "u_intel"}
    )
    ws_id = ws_resp.json()["workspace_id"]
    comp_resp = await test_client.post(
        f"/api/v1/workspaces/{ws_id}/competitors",
        json={"name": "Intel Target", "domain": "inteltarget.com"},
    )
    comp_id = comp_resp.json()["competitor_id"]

    # Seed Intelligence records directly
    async with db_manager_fixture.session() as session:
        src = Source(
            identity=SourceIdentity(source_id="src_i1", company_id=comp_id, workspace_id=ws_id, display_name="Intel Page"),
            classification=SourceClassification(category=SourceCategory.PRICING),
            location=SourceLocation(url_or_identifier="https://inteltarget.com/pricing"),
            acquisition=AcquisitionConfig(),
        )
        await SqlSourceRepository(session).create(src)
        doc = Document(document_id="doc_i1", source_id="src_i1", company_id=comp_id, workspace_id=ws_id, canonical_url="https://inteltarget.com/pricing")
        await SqlDocumentRepository(session).create_document(doc)
        ver = DocumentVersion(version_id="ver_i1", document_id="doc_i1", content="Plan: $49", content_hash="hi1", acquired_at=now)
        await SqlDocumentRepository(session).create_version(ver)

        evidence = EvidenceReference(
            evidence_id="ev_i1",
            source_id="src_i1",
            document_id="doc_i1",
            version_id="ver_i1",
            source_url="https://inteltarget.com/pricing",
            source_category="pricing",
            extracted_text="Plan cut to $49",
            acquired_at=now,
        )
        event = IntelligenceEvent(
            event_id="evt_i1",
            workspace_id=ws_id,
            company_id=comp_id,
            event_type=IntelligenceEventType.PRICING_CHANGE,
            title="Price Reduced",
            summary="Price reduced to $49",
            subject="Pricing",
            event_time=EventTime(acquired_at=now, detected_at=now),
            confidence=ConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
            evidence=[evidence],
        )
        await SqlIntelligenceRepository(session).create_event(event)

        signal = InferredSignal(
            signal_id="sig_i1",
            workspace_id=ws_id,
            company_id=comp_id,
            signal_type=SignalType.AGGRESSIVE_DISCOUNTING,
            subject="Pricing",
            title="Aggressive Price War",
            summary="Cutting prices rapidly",
            supporting_event_ids=["evt_i1"],
        )
        await SqlIntelligenceRepository(session).create_signal(signal)

        insight = Insight(
            insight_id="ins_i1",
            workspace_id=ws_id,
            company_id=comp_id,
            insight_type=InsightType.PRICING_STRATEGY,
            title="Direct Volume Push",
            summary="Competitor aims for direct volume push",
            statement="Volume play underway",
            factual_confidence=ConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
            supporting_signal_ids=["sig_i1"],
            supporting_event_ids=["evt_i1"],
        )
        await SqlIntelligenceRepository(session).create_insight(insight)

    # 1. Events
    events_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/events")
    assert events_resp.status_code == 200
    assert len(events_resp.json()) == 1

    event_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/events/evt_i1")
    assert event_resp.status_code == 200
    assert event_resp.json()["title"] == "Price Reduced"

    # 2. Signals
    signals_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/signals")
    assert signals_resp.status_code == 200
    assert len(signals_resp.json()) == 1

    signal_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/signals/sig_i1")
    assert signal_resp.status_code == 200
    assert signal_resp.json()["signal_type"] == "aggressive_discounting"

    # 3. Insights
    insights_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/insights")
    assert insights_resp.status_code == 200
    assert len(insights_resp.json()) == 1

    insight_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/insights/ins_i1")
    assert insight_resp.status_code == 200
    assert insight_resp.json()["insight_type"] == "pricing_strategy"


# ==============================================================================
# 6. Asynchronous Workflow Submission API Tests (HTTP 202 Accepted)
# ==============================================================================


@pytest.mark.anyio
async def test_workflow_submission_asynchronous_by_default(
    test_client: AsyncClient, db_manager_fixture: DatabaseSessionManager
) -> None:
    """Verify that POST /workflows/runs returns HTTP 202 Accepted, run_id, and Location header."""
    # Setup workspace and competitor
    ws_resp = await test_client.post(
        "/api/v1/workspaces", json={"name": "Workflow WS", "owner_user_id": "u_wf"}
    )
    ws_id = ws_resp.json()["workspace_id"]
    comp_resp = await test_client.post(
        f"/api/v1/workspaces/{ws_id}/competitors",
        json={"name": "Workflow Target", "domain": "wftarget.com"},
    )
    comp_id = comp_resp.json()["competitor_id"]

    # 1. Submit Workflow Run
    wf_payload = {
        "company_id": comp_id,
        "target_stop_stage": "insight_generation",
        "query_text": "Analyze pricing discounts",
        "metadata": {"requester": "unit_test"},
    }
    wf_resp = await test_client.post(f"/api/v1/workspaces/{ws_id}/workflows/runs", json=wf_payload)

    assert wf_resp.status_code == 202, f"Expected 202 Accepted, got {wf_resp.status_code}: {wf_resp.text}"
    wf_data = wf_resp.json()
    run_id = wf_data["run_id"]
    assert wf_data["status"] == "pending"
    assert "Location" in wf_resp.headers
    assert wf_resp.headers["Location"] == f"/api/v1/workspaces/{ws_id}/workflows/runs/{run_id}"

    # 2. Query Workflow Run Details
    get_run = await test_client.get(f"/api/v1/workspaces/{ws_id}/workflows/runs/{run_id}")
    assert get_run.status_code == 200
    run_data = get_run.json()
    assert run_data["run_id"] == run_id
    assert run_data["company_id"] == comp_id
    assert "stage_records" in run_data

    # 3. Cancel Workflow Run
    cancel_resp = await test_client.post(f"/api/v1/workspaces/{ws_id}/workflows/runs/{run_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["cancelled"] is True

    # 4. Set status to FAILED in DB to verify resume endpoint
    from competitive_intelligence.orchestration.models import WorkflowRunStatus
    state_store = SqlWorkflowStateStore(db_manager_fixture)
    loaded_run = await state_store.get_run(run_id)
    if loaded_run:
        loaded_run.status = WorkflowRunStatus.FAILED
        await state_store.update_run(loaded_run)

    # 5. Resume Workflow Run
    resume_resp = await test_client.post(f"/api/v1/workspaces/{ws_id}/workflows/runs/{run_id}/resume")
    assert resume_resp.status_code == 202
    assert resume_resp.json()["run_id"] == run_id


# ==============================================================================
# 7. Full 10-Layer Recommendation Provenance API Test
# ==============================================================================


@pytest.mark.anyio
async def test_recommendation_provenance_api_reconstruction(
    test_client: AsyncClient, db_manager_fixture: DatabaseSessionManager
) -> None:
    """Test GET /recommendations/{id}/provenance reconstructs the 10-layer grounded intelligence tree."""
    now = datetime.now(timezone.utc)
    ws_resp = await test_client.post(
        "/api/v1/workspaces", json={"name": "Prov WS", "owner_user_id": "u_prov"}
    )
    ws_id = ws_resp.json()["workspace_id"]
    comp_resp = await test_client.post(
        f"/api/v1/workspaces/{ws_id}/competitors",
        json={"name": "Prov Target", "domain": "provtarget.com"},
    )
    comp_id = comp_resp.json()["competitor_id"]

    # 1. Create Decision Context over API
    ctx_payload = {
        "title": "Q4 Pricing Strategy Context",
        "decision_domain": "pricing",
        "competitor_ids": [comp_id],
        "business_objectives": [
            {
                "name": "Market Share Protection",
                "dimension": "growth",
                "target_description": "Protect market share against discounting",
                "priority_weight": 0.8,
            }
        ],
        "constraints": [
            {
                "constraint_type": "budget",
                "description": "Max budget $50,000",
                "is_hard_constraint": True,
            }
        ],
        "risk_tolerance": "moderate",
        "time_horizon": "medium_term",
    }
    ctx_resp = await test_client.post(f"/api/v1/workspaces/{ws_id}/decisions/contexts", json=ctx_payload)
    assert ctx_resp.status_code == 201
    ctx_id = ctx_resp.json()["decision_id"]

    # 2. Seed 10-layer lineage in DB
    async with db_manager_fixture.session() as session:
        src = Source(
            identity=SourceIdentity(source_id="src_p1", company_id=comp_id, workspace_id=ws_id, display_name="Prov Page"),
            classification=SourceClassification(category=SourceCategory.PRICING),
            location=SourceLocation(url_or_identifier="https://provtarget.com/pricing"),
            acquisition=AcquisitionConfig(),
        )
        await SqlSourceRepository(session).create(src)
        doc = Document(document_id="doc_p1", source_id="src_p1", company_id=comp_id, workspace_id=ws_id, canonical_url="https://provtarget.com/pricing")
        await SqlDocumentRepository(session).create_document(doc)
        ver = DocumentVersion(version_id="ver_p1", document_id="doc_p1", content="Enterprise Tier $499", content_hash="hp1", acquired_at=now)
        await SqlDocumentRepository(session).create_version(ver)
        chunk = DocumentChunk(
            chunk_id="chk_p1",
            document_id="doc_p1",
            version_id="ver_p1",
            content="Enterprise Tier $499",
            metadata=ChunkMetadata(source_id="src_p1", document_id="doc_p1", version_id="ver_p1", company_id=comp_id, workspace_id=ws_id, source_url="https://provtarget.com/pricing", source_category="pricing", collected_at=now),
        )
        await SqlDocumentRepository(session).create_chunk(chunk)
        evidence = EvidenceReference(evidence_id="ev_p1", source_id="src_p1", document_id="doc_p1", version_id="ver_p1", chunk_id="chk_p1", source_url="https://provtarget.com/pricing", source_category="pricing", extracted_text="Enterprise Tier $499", acquired_at=now)
        event = IntelligenceEvent(
            event_id="evt_p1",
            workspace_id=ws_id,
            company_id=comp_id,
            event_type=IntelligenceEventType.PRICING_CHANGE,
            title="Price Cut",
            summary="Price dropped",
            subject="Pricing",
            event_time=EventTime(acquired_at=now, detected_at=now),
            confidence=ConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
            evidence=[evidence],
        )
        await SqlIntelligenceRepository(session).create_event(event)
        signal = InferredSignal(signal_id="sig_p1", workspace_id=ws_id, company_id=comp_id, signal_type=SignalType.AGGRESSIVE_DISCOUNTING, subject="Pricing Strategy", title="Price Aggression", summary="Aggressive pricing observed", supporting_event_ids=[event.event_id])
        await SqlIntelligenceRepository(session).create_signal(signal)
        insight = Insight(
            insight_id="ins_p1",
            workspace_id=ws_id,
            company_id=comp_id,
            insight_type=InsightType.PRICING_STRATEGY,
            title="Discount Push",
            summary="Competitor discount push",
            statement="Competitor discounting",
            factual_confidence=ConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
            supporting_signal_ids=[signal.signal_id],
            supporting_event_ids=[event.event_id],
        )
        await SqlIntelligenceRepository(session).create_insight(insight)

        option = DecisionOption(
            option_id="opt_p1",
            decision_id=ctx_id,
            title="Value-Add Security Bundle",
            description="Bundle premium security features",
            action_type=ActionType.ASYMMETRIC_RESPONSE,
            supporting_insight_ids=[insight.insight_id],
            supporting_event_ids=[event.event_id],
        )
        await SqlDecisionRepository(session).create_option(option)

        rec = Recommendation(
            recommendation_id="rec_p1",
            decision_id=ctx_id,
            workspace_id=ws_id,
            selected_option_id="opt_p1",
            urgency=DecisionUrgency.RECOMMENDED,
            status=RecommendationStatus.PROPOSED,
            title="Introduce Value-Add Security Bundle",
            summary="Protect margin by bundling security features.",
            rationale="Maintains price discipline while neutralizing competitor.",
            evaluated_alternatives=[EvaluatedAlternative(option_id="opt_p1", title="Security Bundle", is_selected=True)],
            expected_benefits=["Protect ARR"],
            confidence=RecommendationConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH, rationale="Solid evidence"),
            supporting_insight_ids=[insight.insight_id],
            supporting_event_ids=[event.event_id],
        )
        await SqlDecisionRepository(session).create_recommendation(rec)

    # 3. Query candidate options for context over API
    options_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/decisions/contexts/{ctx_id}/options")
    assert options_resp.status_code == 200
    assert len(options_resp.json()) == 1

    # 4. Query recommendations over API
    recs_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/recommendations?context_id={ctx_id}")
    assert recs_resp.status_code == 200
    assert len(recs_resp.json()) == 1

    rec_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/recommendations/rec_p1")
    assert rec_resp.status_code == 200
    assert rec_resp.json()["recommendation_id"] == "rec_p1"

    # 5. Query 10-layer provenance endpoint
    prov_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/recommendations/rec_p1/provenance")
    assert prov_resp.status_code == 200
    prov_data = prov_resp.json()
    assert prov_data["recommendation_id"] == "rec_p1"
    assert prov_data["decision_id"] == ctx_id
    assert prov_data["insight_count"] >= 1
    assert prov_data["signal_count"] >= 1
    assert prov_data["event_count"] >= 1
    assert prov_data["evidence_count"] >= 1
    assert prov_data["chunk_count"] >= 1
    assert prov_data["version_count"] >= 1
    assert prov_data["document_count"] >= 1
    assert prov_data["source_count"] >= 1


# ==============================================================================
# 8. Error Handling & Validation Tests
# ==============================================================================


@pytest.mark.anyio
async def test_error_handling_and_validation(test_client: AsyncClient) -> None:
    """Test RFC 7807 problem error envelopes for not-found and validation failures."""
    # 1. Not Found Error (404)
    resp_404 = await test_client.get("/api/v1/workspaces/non_existent_workspace_123")
    assert resp_404.status_code == 404
    err_data = resp_404.json()
    assert err_data["error_code"] == "NOT_FOUND"

    # 2. Validation Error (422) on empty workspace name
    resp_422 = await test_client.post(
        "/api/v1/workspaces", json={"name": "", "owner_user_id": "u1"}
    )
    assert resp_422.status_code == 422
    err_422 = resp_422.json()
    assert "error_code" in err_422


# ==============================================================================
# 9. Operational Workflow API & Configured Knowledge Pipeline Verification
# ==============================================================================


@pytest.mark.anyio
async def test_api_workflow_uses_configured_knowledge_pipeline(
    test_client: AsyncClient,
    db_manager_fixture: DatabaseSessionManager,
) -> None:
    """Verify that POST /workspaces/{id}/workflows/runs executes the configured KnowledgeProcessingStage
    and KnowledgeIngestionService with production dependencies (PostgreSQL/SqlDocumentRepository
    and QdrantVectorStore) rather than default in-memory stages.
    """
    ws_id = "ws_wf_prod_wiring"
    comp_id = "comp_wf_prod_wiring"

    # 1. Seed Workspace, User, Competitor
    async with db_manager_fixture.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)

        await user_repo.create(User(user_id="u_wf_test", email="wf@test.com"))
        await ws_repo.create(Workspace(workspace_id=ws_id, name="Workflow WS", owner_user_id="u_wf_test"))
        await comp_repo.create(
            TrackedCompetitor(
                competitor_id=comp_id,
                workspace_id=ws_id,
                name="Acme Vector Corp",
                domain="acme.example.com",
            )
        )

        source_repo = SqlSourceRepository(session)
        source = Source(
            identity=SourceIdentity(
                source_id="src_api_wf_acme",
                workspace_id=ws_id,
                company_id=comp_id,
                display_name="Acme Pricing",
            ),
            location=SourceLocation(url_or_identifier="https://acme.example.com/pricing"),
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

    # 2. Provide a dedicated in-memory QdrantVectorStore to inspect vector indexing
    test_vector_store = QdrantVectorStore(
        config=VectorStoreConfig(
            collection_name="test_api_wf_collection",
            vector_dimensions=1024,
            location=":memory:",
        )
    )
    app.dependency_overrides[get_vector_store] = lambda: test_vector_store

    try:
        # 3. Submit workflow run through HTTP API
        payload = {
            "company_id": comp_id,
            "source_id": "src_api_wf_acme",
            "target_stop_stage": "knowledge_processing",
            "trigger_type": "manual_user_request",
            "raw_document_content": "Acme Vector Corp announced Enterprise Plan at $999/month with dedicated support.",
            "source_url": "https://acme.example.com/pricing",
            "document_id": "doc_api_wf_acme",
        }
        resp = await test_client.post(f"/api/v1/workspaces/{ws_id}/workflows/runs", json=payload)
        assert resp.status_code == 202
        run_summary = resp.json()
        run_id = run_summary["run_id"]
        assert run_summary["workspace_id"] == ws_id
        assert run_summary["company_id"] == comp_id

        # 4. Await asynchronous workflow execution completion
        max_wait_seconds = 30.0
        poll_interval = 0.1
        elapsed = 0.0
        run_detail = None
        while elapsed < max_wait_seconds:
            status_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/workflows/runs/{run_id}")
            assert status_resp.status_code == 200
            run_detail = status_resp.json()
            if run_detail["status"] in ("completed", "failed"):
                await asyncio.sleep(0.05)
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        assert run_detail is not None
        assert run_detail["status"] == "completed", f"Run failed with errors: {run_detail.get('errors')}"

        # 5. Verify stage execution record shows knowledge_processing completed successfully
        assert "knowledge_processing" in run_detail["stage_records"]
        kp_rec = run_detail["stage_records"]["knowledge_processing"]
        assert kp_rec["status"] == "completed"

        outputs = run_detail["produced_outputs"]
        assert len(outputs["document_ids"]) == 1
        doc_id = outputs["document_ids"][0]
        assert outputs["chunks_created_count"] > 0
        assert outputs["vector_points_upserted_count"] > 0

        # 6. Verify canonical state persisted into Database via SqlDocumentRepository
        doc_repo = SqlDocumentRepository(db_manager_fixture.sessionmaker)
        persisted_doc = await doc_repo.get_document(doc_id)
        assert persisted_doc is not None
        assert persisted_doc.workspace_id == ws_id

        persisted_versions = await doc_repo.list_versions_for_document(doc_id)
        assert len(persisted_versions) == 1
        version_id = persisted_versions[0].version_id

        persisted_chunks = await doc_repo.get_chunks_for_version(version_id)
        assert len(persisted_chunks) == outputs["chunks_created_count"]

        # 7. Verify vector points indexed in QdrantVectorStore
        for chunk in persisted_chunks:
            vector_res = await test_vector_store.aget_by_id(chunk.chunk_id)
            assert vector_res is not None
            assert vector_res.chunk_id == chunk.chunk_id
            assert vector_res.workspace_id == ws_id
            assert vector_res.company_id == comp_id
    finally:
        app.dependency_overrides.pop(get_vector_store, None)

