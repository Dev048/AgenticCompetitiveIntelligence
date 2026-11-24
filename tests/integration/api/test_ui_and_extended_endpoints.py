"""Integration tests for UI Static File Serving and Extended API Endpoints.

Tests:
1. UI Static Assets: Verifies that root `/`, CSS stylesheets, and JS modules are served properly.
2. Decision Contexts Listing: Verifies GET `/workspaces/{id}/decisions/contexts` lists all contexts.
3. Human Decision Review: Verifies PATCH `/workspaces/{id}/recommendations/{id}/status` allows
   human operators to transition recommendations from PROPOSED to ACCEPTED/REJECTED.
4. Workspace-Wide Recommendations: Verifies GET `/workspaces/{id}/recommendations` works without context_id.
5. On-Demand Source Trigger: Verifies POST `/workspaces/{id}/sources/{id}/trigger` returns 202 Accepted.
6. User-Independent Workspace Listing: Verifies GET `/workspaces` lists workspaces without requiring query params.
"""

from datetime import datetime, timezone
import pytest
from httpx import ASGITransport, AsyncClient

from competitive_intelligence.application.api.app import app
from competitive_intelligence.decision_support.models import (
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    ObjectiveDimension,
    Recommendation,
    RecommendationConfidenceAssessment,
    RecommendationStatus,
    DecisionUrgency,
)
from competitive_intelligence.application.api.deps import (
    get_decision_service,
    get_workspace_service,
    get_competitor_service,
    get_source_service,
)


from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from competitive_intelligence.infrastructure.database.session import DatabaseSessionManager
from competitive_intelligence.infrastructure.database.repositories import (
    SqlDecisionRepository,
    SqlDocumentRepository,
    SqlIntelligenceRepository,
    SqlWorkflowStateStore,
)
from competitive_intelligence.application.api.deps import (
    get_session,
    get_state_store,
    get_workflow_decision_repo,
    get_workflow_document_repo,
    get_workflow_intelligence_repo,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def db_manager_fixture() -> DatabaseSessionManager:
    """Fixture providing an isolated in-memory DatabaseSessionManager."""
    manager = DatabaseSessionManager(database_url="sqlite+aiosqlite:///:memory:")
    await manager.create_tables()
    return manager


@pytest.fixture
async def test_client(
    db_manager_fixture: DatabaseSessionManager,
) -> AsyncGenerator[AsyncClient, None]:
    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with db_manager_fixture.session() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_state_store] = lambda: SqlWorkflowStateStore(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_workflow_document_repo] = lambda: SqlDocumentRepository(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_workflow_decision_repo] = lambda: SqlDecisionRepository(db_manager_fixture.sessionmaker)
    app.dependency_overrides[get_workflow_intelligence_repo] = lambda: SqlIntelligenceRepository(db_manager_fixture.sessionmaker)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()
    await db_manager_fixture.close()


@pytest.mark.anyio
async def test_ui_static_serving(test_client: AsyncClient) -> None:
    """Verify that the SPA index.html shell and static CSS/JS assets are served."""
    # 1. Root index.html
    resp = await test_client.get("/")
    assert resp.status_code == 200
    assert "VANTAGE" in resp.text
    assert "app.js" in resp.text

    # 2. CSS theme
    css_resp = await test_client.get("/css/theme.css")
    assert css_resp.status_code == 200
    assert "--bg-canvas" in css_resp.text

    # 3. JS app module
    js_resp = await test_client.get("/js/app.js")
    assert js_resp.status_code == 200
    assert "initApp" in js_resp.text


@pytest.mark.anyio
async def test_extended_workspaces_and_sources_flow(test_client: AsyncClient) -> None:
    """Test listing workspaces without user_id and triggering source on-demand workflows."""
    # 1. Create workspace
    ws_resp = await test_client.post("/api/v1/workspaces", json={"name": "UI Verification WS", "owner_user_id": "usr_ui_1"})
    assert ws_resp.status_code == 201
    ws_id = ws_resp.json()["workspace_id"]

    # 2. List workspaces without user_id query parameter
    list_ws = await test_client.get("/api/v1/workspaces")
    assert list_ws.status_code == 200
    all_ws = list_ws.json()
    assert any(w["workspace_id"] == ws_id for w in all_ws)

    # 3. Register competitor
    comp_resp = await test_client.post(
        f"/api/v1/workspaces/{ws_id}/competitors",
        json={"name": "UI Competitor", "domain": "uicomp.example.com"}
    )
    assert comp_resp.status_code == 201
    comp_id = comp_resp.json()["competitor_id"]

    # 4. Register source
    src_resp = await test_client.post(
        f"/api/v1/workspaces/{ws_id}/sources",
        json={
            "company_id": comp_id,
            "display_name": "Pricing Feed",
            "url_or_identifier": "https://uicomp.example.com/pricing",
            "category": "pricing"
        }
    )
    assert src_resp.status_code == 201
    src_id = src_resp.json()["source_id"]

    # 5. Trigger on-demand source workflow
    trigger_resp = await test_client.post(
        f"/api/v1/workspaces/{ws_id}/sources/{src_id}/trigger",
        json={"target_stop_stage": "event_analysis"}
    )
    assert trigger_resp.status_code == 202
    assert "Location" in trigger_resp.headers
    assert trigger_resp.json()["status"] in ("pending", "running", "completed")


@pytest.mark.anyio
async def test_decision_contexts_and_human_review_flow(
    test_client: AsyncClient,
    db_manager_fixture: DatabaseSessionManager,
) -> None:
    """Test listing decision contexts, workspace recommendations, and PATCH review status updates."""
    # 1. Create workspace
    ws_resp = await test_client.post("/api/v1/workspaces", json={"name": "Decisions WS", "owner_user_id": "usr_dec_1"})
    assert ws_resp.status_code == 201
    ws_id = ws_resp.json()["workspace_id"]

    # 2. Create decision context
    ctx_payload = {
        "title": "Quarterly SaaS Price Defense",
        "decision_domain": "pricing",
        "risk_tolerance": "moderate",
        "time_horizon": "medium_term",
        "business_objectives": [
          {
            "name": "Limit ARR Churn",
            "dimension": "revenue",
            "target_description": "Keep churn below 3%",
            "priority_weight": 0.9,
          }
        ],
        "constraints": [],
    }
    ctx_resp = await test_client.post(f"/api/v1/workspaces/{ws_id}/decisions/contexts", json=ctx_payload)
    assert ctx_resp.status_code == 201
    ctx_id = ctx_resp.json()["decision_id"]

    # 3. List decision contexts in workspace
    list_ctx_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/decisions/contexts")
    assert list_ctx_resp.status_code == 200
    contexts = list_ctx_resp.json()
    assert len(contexts) >= 1
    assert any(c["decision_id"] == ctx_id for c in contexts)

    # 4. Insert sample recommendation into database for testing human review
    now = datetime.now(timezone.utc)
    from competitive_intelligence.infrastructure.database.repositories.decision_repo import SqlDecisionRepository
    
    async with db_manager_fixture.session() as session:
        repo = SqlDecisionRepository(session)
        rec = Recommendation(
            decision_id=ctx_id,
            workspace_id=ws_id,
            selected_option_id="opt_sample_1",
            supporting_insight_ids=["ins_sample_1"],
            supporting_event_ids=["evt_sample_1"],
            urgency=DecisionUrgency.RECOMMENDED,
            status=RecommendationStatus.PROPOSED,
            title="Maintain Tiered Feature Superiority",
            summary="Protect margins by expanding enterprise SLA",
            rationale="Competitor discount is margin-dilutive",
            evaluated_alternatives=[],
            expected_benefits=["Protects 95% ARR"],
            tradeoffs=["Requires customer success prioritization"],
            critical_assumptions=["Competitor lacks enterprise SLA"],
            unresolved_uncertainties=[],
            confidence=RecommendationConfidenceAssessment(
                overall_confidence=0.88,
                evidence_grounding_score=0.9,
                causal_rigor_score=0.85,
                alternative_coverage_score=0.9,
                uncertainty_level="low",
                rationale="Evidence-backed pricing models confirm high confidence",
            ),
            generated_at=now,
        )
        saved_rec = await repo.create_recommendation(rec)
        await session.commit()
        rec_id = saved_rec.recommendation_id

    # 5. List recommendations without context filter
    recs_resp = await test_client.get(f"/api/v1/workspaces/{ws_id}/recommendations")
    assert recs_resp.status_code == 200
    all_recs = recs_resp.json()
    assert len(all_recs) >= 1
    assert any(r["recommendation_id"] == rec_id for r in all_recs)

    # 6. Verify initial status is strictly PROPOSED
    get_rec = await test_client.get(f"/api/v1/workspaces/{ws_id}/recommendations/{rec_id}")
    assert get_rec.status_code == 200
    assert get_rec.json()["status"] == "proposed"

    # 7. Perform Human Review: Accept recommendation
    patch_resp = await test_client.patch(
        f"/api/v1/workspaces/{ws_id}/recommendations/{rec_id}/status",
        json={"status": "accepted", "review_notes": "Approved by VP of Product Strategy"}
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "accepted"

    # 8. Perform Human Review: Reject recommendation
    patch_reject_resp = await test_client.patch(
        f"/api/v1/workspaces/{ws_id}/recommendations/{rec_id}/status",
        json={"status": "rejected", "review_notes": "Superseded by alternative motion"}
    )
    assert patch_reject_resp.status_code == 200
    assert patch_reject_resp.json()["status"] == "rejected"
