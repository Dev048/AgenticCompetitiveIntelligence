"""Comprehensive unit tests for Observability & Operational Reliability subsystem."""

import re
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from competitive_intelligence.application.api.app import create_app
from competitive_intelligence.core.config import get_settings
from competitive_intelligence.core.logger import (
    bind_log_context,
    clear_log_context,
    get_log_context,
    redact_sensitive_processor,
    redact_sensitive_value,
)
from competitive_intelligence.core.metrics import (
    APP_REGISTRY,
    HTTP_REQUESTS_TOTAL,
    LLM_BUDGET_EXHAUSTION_TOTAL,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    MONITORING_ACTIVE_LEASES,
    MONITORING_JOBS_TOTAL,
    RETRIEVAL_ESCALATED_TOTAL,
    RETRIEVAL_REQUESTS_TOTAL,
    RETRIEVAL_RERANKED_TOTAL,
    SOURCE_ACQUISITION_FAILURES_TOTAL,
    WORKFLOW_FAILURES_TOTAL,
    WORKFLOW_RUNS_TOTAL,
    WORKFLOW_STAGE_DURATION_SECONDS,
    generate_prometheus_metrics,
    get_metrics_content_type,
    normalize_endpoint,
    set_active_monitoring_leases,
    track_http_request,
    track_llm_budget_exhaustion,
    track_llm_call,
    track_monitoring_job,
    track_retrieval,
    track_stage_execution,
    track_workflow_run,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestStructuredLoggingAndRedaction:
    """Test structured logging context management and secret/PII redaction."""

    def test_log_context_binding_and_clearing(self) -> None:
        clear_log_context()
        assert get_log_context() == {}

        bind_log_context(correlation_id="corr-123", workspace_id="ws-456", stage_id="retrieval")
        ctx = get_log_context()
        assert ctx["correlation_id"] == "corr-123"
        assert ctx["workspace_id"] == "ws-456"
        assert ctx["stage_id"] == "retrieval"

        clear_log_context()
        assert get_log_context() == {}

    def test_redact_sensitive_value_strings(self) -> None:
        # OpenAI style key
        assert redact_sensitive_value("sk-abc1234567890abcdef1234567890") == "[REDACTED]"

        # Bearer token
        assert redact_sensitive_value("Bearer mysecretjwttoken12345") == "[REDACTED]"

        # Anthropic key
        assert redact_sensitive_value("sk-ant-api03-abcdef123456") == "[REDACTED]"

        # Safe string
        assert redact_sensitive_value("https://example.com/pricing") == "https://example.com/pricing"

    def test_redact_sensitive_processor_dict(self) -> None:
        event_dict = {
            "event": "LLM call initiated",
            "api_key": "supersecretkey",
            "openai_api_key": "sk-1234567890abcdef1234567890",
            "password": "mypassword",
            "access_token": "token123",
            "raw_document_content": "A" * 800,  # Should be truncated
            "company_name": "Acme Corp",
        }

        scrubbed = redact_sensitive_processor(None, "info", event_dict)

        assert scrubbed["api_key"] == "[REDACTED]"
        assert scrubbed["openai_api_key"] == "[REDACTED]"
        assert scrubbed["password"] == "[REDACTED]"
        assert scrubbed["access_token"] == "[REDACTED]"
        assert scrubbed["company_name"] == "Acme Corp"
        assert len(scrubbed["raw_document_content"]) < 800
        assert "[TRUNCATED" in scrubbed["raw_document_content"]


class TestPrometheusMetricsInstrumentation:
    """Test Prometheus counters, histograms, gauges and cardinality normalization."""

    def test_normalize_endpoint_cardinality(self) -> None:
        # UUID normalization
        url1 = "/api/v1/workspaces/a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d/sources"
        assert normalize_endpoint(url1) == "/api/v1/workspaces/{id}/sources"

        # Custom slug normalization
        url2 = "/api/v1/workspaces/ws_alpha/competitors/comp_beta"
        assert normalize_endpoint(url2) == "/api/v1/workspaces/{id}/competitors/{id}"

        # Root and static paths
        assert normalize_endpoint("/health") == "/health"
        assert normalize_endpoint("/metrics") == "/metrics"

    def test_track_http_request(self) -> None:
        track_http_request(
            method="GET",
            path="/api/v1/workspaces/test-ws/sources",
            status_code=200,
            duration_seconds=0.045,
        )
        metrics_bytes = generate_prometheus_metrics()
        metrics_text = metrics_bytes.decode("utf-8")
        assert "http_requests_total" in metrics_text
        assert 'endpoint="/api/v1/workspaces/{id}/sources"' in metrics_text

    def test_track_workflow_and_stage_execution(self) -> None:
        track_workflow_run(status="completed", trigger_type="scheduled_monitoring", duration_seconds=1.23)
        track_stage_execution(
            stage="event_analysis",
            status="failed",
            duration_seconds=0.45,
            error_category="transient",
            retries=2,
        )

        metrics_text = generate_prometheus_metrics().decode("utf-8")
        assert "workflow_runs_total" in metrics_text
        assert 'trigger_type="scheduled_monitoring"' in metrics_text
        assert "workflow_failures_total" in metrics_text
        assert 'error_category="transient"' in metrics_text
        assert "workflow_stage_retries_total" in metrics_text

    def test_track_monitoring_jobs_and_leases(self) -> None:
        set_active_monitoring_leases(7)
        track_monitoring_job(
            status="failed",
            duration_seconds=2.5,
            platform="crawl4ai",
            category="pricing",
            failure_reason="timeout",
        )

        metrics_text = generate_prometheus_metrics().decode("utf-8")
        assert "monitoring_active_leases 7.0" in metrics_text
        assert "source_acquisition_failures_total" in metrics_text
        assert 'platform="crawl4ai"' in metrics_text
        assert 'failure_reason="timeout"' in metrics_text

    def test_track_llm_and_retrieval(self) -> None:
        track_llm_call(
            provider="openai",
            model="gpt-4o-mini",
            status="success",
            duration_seconds=0.35,
            prompt_tokens=150,
            completion_tokens=50,
            retries=1,
        )
        track_llm_budget_exhaustion("token_budget")

        track_retrieval(
            route="hybrid_rrf",
            mode="adaptive",
            status="success",
            duration_seconds=0.08,
            chunk_count=5,
            reranked=True,
            escalated=True,
        )

        metrics_text = generate_prometheus_metrics().decode("utf-8")
        assert "llm_requests_total" in metrics_text
        assert "llm_tokens_total" in metrics_text
        assert "llm_budget_exhaustion_total" in metrics_text
        assert "retrieval_requests_total" in metrics_text
        assert "retrieval_reranked_total" in metrics_text
        assert "retrieval_escalated_total" in metrics_text

    def test_observability_failure_isolation(self) -> None:
        # Verify that corrupt or unexpected input does not raise unhandled exceptions
        track_http_request(method="", path="", status_code=0, duration_seconds=-1.0)
        track_stage_execution(stage="", status="", duration_seconds=-5.0)
        track_monitoring_job(status="", duration_seconds=-2.0)
        track_llm_call(provider="", model="", status="", duration_seconds=-1.0)
        track_retrieval(route="", mode="", status="", duration_seconds=-1.0)
        set_active_monitoring_leases(-10)


class TestMetricsEndpointAndDiagnosticsAPI:
    """Test HTTP API exposure for /metrics and monitoring diagnostics."""

    @pytest.fixture
    def test_client(self) -> TestClient:
        app = create_app()
        return TestClient(app)

    def test_metrics_scrape_endpoint(self, test_client: TestClient) -> None:
        response = test_client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        assert "http_requests_total" in body
        assert "monitoring_active_leases" in body

    def test_health_endpoints_remain_operational(self, test_client: TestClient) -> None:
        res = test_client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ("ok", "degraded")

        res_live = test_client.get("/health/live")
        assert res_live.status_code == 200
        assert res_live.json()["status"] == "ok"
        assert res_live.json()["live"] is True


    def test_correlation_id_propagated_in_headers(self, test_client: TestClient) -> None:
        custom_id = "test-corr-abc-123"
        response = test_client.get("/health", headers={"X-Correlation-ID": custom_id})
        assert response.status_code == 200
        assert response.headers.get("X-Correlation-ID") == custom_id

    @pytest.mark.anyio
    async def test_monitoring_diagnostics_endpoint(self) -> None:
        from collections.abc import AsyncGenerator
        from httpx import ASGITransport, AsyncClient
        from sqlalchemy.ext.asyncio import AsyncSession
        from competitive_intelligence.infrastructure.database.session import DatabaseSessionManager
        from competitive_intelligence.application.api.deps import get_session, get_db_manager

        db_mgr = DatabaseSessionManager(database_url="sqlite+aiosqlite:///:memory:")
        await db_mgr.create_tables()

        app = create_app()

        async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
            async with db_mgr.session() as session:
                yield session

        app.dependency_overrides[get_session] = override_get_session
        app.dependency_overrides[get_db_manager] = lambda: db_mgr

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create a workspace
            ws_res = await client.post(
                "/api/v1/workspaces",
                json={"name": "Obs Workspace", "owner_user_id": "usr_test123"},
            )
            assert ws_res.status_code == 201
            ws_id = ws_res.json()["workspace_id"]

            # Call monitoring diagnostics
            diag_res = await client.get(f"/api/v1/workspaces/{ws_id}/sources/monitoring/diagnostics")
            assert diag_res.status_code == 200
            diag_data = diag_res.json()
            assert diag_data["workspace_id"] == ws_id
            assert "total_sources" in diag_data
            assert "healthy_sources_count" in diag_data
            assert "failing_sources_count" in diag_data
            assert "active_leases_count" in diag_data
            assert "scheduler_mode" in diag_data

        await db_mgr.close()


    def test_metrics_token_protection(self, test_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = get_settings()
        monkeypatch.setattr(settings, "metrics_token", "secret-metrics-token-123")

        # Without token -> 401
        unauth_res = test_client.get("/metrics")
        assert unauth_res.status_code == 401

        # With query token -> 200
        auth_res_query = test_client.get("/metrics?token=secret-metrics-token-123")
        assert auth_res_query.status_code == 200
        assert "http_requests_total" in auth_res_query.text

        # With Bearer header -> 200
        auth_res_header = test_client.get("/metrics", headers={"Authorization": "Bearer secret-metrics-token-123"})
        assert auth_res_header.status_code == 200
        assert "http_requests_total" in auth_res_header.text

