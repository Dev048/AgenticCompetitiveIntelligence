"""Integration tests for Health, Liveness, and Readiness probes."""

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch

from competitive_intelligence.application.api.app import create_app
from competitive_intelligence.infrastructure.database.session import DatabaseSessionManager


@pytest.fixture
def in_memory_db_manager() -> DatabaseSessionManager:
    return DatabaseSessionManager("sqlite+aiosqlite:///:memory:")


@pytest.mark.anyio
async def test_health_liveness_probe() -> None:
    """GET /health/live returns HTTP 200 indicating process liveness."""
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["live"] is True


@pytest.mark.anyio
async def test_health_readiness_probe_healthy(in_memory_db_manager: DatabaseSessionManager) -> None:
    """GET /health/ready returns HTTP 200 when database and vector store are responsive."""
    app = create_app()
    from competitive_intelligence.application.api.deps import get_db_manager

    app.dependency_overrides[get_db_manager] = lambda: in_memory_db_manager

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["ready"] is True
        assert data["database"] == "connected"
        assert data["qdrant"] == "connected"


@pytest.mark.anyio
async def test_health_readiness_probe_degraded_returns_503() -> None:
    """GET /health/ready returns HTTP 503 when an infrastructure service is disconnected."""
    app = create_app()

    with patch(
        "competitive_intelligence.application.api.v1.endpoints.health._check_database",
        return_value="unhealthy",
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "unready"
            assert data["ready"] is False
            assert data["database"] == "unhealthy"


@pytest.mark.anyio
async def test_health_comprehensive_endpoint() -> None:
    """GET /health returns comprehensive status with database, qdrant, and environment."""
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "database" in data
        assert "qdrant" in data
        assert "environment" in data
