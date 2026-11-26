"""ENVIRONMENT=demo validation, the DEMO_MODE read-only guard, and query rate limiting."""

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from competitive_intelligence.application.api import demo_guard
from competitive_intelligence.application.api.app import create_app
from competitive_intelligence.application.api.demo_guard import (
    Clock,
    QueryRateLimiter,
    client_key_for,
    is_write_allowed_in_demo,
)
from competitive_intelligence.application.api.error_handlers import register_exception_handlers
from competitive_intelligence.core.config import (
    ProductionConfigurationError,
    Settings,
    get_settings,
)
from competitive_intelligence.core.errors import RateLimitExceededError

# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def _demo_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "demo",
        "demo_mode": True,
        "database_url": "sqlite+aiosqlite:////data/demo.db",
        "qdrant_path": "/data/qdrant",
        "llm_provider": "openai",
        "openai_api_key": "test-key-not-real",
        "embedding_provider": "bge_m3",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_demo_allows_sqlite_and_embedded_qdrant_with_real_providers() -> None:
    assert _demo_settings().validate_production_configuration(raise_exception=False) == []


def test_demo_allows_in_memory_qdrant() -> None:
    cfg = _demo_settings(qdrant_path=None, qdrant_location=":memory:")
    assert cfg.validate_production_configuration(raise_exception=False) == []


def test_demo_rejects_mock_llm() -> None:
    errors = _demo_settings(llm_provider="mock").validate_production_configuration(False)
    assert errors == [
        "Mock LLM provider is not permitted in demo. "
        "Configure 'openai' or 'anthropic' with valid credentials."
    ]


def test_demo_rejects_mock_embeddings() -> None:
    errors = _demo_settings(embedding_provider="mock").validate_production_configuration(False)
    assert errors == [
        "Mock embedding provider is not permitted in demo. "
        "Configure 'bge_m3' for production embeddings."
    ]


def test_demo_requires_llm_credentials() -> None:
    errors = _demo_settings(openai_api_key="  ").validate_production_configuration(False)
    assert errors == ["OPENAI_API_KEY must be provided when llm_provider is 'openai'."]


def test_demo_requires_demo_mode_and_no_debug() -> None:
    errors = _demo_settings(demo_mode=False, debug=True).validate_production_configuration(False)
    assert errors == [
        "DEBUG mode must be disabled in demo.",
        "DEMO_MODE must be enabled when ENVIRONMENT=demo (public read-only).",
    ]


def test_demo_validation_raises_on_startup_path() -> None:
    with pytest.raises(ProductionConfigurationError):
        _demo_settings(llm_provider="mock").validate_production_configuration()


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_production_and_staging_reject_everything_demo_allows(environment: str) -> None:
    """Pins the full production/staging error list so the demo refactor cannot loosen it."""
    cfg = Settings(
        environment=environment,  # type: ignore[arg-type]
        debug=True,
        demo_mode=True,
        database_url="sqlite+aiosqlite:///./demo.db",
        llm_provider="mock",
        embedding_provider="mock",
        qdrant_location=":memory:",
        qdrant_path="/data/qdrant",
    )

    assert cfg.validate_production_configuration(raise_exception=False) == [
        "DEBUG mode must be disabled in production/staging.",
        f"SQLite database is not permitted in {environment}. "
        "A production PostgreSQL URL must be configured.",
        f"Mock LLM provider is not permitted in {environment}. "
        "Configure 'openai' or 'anthropic' with valid credentials.",
        f"Mock embedding provider is not permitted in {environment}. "
        "Configure 'bge_m3' for production embeddings.",
        f"In-memory Qdrant (:memory:) is not permitted in {environment}. "
        "Configure external QDRANT_URL or QDRANT_HOST.",
        f"Embedded Qdrant (QDRANT_PATH) is not permitted in {environment}. "
        "Configure external QDRANT_URL or QDRANT_HOST.",
    ]


@pytest.mark.parametrize("environment", ["production", "staging"])
@pytest.mark.parametrize(
    ("provider", "key_field", "message"),
    [
        ("openai", "openai_api_key", "OPENAI_API_KEY must be provided when llm_provider is 'openai'."),
        (
            "anthropic",
            "anthropic_api_key",
            "ANTHROPIC_API_KEY must be provided when llm_provider is 'anthropic'.",
        ),
    ],
)
def test_production_and_staging_still_require_llm_credentials(
    environment: str, provider: str, key_field: str, message: str
) -> None:
    cfg = Settings(
        environment=environment,  # type: ignore[arg-type]
        database_url="postgresql+asyncpg://u:p@db:5432/ci",
        llm_provider=provider,  # type: ignore[arg-type]
        embedding_provider="bge_m3",
        qdrant_url="http://qdrant:6333",
        **{key_field: None},
    )
    assert cfg.validate_production_configuration(raise_exception=False) == [message]


def test_demo_settings_are_configurable_by_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_QUERY_PER_IP_PER_MINUTE", "3")
    monkeypatch.setenv("DEMO_QUERY_DAILY_CAP", "42")
    monkeypatch.setenv("DEMO_TRUSTED_PROXY_HOPS", "2")
    monkeypatch.setenv("DEMO_ALLOW_RECOMMENDATION_REVIEW", "false")
    monkeypatch.setenv("QDRANT_PATH", "/data/qdrant")

    cfg = Settings()

    assert cfg.demo_mode is True
    assert cfg.demo_query_per_ip_per_minute == 3
    assert cfg.demo_query_daily_cap == 42
    assert cfg.demo_trusted_proxy_hops == 2
    assert cfg.demo_allow_recommendation_review is False
    assert cfg.qdrant_path == "/data/qdrant"


# ---------------------------------------------------------------------------
# Read-only guard
# ---------------------------------------------------------------------------

_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


def _mutating_routes() -> list[tuple[str, str]]:
    # The OpenAPI schema lists every route, including nested routers (app.routes does not
    # flatten included routers in recent FastAPI versions).
    paths = create_app().openapi()["paths"]
    return sorted(
        (method.upper(), _concrete(path))
        for path, operations in paths.items()
        for method in operations
        if method.upper() in _MUTATING
    )


_ALLOW_LIST = {
    ("PATCH", "/api/v1/workspaces/x/recommendations/x/status"),
    ("POST", "/api/v1/workspaces/x/query"),
}


def test_every_mutating_route_is_blocked_except_the_allow_list() -> None:
    """Walks every registered route, so new write endpoints are blocked by default."""
    cfg = Settings(demo_mode=True)
    routes = _mutating_routes()
    allowed = {(m, p) for m, p in routes if is_write_allowed_in_demo(m, p, cfg)}

    assert len(routes) > len(_ALLOW_LIST)
    assert allowed == _ALLOW_LIST


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/v1/workspaces"),
        ("POST", "/api/v1/workspaces/x/competitors"),
        ("PATCH", "/api/v1/workspaces/x/competitors/x"),
        ("POST", "/api/v1/workspaces/x/sources"),
        ("PATCH", "/api/v1/workspaces/x/sources/x"),
        ("POST", "/api/v1/workspaces/x/sources/x/trigger"),
        ("POST", "/api/v1/workspaces/x/workflows/runs"),
        ("POST", "/api/v1/workspaces/x/workflows/runs/x/resume"),
        ("POST", "/api/v1/workspaces/x/workflows/runs/x/cancel"),
        ("POST", "/api/v1/workspaces/x/decisions/contexts"),
        ("DELETE", "/api/v1/workspaces/x"),
        ("POST", "/metrics"),
        ("PUT", "/index.html"),
        # Lookalikes of allow-listed routes must not slip through.
        ("POST", "/api/v1/workspaces/x/query/extra"),
        ("PUT", "/api/v1/workspaces/x/query"),
        ("POST", "/api/v1/workspaces/x/recommendations/x/status"),
        ("PATCH", "/api/v1/workspaces/x/recommendations/x"),
        ("PATCH", "/api/v1/workspaces/x/y/recommendations/x/status"),
    ],
)
def test_guard_blocks_writes(method: str, path: str) -> None:
    assert not is_write_allowed_in_demo(method, path, Settings(demo_mode=True))


def test_guard_allows_reads_query_and_review() -> None:
    cfg = Settings(demo_mode=True)
    assert is_write_allowed_in_demo("GET", "/api/v1/workspaces", cfg)
    assert is_write_allowed_in_demo("OPTIONS", "/api/v1/workspaces", cfg)
    assert is_write_allowed_in_demo("POST", "/api/v1/workspaces/ws_1/query", cfg)
    assert is_write_allowed_in_demo("PATCH", "/api/v1/workspaces/ws_1/recommendations/r1/status", cfg)


def test_operator_review_can_be_disabled() -> None:
    cfg = Settings(demo_mode=True, demo_allow_recommendation_review=False)
    assert not is_write_allowed_in_demo(
        "PATCH", "/api/v1/workspaces/ws_1/recommendations/r1/status", cfg
    )


@pytest.fixture
def demo_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'guard.db'}")
    get_settings.cache_clear()
    yield TestClient(create_app())
    get_settings.cache_clear()


def test_blocked_route_returns_clear_read_only_json(demo_client: TestClient) -> None:
    response = demo_client.post(
        "/api/v1/workspaces/ws_1/sources/src_1/trigger",
        json={},
        headers={"X-Correlation-ID": "corr-123"},
    )

    assert response.status_code == 403
    body = response.json()
    assert body["error_code"] == "DEMO_READ_ONLY"
    assert "read-only public demo" in body["message"]
    assert body["correlation_id"] == "corr-123"
    assert body["details"][0]["code"] == "demo_allow_list"


def test_guard_is_inert_without_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "false")
    get_settings.cache_clear()
    try:
        # Invalid body: reaching request validation (422) proves the guard let it through.
        response = TestClient(create_app()).post("/api/v1/workspaces", json={})
    finally:
        get_settings.cache_clear()
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class _FakeClock(Clock):
    def __init__(self) -> None:
        self.mono = 1000.0
        self.now = datetime(2026, 11, 26, 23, 59, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.mono

    def utc_now(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.now += timedelta(seconds=seconds)


def test_per_client_limit_with_sliding_window() -> None:
    clock = _FakeClock()
    limiter = QueryRateLimiter(per_client_per_minute=2, daily_cap=100, clock=clock)
    limiter.check("1.1.1.1")
    clock.advance(10)
    limiter.check("1.1.1.1")

    with pytest.raises(RateLimitExceededError) as exc:
        limiter.check("1.1.1.1")
    assert exc.value.retry_after_seconds == 51
    assert exc.value.details == {"scope": "per_client"}

    limiter.check("2.2.2.2")  # other clients are unaffected
    clock.advance(50)  # first hit leaves the window
    limiter.check("1.1.1.1")


def test_rejected_requests_do_not_consume_quota() -> None:
    clock = _FakeClock()
    limiter = QueryRateLimiter(per_client_per_minute=1, daily_cap=10, clock=clock)
    limiter.check("a")
    for _ in range(5):
        with pytest.raises(RateLimitExceededError):
            limiter.check("a")
    assert limiter.remaining_today == 9


def test_global_daily_cap_and_utc_midnight_reset() -> None:
    clock = _FakeClock()
    limiter = QueryRateLimiter(per_client_per_minute=100, daily_cap=2, clock=clock)
    limiter.check("a")
    limiter.check("b")

    with pytest.raises(RateLimitExceededError) as exc:
        limiter.check("c")
    assert exc.value.details == {"scope": "global_daily"}
    assert exc.value.retry_after_seconds == 60  # 23:59 -> midnight

    clock.advance(61)  # next UTC day
    limiter.check("c")
    assert limiter.remaining_today == 1


def _request(peer: str, forwarded: str | None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_client_key_uses_trusted_proxy_hop_not_spoofable_left_entries() -> None:
    req = _request("10.0.0.1", "6.6.6.6, 203.0.113.7")
    assert client_key_for(req, trusted_proxy_hops=1) == "203.0.113.7"
    assert client_key_for(req, trusted_proxy_hops=2) == "6.6.6.6"
    assert client_key_for(req, trusted_proxy_hops=0) == "10.0.0.1"
    assert client_key_for(_request("10.0.0.1", None), trusted_proxy_hops=1) == "10.0.0.1"


def test_limiter_singleton_reads_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_QUERY_PER_IP_PER_MINUTE", "7")
    monkeypatch.setenv("DEMO_QUERY_DAILY_CAP", "11")
    get_settings.cache_clear()
    demo_guard.reset_query_rate_limiter()
    try:
        limiter = demo_guard.get_query_rate_limiter()
        assert (limiter.per_client_per_minute, limiter.daily_cap) == (7, 11)
    finally:
        demo_guard.reset_query_rate_limiter()
        get_settings.cache_clear()


def test_rate_limit_error_maps_to_429_with_retry_after() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/_raise_rate_limit")
    async def _raise() -> None:
        raise RateLimitExceededError("Too many live queries.", retry_after_seconds=17)

    response = TestClient(app).get("/_raise_rate_limit")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "17"
    assert response.json()["error_code"] == "RATE_LIMITED"
