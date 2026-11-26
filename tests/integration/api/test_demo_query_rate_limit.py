"""DEMO_MODE rate-limits POST /workspaces/{id}/query per client."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from competitive_intelligence.agentic_rag.llm.guarded_client import reset_llm_call_guard
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.application.api import deps
from competitive_intelligence.application.api.app import create_app
from competitive_intelligence.application.api.demo_guard import reset_query_rate_limiter
from competitive_intelligence.core.config import get_settings

URL = "/api/v1/workspaces/test-workspace/query"  # seeded by the seeded_workspace fixture


@pytest.fixture
def client(
    seeded_workspace: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    for key, value in seeded_workspace.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEMO_QUERY_PER_IP_PER_MINUTE", "2")
    get_settings.cache_clear()
    reset_llm_call_guard()
    reset_query_rate_limiter()
    app = create_app()
    llm = MockLLMClient()
    app.dependency_overrides[deps.get_query_llm_client] = lambda: llm
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    reset_llm_call_guard()
    reset_query_rate_limiter()
    get_settings.cache_clear()


def test_demo_mode_rate_limits_queries(client: TestClient) -> None:
    statuses = [
        client.post(URL, json={"query_text": "Acme pricing"}).status_code for _ in range(3)
    ]

    assert statuses == [200, 200, 429]
    limited = client.post(URL, json={"query_text": "Acme pricing"})
    assert limited.json()["error_code"] == "RATE_LIMITED"
    assert int(limited.headers["Retry-After"]) > 0
