"""POST /workspaces/{id}/query: evidence with provenance and an agentic trace."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from competitive_intelligence.agentic_rag.llm.errors import LLMRateLimitError
from competitive_intelligence.agentic_rag.llm.guarded_client import (
    GuardedLLMClient,
    LLMCallGuard,
    reset_llm_call_guard,
)
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient, T
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.application.api import deps
from competitive_intelligence.application.api.app import create_app
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.config import get_settings

WORKSPACE_ID = "test-workspace"  # seeded by the seeded_workspace fixture
URL = f"/api/v1/workspaces/{WORKSPACE_ID}/query"


class _RateLimitedLLM(BaseLLMClient):
    """Simulates Groq answering 429 on every call."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> T:
        self.calls += 1
        raise LLMRateLimitError("OpenAI rate limit exceeded (429) after 1 retries")

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> str:
        raise LLMRateLimitError("429")


@pytest.fixture
def client(seeded_workspace: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    for key, value in seeded_workspace.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    reset_llm_call_guard()
    app = create_app()
    with TestClient(app) as test_client:
        test_client.app_ref = app  # type: ignore[attr-defined]
        yield test_client
    app.dependency_overrides.clear()
    reset_llm_call_guard()
    get_settings.cache_clear()


def _use_llm(client: TestClient, llm: BaseLLMClient) -> None:
    client.app_ref.dependency_overrides[deps.get_query_llm_client] = lambda: llm  # type: ignore[attr-defined]


def _post(client: TestClient, **body: Any) -> Any:
    payload = {"query_text": "Acme Analytics pricing change", **body}
    response = client.post(URL, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_auto_mode_returns_grounded_evidence_with_provenance(client: TestClient) -> None:
    _use_llm(client, MockLLMClient())
    body = _post(client, top_k=3)

    assert body["workspace_id"] == WORKSPACE_ID
    assert 0 < len(body["evidence"]) <= 3
    for item in body["evidence"]:
        assert item["chunk_id"] and item["document_id"] and item["version_id"]
        assert item["canonical_url"].endswith(".example/pricing") or ".example" in item["canonical_url"]
    trace = body["trace"]
    assert trace["mode"] == "auto"
    assert trace["final_source"] in {"adaptive", "agentic", "adaptive_fallback"}
    assert trace["degraded"] is False


def test_agentic_mode_exposes_query_analysis_and_verification(client: TestClient) -> None:
    _use_llm(client, MockLLMClient())
    trace = _post(client, mode="agentic")["trace"]

    assert trace["mode"] == "agentic"
    assert trace["escalated"] is True
    assert trace["query_decision"] in {"direct_retrieval", "decompose_and_retrieve"}
    assert trace["sub_queries"], trace
    assert trace["verifications"], trace
    assert trace["llm_calls"] >= 2
    assert trace["degraded"] is False


def test_groq_429_degrades_gracefully_and_cools_down(client: TestClient) -> None:
    inner = _RateLimitedLLM()
    guard = LLMCallGuard(max_concurrency=2, queue_timeout_seconds=5, cooldown_seconds=60)
    _use_llm(client, GuardedLLMClient(inner, guard))

    first = _post(client, mode="agentic")
    assert first["evidence"], "deterministic retrieval must still answer"
    assert first["trace"]["degraded"] is True
    assert "rate limit" in first["trace"]["degraded_reason"]
    assert inner.calls == 1

    second = _post(client, mode="agentic")
    assert second["trace"]["degraded"] is True
    assert inner.calls == 1, "cool-down must stop calls to the rate-limited provider"


def test_query_is_workspace_scoped(client: TestClient) -> None:
    _use_llm(client, MockLLMClient())
    response = client.post(
        "/api/v1/workspaces/some-other-workspace/query",
        json={"query_text": "Acme Analytics pricing change"},
    )
    assert response.status_code == 404


def test_query_validation(client: TestClient) -> None:
    assert client.post(URL, json={"query_text": "hi"}).status_code == 422
    assert client.post(URL, json={"query_text": "valid query", "top_k": 50}).status_code == 422
    assert client.post(URL, json={"query_text": "valid query", "mode": "yolo"}).status_code == 422


def test_query_retriever_uses_the_low_query_budget(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QUERY_MAX_LLM_CALLS", "3")
    monkeypatch.setenv("QUERY_TOKEN_BUDGET", "4000")
    monkeypatch.setenv("QUERY_MAX_HOPS", "1")
    monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "12")
    get_settings.cache_clear()

    retriever = deps.get_query_retriever(
        adaptive_retriever=deps.get_adaptive_retriever(), llm_client=MockLLMClient()
    )
    agentic = retriever.agentic_retriever
    assert (agentic.max_llm_calls, agentic.task_token_budget, agentic.max_hops) == (3, 4000, 1)  # type: ignore[attr-defined]
    assert agentic.timeout_seconds == 12  # type: ignore[attr-defined]


def test_live_llm_client_is_guarded(seeded_workspace: dict[str, str]) -> None:
    assert isinstance(deps.get_llm_client(), GuardedLLMClient)
    assert isinstance(deps.get_query_llm_client(), GuardedLLMClient)

