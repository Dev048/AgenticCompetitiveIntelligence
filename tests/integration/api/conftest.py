"""Shared fixtures for API integration tests."""

import asyncio
from collections.abc import Iterator

import pytest

from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.application.api import deps
from competitive_intelligence.core.config import get_settings
from competitive_intelligence.demo.replay_llm import LLMFixtureStore, ReplayLLMClient
from competitive_intelligence.demo.seed import seed_demo_workspace
from competitive_intelligence.infrastructure.database.session import (
    get_db_manager,
    reset_db_managers,
)


@pytest.fixture(scope="session")
def seeded_demo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    """Demo workspace seeded once via the real pipeline (mock LLM + mock embeddings).

    Yields the environment the API must run with to see that data.
    """
    root = tmp_path_factory.mktemp("query")
    env = {
        "ENVIRONMENT": "development",
        "LLM_PROVIDER": "mock",
        "EMBEDDING_PROVIDER": "mock",
        "QDRANT_PATH": str(root / "qdrant"),
        "DATABASE_URL": f"sqlite+aiosqlite:///{root / 'query.db'}",
    }
    patch = pytest.MonkeyPatch()
    for key, value in env.items():
        patch.setenv(key, value)
    get_settings.cache_clear()
    deps.reset_resource_caches()

    async def _seed() -> None:
        recorder = ReplayLLMClient(
            LLMFixtureStore(root / "fx.json"), mode="record", inner=MockLLMClient()
        )
        summary = await seed_demo_workspace(llm_client=recorder, db_manager=get_db_manager())
        assert summary.steps_failed == 0
        await reset_db_managers()

    try:
        asyncio.run(_seed())
    finally:
        # Release the embedded Qdrant folder and restore the environment right away so
        # the seeding env never leaks into unrelated tests; users of this fixture apply
        # ``env`` themselves (e.g. with monkeypatch).
        deps.reset_resource_caches()
        patch.undo()
        get_settings.cache_clear()
    yield env
