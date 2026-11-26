"""Demo seeding: real pipeline over the fictional corpus, replayed LLM, idempotent, no crawl."""

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import func, select

from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.application.api import deps
from competitive_intelligence.core.config import get_settings
from competitive_intelligence.demo.corpus import DEMO_WORKSPACE_ID, pipeline_steps
from competitive_intelligence.demo.replay_llm import LLMFixtureStore, ReplayLLMClient
from competitive_intelligence.demo.seed import seed_demo_workspace
from competitive_intelligence.infrastructure.database.models import (
    IntelligenceEventOrm,
    RecommendationOrm,
    SourceOrm,
    TrackedCompetitorOrm,
)
from competitive_intelligence.infrastructure.database.session import DatabaseSessionManager
from competitive_intelligence.source_acquisition.providers import crawl4ai_client

N_STEPS = len(pipeline_steps())


@pytest.fixture
def demo_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))

    def _no_crawling(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Demo seeding must never crawl")

    for name in dir(crawl4ai_client):
        obj = getattr(crawl4ai_client, name)
        if isinstance(obj, type) and obj.__module__ == crawl4ai_client.__name__:
            monkeypatch.setattr(obj, "__init__", _no_crawling)

    get_settings.cache_clear()
    deps.reset_resource_caches()
    yield tmp_path
    deps.reset_resource_caches()
    get_settings.cache_clear()


def _db(tmp_path: Path, name: str) -> DatabaseSessionManager:
    return DatabaseSessionManager(database_url=f"sqlite+aiosqlite:///{tmp_path / name}")


async def _count(manager: DatabaseSessionManager, orm: type) -> int:
    async with manager.session() as session:
        stmt = select(func.count()).select_from(orm).where(orm.workspace_id == DEMO_WORKSPACE_ID)
        return int((await session.execute(stmt)).scalar_one())


def _fresh_vector_store(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("QDRANT_PATH", str(path))
    get_settings.cache_clear()
    deps.reset_resource_caches()


async def test_record_then_replay_is_idempotent_and_needs_no_live_llm(
    demo_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = demo_env / "llm_replay.json"

    # 1. Record once (MockLLMClient stands in for Groq here).
    recorder = ReplayLLMClient(LLMFixtureStore(fixtures), mode="record", inner=MockLLMClient())
    recorded = await seed_demo_workspace(llm_client=recorder, db_manager=_db(demo_env, "rec.db"))
    recorder.store.save()
    assert recorded.steps_failed == 0, recorded.steps
    assert recorded.llm.recorded > 0
    assert fixtures.exists()

    # 2. Fresh boot: new database and vector store, replay only.
    _fresh_vector_store(monkeypatch, demo_env / "qdrant-boot")
    replayer = ReplayLLMClient(LLMFixtureStore.load(fixtures))
    boot = _db(demo_env, "boot.db")
    first = await seed_demo_workspace(llm_client=replayer, db_manager=boot)

    assert first.steps_failed == 0, first.steps
    assert first.steps_run == N_STEPS
    assert first.llm.misses == 0
    assert first.llm.exact_hits + first.llm.ordinal_hits == recorded.llm.recorded
    assert await _count(boot, TrackedCompetitorOrm) == 3
    assert await _count(boot, SourceOrm) == 6
    assert await _count(boot, IntelligenceEventOrm) > 0
    recommendations = await _count(boot, RecommendationOrm)
    assert recommendations > 0

    async with boot.session() as session:
        statuses = (await session.execute(select(RecommendationOrm.status))).scalars().all()
    assert set(statuses) == {"proposed"}

    # 3. Second boot on the same disk: nothing re-runs, nothing duplicates.
    events_before = await _count(boot, IntelligenceEventOrm)
    second = await seed_demo_workspace(llm_client=replayer, db_manager=boot)
    assert (second.steps_run, second.steps_skipped) == (0, N_STEPS)
    assert await _count(boot, IntelligenceEventOrm) == events_before
    assert await _count(boot, RecommendationOrm) == recommendations
    await boot.close()


async def test_replay_miss_falls_back_without_failing_the_seed(demo_env: Path) -> None:
    empty = ReplayLLMClient(LLMFixtureStore(demo_env / "missing.json"))
    manager = _db(demo_env, "fallback.db")

    summary = await seed_demo_workspace(llm_client=empty, db_manager=manager)

    assert summary.llm.misses > 0
    assert summary.steps_failed == 0, summary.steps
    await manager.close()


@pytest.fixture
def demo_boot_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("ENVIRONMENT", "demo")
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "bge_m3")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'boot.db'}")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    get_settings.cache_clear()
    deps.reset_resource_caches()
    yield
    deps.reset_resource_caches()
    get_settings.cache_clear()


def _patched_app(monkeypatch: pytest.MonkeyPatch, seed: object) -> object:
    app_module = importlib.import_module("competitive_intelligence.application.api.app")
    from competitive_intelligence.demo import seed as seed_module
    monkeypatch.setattr(seed_module, "seed_demo_workspace", seed)
    return app_module.create_app()


def test_demo_boot_seeds_from_replayed_fixtures(
    demo_boot_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from competitive_intelligence.demo.seed import SeedSummary

    calls: list[object] = []

    async def _fake_seed(llm_client: object = None, **_: object) -> SeedSummary:
        calls.append(llm_client)
        return SeedSummary(workspace_id=DEMO_WORKSPACE_ID)

    with TestClient(_patched_app(monkeypatch, _fake_seed)) as client:  # type: ignore[arg-type]
        assert client.get("/health/live").status_code == 200

    assert len(calls) == 1
    assert isinstance(calls[0], ReplayLLMClient)
    assert calls[0].mode == "replay"


def test_demo_boot_survives_seed_failure(
    demo_boot_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    async def _broken_seed(**_: object) -> None:
        raise RuntimeError("fixture file corrupted")

    with TestClient(_patched_app(monkeypatch, _broken_seed)) as client:  # type: ignore[arg-type]
        assert client.get("/health/live").status_code == 200
