"""get_db_manager() returns one engine/pool per database configuration per process."""

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import text

from competitive_intelligence.core.config import Settings
from competitive_intelligence.infrastructure.database import session as session_module
from competitive_intelligence.infrastructure.database.session import (
    get_db_manager,
    reset_db_managers,
)


@pytest.fixture(autouse=True)
async def _clean_cache() -> AsyncIterator[None]:
    await reset_db_managers()
    yield
    await reset_db_managers()


def _settings(path: Path, **overrides: object) -> Settings:
    return Settings(database_url=f"sqlite+aiosqlite:///{path}", **overrides)  # type: ignore[arg-type]


def test_same_configuration_returns_the_same_manager(tmp_path: Path) -> None:
    cfg = _settings(tmp_path / "a.db")
    assert get_db_manager(cfg) is get_db_manager(cfg)
    # An equal but distinct Settings object maps to the same manager.
    assert get_db_manager(_settings(tmp_path / "a.db")) is get_db_manager(cfg)


def test_different_configurations_get_separate_managers(tmp_path: Path) -> None:
    a = get_db_manager(_settings(tmp_path / "a.db"))
    b = get_db_manager(_settings(tmp_path / "b.db"))
    echo = get_db_manager(_settings(tmp_path / "a.db", db_echo=True))
    assert len({id(a), id(b), id(echo)}) == 3


def test_concurrent_first_calls_create_one_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[object] = []
    real_cls = session_module.DatabaseSessionManager

    class _Counting(real_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            created.append(self)
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(session_module, "DatabaseSessionManager", _Counting)
    cfg = _settings(tmp_path / "race.db")

    with ThreadPoolExecutor(max_workers=16) as pool:
        managers = list(pool.map(lambda _: get_db_manager(cfg), range(64)))

    assert len(created) == 1
    assert all(m is managers[0] for m in managers)


async def test_cached_manager_still_works_after_close(tmp_path: Path) -> None:
    manager = get_db_manager(_settings(tmp_path / "reuse.db"))
    await manager.create_tables()
    await manager.close()  # app shutdown disposes the pool

    async with get_db_manager(_settings(tmp_path / "reuse.db")).session() as session:
        assert (await session.execute(text("SELECT 1"))).scalar_one() == 1


async def test_reset_forgets_managers(tmp_path: Path) -> None:
    cfg = _settings(tmp_path / "reset.db")
    first = get_db_manager(cfg)
    await reset_db_managers()
    assert get_db_manager(cfg) is not first
