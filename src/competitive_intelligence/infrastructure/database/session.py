"""Database session and async engine management for PostgreSQL persistence."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from competitive_intelligence.core.config import Settings, get_settings
from competitive_intelligence.infrastructure.database.models import Base


class DatabaseSessionManager:
    """Manages AsyncEngine lifecycle and creates scoped AsyncSessions."""

    def __init__(self, database_url: str, **engine_kwargs: object) -> None:
        self._database_url = database_url
        self._engine_kwargs = engine_kwargs
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

    @property
    def engine(self) -> AsyncEngine:
        """Return the initialized AsyncEngine or initialize on demand."""
        if self._engine is None:
            # Handle SQLite specific parameters vs PostgreSQL
            is_sqlite = "sqlite" in self._database_url
            kwargs = dict(self._engine_kwargs)
            if is_sqlite:
                kwargs.pop("pool_size", None)
                kwargs.pop("max_overflow", None)
                kwargs.pop("pool_timeout", None)
                if ":memory:" in self._database_url:
                    from sqlalchemy.pool import StaticPool

                    kwargs.setdefault("poolclass", StaticPool)
                    raw_connect_args = kwargs.get("connect_args")
                    connect_args: dict[str, object] = dict(raw_connect_args) if isinstance(raw_connect_args, dict) else {}
                    connect_args.setdefault("check_same_thread", False)
                    kwargs["connect_args"] = connect_args

            self._engine = create_async_engine(self._database_url, **kwargs)
            self._sessionmaker = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
        return self._engine

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        """Return the initialized session factory."""
        if self._sessionmaker is None:
            _ = self.engine  # Trigger initialization
        assert self._sessionmaker is not None
        return self._sessionmaker

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Async context manager yielding a managed transaction session."""
        session: AsyncSession = self.sessionmaker()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_tables(self) -> None:
        """Create all declarative tables (useful for test isolation)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all declarative tables (useful for test cleanup)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def close(self) -> None:
        """Dispose of the underlying engine connection pool."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None


def get_db_manager(settings: Settings | None = None) -> DatabaseSessionManager:
    """Factory creating a DatabaseSessionManager from application Settings."""
    cfg = settings or get_settings()
    return DatabaseSessionManager(
        database_url=cfg.database_url,
        pool_size=cfg.db_pool_size,
        max_overflow=cfg.db_max_overflow,
        pool_timeout=cfg.db_pool_timeout,
        echo=cfg.db_echo,
    )
