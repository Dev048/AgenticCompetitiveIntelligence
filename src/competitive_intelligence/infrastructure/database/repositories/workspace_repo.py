"""SQL repositories for User and Workspace management."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from competitive_intelligence.application.interfaces import (
    BaseUserRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.application.models import User, Workspace
from competitive_intelligence.infrastructure.database.models import UserOrm, WorkspaceOrm


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlUserRepository(BaseUserRepository):
    """Repository managing User entity persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, user: User) -> User:
        """Persist a new User entity."""
        orm_user = UserOrm(
            user_id=user.user_id,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.created_at,
        )
        self._session.add(orm_user)
        await self._session.flush()
        return user

    async def get_by_id(self, user_id: str) -> User | None:
        """Fetch a User by primary user_id."""
        stmt = select(UserOrm).where(UserOrm.user_id == user_id)
        result = await self._session.execute(stmt)
        orm_user = result.scalar_one_or_none()
        if orm_user is None:
            return None
        created_at = _ensure_utc(orm_user.created_at)
        assert created_at is not None
        return User(
            user_id=orm_user.user_id,
            email=orm_user.email,
            full_name=orm_user.full_name,
            is_active=orm_user.is_active,
            created_at=created_at,
        )

    async def get_by_email(self, email: str) -> User | None:
        """Fetch a User by unique email."""
        stmt = select(UserOrm).where(UserOrm.email == email)
        result = await self._session.execute(stmt)
        orm_user = result.scalar_one_or_none()
        if orm_user is None:
            return None
        created_at = _ensure_utc(orm_user.created_at)
        assert created_at is not None
        return User(
            user_id=orm_user.user_id,
            email=orm_user.email,
            full_name=orm_user.full_name,
            is_active=orm_user.is_active,
            created_at=created_at,
        )


class SqlWorkspaceRepository(BaseWorkspaceRepository):
    """Repository managing Workspace boundaries."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, workspace: Workspace) -> Workspace:
        """Persist a new Workspace boundary."""
        orm_ws = WorkspaceOrm(
            workspace_id=workspace.workspace_id,
            name=workspace.name,
            owner_user_id=workspace.owner_user_id,
            created_at=workspace.created_at,
            updated_at=workspace.created_at,
        )
        self._session.add(orm_ws)
        await self._session.flush()
        return workspace

    async def get_by_id(self, workspace_id: str) -> Workspace | None:
        """Fetch a Workspace by primary workspace_id."""
        stmt = select(WorkspaceOrm).where(WorkspaceOrm.workspace_id == workspace_id)
        result = await self._session.execute(stmt)
        orm_ws = result.scalar_one_or_none()
        if orm_ws is None:
            return None
        created_at = _ensure_utc(orm_ws.created_at)
        assert created_at is not None
        return Workspace(
            workspace_id=orm_ws.workspace_id,
            name=orm_ws.name,
            owner_user_id=orm_ws.owner_user_id,
            created_at=created_at,
        )

    async def list_by_owner(self, owner_user_id: str) -> list[Workspace]:
        """List all Workspaces owned by a given user."""
        stmt = (
            select(WorkspaceOrm)
            .where(WorkspaceOrm.owner_user_id == owner_user_id)
            .order_by(WorkspaceOrm.created_at.asc())
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        workspaces: list[Workspace] = []
        for r in rows:
            created_at = _ensure_utc(r.created_at)
            assert created_at is not None
            workspaces.append(
                Workspace(
                    workspace_id=r.workspace_id,
                    name=r.name,
                    owner_user_id=r.owner_user_id,
                    created_at=created_at,
                )
            )
        return workspaces

    async def list_by_user(self, owner_user_id: str) -> list[Workspace]:
        """Alias for list_by_owner conforming to BaseWorkspaceRepository."""
        return await self.list_by_owner(owner_user_id)

    async def list_all(self, limit: int = 100) -> list[Workspace]:
        """List all workspaces in repository."""
        stmt = (
            select(WorkspaceOrm)
            .order_by(WorkspaceOrm.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        workspaces: list[Workspace] = []
        for r in rows:
            created_at = _ensure_utc(r.created_at)
            assert created_at is not None
            workspaces.append(
                Workspace(
                    workspace_id=r.workspace_id,
                    name=r.name,
                    owner_user_id=r.owner_user_id,
                    created_at=created_at,
                )
            )
        return workspaces
