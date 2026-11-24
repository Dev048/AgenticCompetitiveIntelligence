"""SQL repository for TrackedCompetitor management."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from competitive_intelligence.application.interfaces import BaseCompetitorRepository
from competitive_intelligence.application.models import TrackedCompetitor
from competitive_intelligence.infrastructure.database.models import TrackedCompetitorOrm


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlCompetitorRepository(BaseCompetitorRepository):
    """Repository managing TrackedCompetitor entity persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, competitor: TrackedCompetitor) -> TrackedCompetitor:
        """Persist a new TrackedCompetitor entity."""
        orm_comp = TrackedCompetitorOrm(
            competitor_id=competitor.competitor_id,
            workspace_id=competitor.workspace_id,
            name=competitor.name,
            domain=competitor.domain,
            description=competitor.description,
            is_active=competitor.is_active,
            created_at=competitor.created_at,
            updated_at=competitor.created_at,
        )
        self._session.add(orm_comp)
        await self._session.flush()
        return competitor

    async def get_by_id(self, competitor_id: str) -> TrackedCompetitor | None:
        """Fetch a TrackedCompetitor by primary competitor_id."""
        stmt = select(TrackedCompetitorOrm).where(TrackedCompetitorOrm.competitor_id == competitor_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        created_at = _ensure_utc(row.created_at)
        assert created_at is not None
        return TrackedCompetitor(
            competitor_id=row.competitor_id,
            workspace_id=row.workspace_id,
            name=row.name,
            domain=row.domain,
            description=row.description,
            is_active=row.is_active,
            created_at=created_at,
        )

    async def list_by_workspace(
        self,
        workspace_id: str,
        is_active: bool | None = None,
        is_active_only: bool | None = None,
    ) -> list[TrackedCompetitor]:
        """List all competitors in a workspace."""
        stmt = select(TrackedCompetitorOrm).where(
            TrackedCompetitorOrm.workspace_id == workspace_id
        )
        if is_active is not None:
            stmt = stmt.where(TrackedCompetitorOrm.is_active == is_active)
        elif is_active_only is True:
            stmt = stmt.where(TrackedCompetitorOrm.is_active == True)  # noqa: E712
        stmt = stmt.order_by(TrackedCompetitorOrm.name.asc())

        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        competitors: list[TrackedCompetitor] = []
        for r in rows:
            created_at = _ensure_utc(r.created_at)
            assert created_at is not None
            competitors.append(
                TrackedCompetitor(
                    competitor_id=r.competitor_id,
                    workspace_id=r.workspace_id,
                    name=r.name,
                    domain=r.domain,
                    description=r.description,
                    is_active=r.is_active,
                    created_at=created_at,
                )
            )
        return competitors

    async def update(self, competitor: TrackedCompetitor) -> TrackedCompetitor:
        """Update an existing TrackedCompetitor entity."""
        stmt = select(TrackedCompetitorOrm).where(
            TrackedCompetitorOrm.competitor_id == competitor.competitor_id
        )
        result = await self._session.execute(stmt)
        orm_comp = result.scalar_one_or_none()
        if orm_comp is not None:
            orm_comp.name = competitor.name
            orm_comp.description = competitor.description
            orm_comp.is_active = competitor.is_active
            orm_comp.updated_at = datetime.now(timezone.utc)
            await self._session.flush()
        return competitor
