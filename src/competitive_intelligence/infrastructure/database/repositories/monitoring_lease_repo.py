"""SQL repository for durable Source Monitoring Leases."""

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from competitive_intelligence.infrastructure.database.models import (
    SourceMonitoringLeaseOrm,
)
from competitive_intelligence.monitoring.interfaces import (
    BaseMonitoringLeaseRepository,
)
from competitive_intelligence.monitoring.models import SourceMonitoringLease


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _orm_to_lease(orm: SourceMonitoringLeaseOrm) -> SourceMonitoringLease:
    acq_at = _ensure_utc(orm.lease_acquired_at)
    exp_at = _ensure_utc(orm.lease_expires_at)
    hb_at = _ensure_utc(orm.heartbeat_at)
    cr_at = _ensure_utc(orm.created_at)
    up_at = _ensure_utc(orm.updated_at)
    assert acq_at is not None and exp_at is not None and hb_at is not None
    assert cr_at is not None and up_at is not None

    return SourceMonitoringLease(
        source_id=orm.source_id,
        lease_owner=orm.lease_owner,
        lease_acquired_at=acq_at,
        lease_expires_at=exp_at,
        heartbeat_at=hb_at,
        attempt_count=orm.attempt_count,
        created_at=cr_at,
        updated_at=up_at,
    )


class SqlMonitoringLeaseRepository(BaseMonitoringLeaseRepository):
    """PostgreSQL-backed implementation of BaseMonitoringLeaseRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def claim_due_source(
        self,
        source_id: str,
        lease_owner: str,
        lease_duration_seconds: int = 300,
    ) -> SourceMonitoringLease | None:
        """Atomically claim a lease on a source if unleased or expired.

        Uses PostgreSQL row-level lock (FOR UPDATE) within a short transaction boundary
        and nested savepoints to gracefully handle concurrent initial claim races.
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=lease_duration_seconds)

        # Query existing lease with FOR UPDATE lock
        stmt = (
            select(SourceMonitoringLeaseOrm)
            .where(SourceMonitoringLeaseOrm.source_id == source_id)
            .with_for_update()
        )
        result = await self._session.execute(stmt)
        existing_lease = result.scalar_one_or_none()

        if existing_lease is None:
            # Create brand new lease using nested savepoint to isolate concurrent insert collision
            try:
                async with self._session.begin_nested():
                    new_lease = SourceMonitoringLeaseOrm(
                        source_id=source_id,
                        lease_owner=lease_owner,
                        lease_acquired_at=now,
                        lease_expires_at=expires_at,
                        heartbeat_at=now,
                        attempt_count=1,
                        created_at=now,
                        updated_at=now,
                    )
                    self._session.add(new_lease)
                    await self._session.flush()
                    return _orm_to_lease(new_lease)
            except IntegrityError:
                # Concurrent worker created the lease row first.
                # Re-query under FOR UPDATE lock to evaluate active vs expired status
                stmt = (
                    select(SourceMonitoringLeaseOrm)
                    .where(SourceMonitoringLeaseOrm.source_id == source_id)
                    .with_for_update()
                )
                result = await self._session.execute(stmt)
                existing_lease = result.scalar_one_or_none()
                if existing_lease is None:
                    return None

        # If existing lease is still active under another owner, cannot claim
        existing_exp = _ensure_utc(existing_lease.lease_expires_at)
        assert existing_exp is not None
        if existing_exp > now and existing_lease.lease_owner != lease_owner:
            return None

        # Claim / take over expired lease or re-claim existing lease
        existing_lease.lease_owner = lease_owner
        existing_lease.lease_acquired_at = now
        existing_lease.lease_expires_at = expires_at
        existing_lease.heartbeat_at = now
        existing_lease.attempt_count += 1
        existing_lease.updated_at = now
        await self._session.flush()
        return _orm_to_lease(existing_lease)

    async def renew_source_lease(
        self,
        source_id: str,
        lease_owner: str,
        extension_seconds: int = 300,
    ) -> bool:
        """Extend the lease expiration for the active owner."""
        now = datetime.now(timezone.utc)
        new_expires_at = now + timedelta(seconds=extension_seconds)

        stmt = (
            update(SourceMonitoringLeaseOrm)
            .where(
                SourceMonitoringLeaseOrm.source_id == source_id,
                SourceMonitoringLeaseOrm.lease_owner == lease_owner,
                SourceMonitoringLeaseOrm.lease_expires_at > now,
            )
            .values(
                heartbeat_at=now,
                lease_expires_at=new_expires_at,
                updated_at=now,
            )
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        rowcount = int(getattr(result, "rowcount", 0))
        return rowcount > 0

    async def release_source_lease(
        self,
        source_id: str,
        lease_owner: str,
    ) -> bool:
        """Release the lease held by the owner."""
        stmt = delete(SourceMonitoringLeaseOrm).where(
            SourceMonitoringLeaseOrm.source_id == source_id,
            SourceMonitoringLeaseOrm.lease_owner == lease_owner,
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        rowcount = int(getattr(result, "rowcount", 0))
        return rowcount > 0

    async def get_active_lease(
        self,
        source_id: str,
    ) -> SourceMonitoringLease | None:
        """Retrieve active lease on source."""
        stmt = select(SourceMonitoringLeaseOrm).where(
            SourceMonitoringLeaseOrm.source_id == source_id
        )
        result = await self._session.execute(stmt)
        orm = result.scalar_one_or_none()
        if orm is None:
            return None
        return _orm_to_lease(orm)

    async def reclaim_expired_leases(
        self,
        as_of: datetime | None = None,
    ) -> int:
        """Clean up expired leases whose lease_expires_at has elapsed."""
        ref_time = as_of or datetime.now(timezone.utc)
        stmt = delete(SourceMonitoringLeaseOrm).where(
            SourceMonitoringLeaseOrm.lease_expires_at <= ref_time
        )
        result = await self._session.execute(stmt)
        await self._session.flush()
        return int(getattr(result, "rowcount", 0))
