"""SQL repository for Source registry management and lifecycle persistence."""

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from competitive_intelligence.application.interfaces import BaseSourceRepository
from competitive_intelligence.infrastructure.database.models import (
    SourceMonitoringLeaseOrm,
    SourceOrm,
)
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    ContentType,
    DataSensitivity,
    FetchStatus,
    LifecycleStatus,
    MonitoringPolicy,
    PlatformType,
    RateLimitConstraints,
    Source,
    SourceCategory,
    SourceClassification,
    SourceGovernance,
    SourceIdentity,
    SourceLifecycle,
    SourceLocation,
    SourceReliability,
    SourceTier,
    ToolChain,
    ToSStatus,
)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _source_to_orm(source: Source) -> SourceOrm:
    """Map pure domain Source model to SQLAlchemy SourceOrm."""
    tool_chain_dict: dict[str, Any] | None = None
    if source.acquisition.tool_chain:
        tool_chain_dict = {
            "primary": source.acquisition.tool_chain.primary.value,
            "fallbacks": [f.value for f in source.acquisition.tool_chain.fallbacks],
        }

    rate_limit_dict = {
        "requests_per_minute": source.acquisition.rate_limit_constraints.requests_per_minute,
        "min_interval_seconds": source.acquisition.rate_limit_constraints.min_interval_seconds,
        "max_concurrent_requests": source.acquisition.rate_limit_constraints.max_concurrent_requests,
    }

    return SourceOrm(
        source_id=source.source_id,
        workspace_id=source.workspace_id,
        company_id=source.company_id,
        display_name=source.display_name,
        category=source.classification.category.value,
        platform=source.classification.platform.value,
        content_type=source.classification.content_type.value,
        url_or_identifier=source.location.url_or_identifier,
        access_pattern=source.location.access_pattern.value,
        requires_javascript=source.location.requires_javascript,
        acquisition_method=source.acquisition.acquisition_method.value,
        tool_chain=tool_chain_dict,
        requires_authentication=source.acquisition.requires_authentication,
        rate_limit_constraints=rate_limit_dict,
        last_fetch_status=source.acquisition.last_fetch_status.value,
        consecutive_failure_count=source.acquisition.consecutive_failure_count,
        monitoring_enabled=source.monitoring.enabled,
        frequency_seconds=source.monitoring.frequency,
        priority=source.monitoring.priority,
        volatility_score=source.monitoring.volatility_score,
        last_checked_at=source.monitoring.last_checked_at,
        last_changed_at=source.monitoring.last_changed_at,
        credibility_score=source.reliability.credibility_score,
        source_tier=source.reliability.source_tier.value,
        historical_accuracy=source.reliability.historical_accuracy,
        terms_of_service_status=source.governance.terms_of_service_status.value,
        robots_txt_respected=source.governance.robots_txt_respected,
        data_sensitivity=source.governance.data_sensitivity.value,
        lifecycle_status=source.lifecycle.status.value,
        replaced_by_source_id=source.lifecycle.replaced_by_source_id,
        created_at=source.lifecycle.created_at,
        updated_at=source.lifecycle.updated_at,
    )


def _orm_to_source(orm: SourceOrm) -> Source:
    """Map SQLAlchemy SourceOrm to pure domain Source model."""
    tool_chain: ToolChain | None = None
    if orm.tool_chain:
        tool_chain = ToolChain(
            primary=AcquisitionMethod(orm.tool_chain["primary"]),
            fallbacks=[AcquisitionMethod(f) for f in orm.tool_chain.get("fallbacks", [])],
        )

    rate_limits = RateLimitConstraints(
        requests_per_minute=orm.rate_limit_constraints.get("requests_per_minute", 10),
        min_interval_seconds=orm.rate_limit_constraints.get("min_interval_seconds", 0.0),
        max_concurrent_requests=orm.rate_limit_constraints.get("max_concurrent_requests", 1),
    )

    created_at = _ensure_utc(orm.created_at)
    assert created_at is not None
    updated_at = _ensure_utc(orm.updated_at)
    assert updated_at is not None

    return Source(
        identity=SourceIdentity(
            source_id=orm.source_id,
            company_id=orm.company_id,
            workspace_id=orm.workspace_id,
            display_name=orm.display_name,
        ),
        classification=SourceClassification(
            category=SourceCategory(orm.category),
            platform=PlatformType(orm.platform),
            content_type=ContentType(orm.content_type),
        ),
        location=SourceLocation(
            url_or_identifier=orm.url_or_identifier,
            access_pattern=AccessPattern(orm.access_pattern),
            requires_javascript=orm.requires_javascript,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod(orm.acquisition_method),
            tool_chain=tool_chain,
            requires_authentication=orm.requires_authentication,
            rate_limit_constraints=rate_limits,
            last_fetch_status=FetchStatus(orm.last_fetch_status),
            consecutive_failure_count=orm.consecutive_failure_count,
        ),
        monitoring=MonitoringPolicy(
            enabled=orm.monitoring_enabled,
            frequency=orm.frequency_seconds,
            priority=orm.priority,
            volatility_score=orm.volatility_score,
            last_checked_at=_ensure_utc(orm.last_checked_at),
            last_changed_at=_ensure_utc(orm.last_changed_at),
        ),
        reliability=SourceReliability(
            credibility_score=orm.credibility_score,
            source_tier=SourceTier(orm.source_tier),
            historical_accuracy=orm.historical_accuracy,
        ),
        governance=SourceGovernance(
            terms_of_service_status=ToSStatus(orm.terms_of_service_status),
            robots_txt_respected=orm.robots_txt_respected,
            data_sensitivity=DataSensitivity(orm.data_sensitivity),
        ),
        lifecycle=SourceLifecycle(
            created_at=created_at,
            updated_at=updated_at,
            status=LifecycleStatus(orm.lifecycle_status),
            replaced_by_source_id=orm.replaced_by_source_id,
        ),
    )


class SqlSourceRepository(BaseSourceRepository):
    """Repository managing Source registry and lifecycle persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, source: Source) -> Source:
        """Persist a new monitored Source."""
        orm_source = _source_to_orm(source)
        self._session.add(orm_source)
        await self._session.flush()
        return source

    async def get_by_id(self, source_id: str) -> Source | None:
        """Fetch a Source by primary source_id."""
        stmt = select(SourceOrm).where(SourceOrm.source_id == source_id)
        result = await self._session.execute(stmt)
        orm_source = result.scalar_one_or_none()
        if orm_source is None:
            return None
        return _orm_to_source(orm_source)

    async def list_by_company(
        self, company_id: str, workspace_id: str | None = None
    ) -> list[Source]:
        """List all sources monitored for a tracked competitor."""
        stmt = select(SourceOrm).where(SourceOrm.company_id == company_id)
        if workspace_id:
            stmt = stmt.where(SourceOrm.workspace_id == workspace_id)
        stmt = stmt.order_by(SourceOrm.created_at.asc())

        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [_orm_to_source(r) for r in rows]

    async def list_by_workspace(
        self,
        workspace_id: str,
        company_id: str | None = None,
        category: SourceCategory | None = None,
    ) -> list[Source]:
        """List sources in a workspace with optional filters."""
        stmt = select(SourceOrm).where(SourceOrm.workspace_id == workspace_id)
        if company_id:
            stmt = stmt.where(SourceOrm.company_id == company_id)
        if category:
            stmt = stmt.where(SourceOrm.category == category.value)
        stmt = stmt.order_by(SourceOrm.created_at.asc())
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [_orm_to_source(r) for r in rows]

    async def update(self, source: Source) -> Source:
        """Update an existing Source's configuration and monitoring policy."""
        stmt = select(SourceOrm).where(SourceOrm.source_id == source.source_id)
        result = await self._session.execute(stmt)
        orm_source = result.scalar_one_or_none()
        if orm_source is not None:
            orm_source.display_name = source.display_name
            orm_source.monitoring_enabled = source.monitoring.enabled
            orm_source.frequency_seconds = source.monitoring.frequency
            orm_source.priority = source.monitoring.priority
            orm_source.volatility_score = source.monitoring.volatility_score
            orm_source.updated_at = datetime.now(timezone.utc)
            await self._session.flush()
        return source

    async def get_due_sources(
        self,
        limit: int = 50,
        as_of: datetime | None = None,
        workspace_id: str | None = None,
    ) -> list[Source]:
        """Retrieve sources due for monitoring whose frequency interval has elapsed and are not actively leased."""
        ref_time = as_of or datetime.now(timezone.utc)

        stmt = (
            select(SourceOrm)
            .outerjoin(
                SourceMonitoringLeaseOrm,
                SourceOrm.source_id == SourceMonitoringLeaseOrm.source_id,
            )
            .where(
                SourceOrm.monitoring_enabled.is_(True),
                SourceOrm.lifecycle_status == "active",
                (SourceMonitoringLeaseOrm.source_id.is_(None))
                | (SourceMonitoringLeaseOrm.lease_expires_at <= ref_time),
            )
        )

        if workspace_id:
            stmt = stmt.where(SourceOrm.workspace_id == workspace_id)

        stmt = stmt.order_by(
            SourceOrm.priority.desc(),
            SourceOrm.last_checked_at.asc().nullsfirst(),
        ).limit(limit)

        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        due_sources: list[Source] = []
        for r in rows:
            if r.last_checked_at is None:
                due_sources.append(_orm_to_source(r))
            else:
                last_chk = _ensure_utc(r.last_checked_at)
                assert last_chk is not None
                if last_chk + timedelta(seconds=r.frequency_seconds) <= ref_time:
                    due_sources.append(_orm_to_source(r))

        return due_sources

    async def update_fetch_status(
        self,
        source_id: str,
        last_fetch_status: str | FetchStatus,
        consecutive_failure_count: int,
        last_checked_at: datetime | None = None,
        last_changed_at: datetime | None = None,
    ) -> bool:
        """Update source fetch telemetry after an acquisition attempt."""
        stmt = select(SourceOrm).where(SourceOrm.source_id == source_id)
        result = await self._session.execute(stmt)
        orm_source = result.scalar_one_or_none()
        if orm_source is None:
            return False

        status_val = (
            last_fetch_status.value
            if isinstance(last_fetch_status, FetchStatus)
            else str(last_fetch_status)
        )
        orm_source.last_fetch_status = status_val
        orm_source.consecutive_failure_count = consecutive_failure_count
        if last_checked_at is not None:
            orm_source.last_checked_at = last_checked_at
        if last_changed_at is not None:
            orm_source.last_changed_at = last_changed_at
        orm_source.updated_at = datetime.now(timezone.utc)
        await self._session.flush()
        return True
