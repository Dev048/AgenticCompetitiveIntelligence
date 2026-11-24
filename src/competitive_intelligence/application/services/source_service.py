"""Application use-case service for Monitored Sources."""

from competitive_intelligence.application.interfaces import (
    BaseCompetitorRepository,
    BaseSourceRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.core.errors import (
    DuplicateEntityError,
    EntityNotFoundError,
    ValidationError,
    WorkspaceIsolationError,
)
from competitive_intelligence.source_acquisition.models import (
    AcquisitionConfig,
    MonitoringPolicy,
    Source,
    SourceCategory,
    SourceClassification,
    SourceGovernance,
    SourceIdentity,
    SourceLifecycle,
    SourceLocation,
    SourceReliability,
)


class SourceService:
    """Coordinates monitored source provisioning, lifecycle, and workspace isolation."""

    def __init__(
        self,
        source_repo: BaseSourceRepository,
        competitor_repo: BaseCompetitorRepository | None = None,
        workspace_repo: BaseWorkspaceRepository | None = None,
    ) -> None:
        self._source_repo = source_repo
        self._competitor_repo = competitor_repo
        self._workspace_repo = workspace_repo

    async def register_source(
        self,
        workspace_id: str,
        company_id: str,
        display_name: str,
        url_or_identifier: str,
        category: SourceCategory = SourceCategory.NEWS,
        acquisition: AcquisitionConfig | None = None,
        monitoring: MonitoringPolicy | None = None,
        reliability: SourceReliability | None = None,
        governance: SourceGovernance | None = None,
    ) -> Source:
        """Register a new source for active monitoring."""
        if not display_name or not display_name.strip():
            raise ValidationError("Source display name cannot be empty")
        if not url_or_identifier or not url_or_identifier.strip():
            raise ValidationError("Source URL or identifier cannot be empty")

        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        if self._competitor_repo is not None:
            comp = await self._competitor_repo.get_by_id(company_id)
            if comp is None:
                raise EntityNotFoundError(f"Competitor '{company_id}' not found")
            if comp.workspace_id != workspace_id:
                raise WorkspaceIsolationError(
                    f"Competitor '{company_id}' does not belong to workspace '{workspace_id}'"
                )

        source = Source(
            identity=SourceIdentity(
                company_id=company_id,
                workspace_id=workspace_id,
                display_name=display_name.strip(),
            ),
            classification=SourceClassification(category=category),
            location=SourceLocation(url_or_identifier=url_or_identifier.strip()),
            acquisition=acquisition or AcquisitionConfig(),
            monitoring=monitoring or MonitoringPolicy(),
            reliability=reliability or SourceReliability(),
            governance=governance or SourceGovernance(),
            lifecycle=SourceLifecycle(),
        )

        try:
            return await self._source_repo.create(source)
        except Exception as err:
            raise DuplicateEntityError(
                f"Source for '{url_or_identifier}' already registered for competitor '{company_id}'"
            ) from err

    async def get_source(self, workspace_id: str, source_id: str) -> Source:
        """Get source by ID and enforce workspace isolation."""
        src = await self._source_repo.get_by_id(source_id)
        if src is None:
            raise EntityNotFoundError(f"Source '{source_id}' not found")
        if src.identity.workspace_id != workspace_id:
            raise WorkspaceIsolationError(
                f"Source '{source_id}' does not belong to workspace '{workspace_id}'"
            )
        return src

    async def list_sources(
        self,
        workspace_id: str,
        company_id: str | None = None,
        category: SourceCategory | None = None,
    ) -> list[Source]:
        """List sources in workspace with optional competitor and category filtering."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        return await self._source_repo.list_by_workspace(
            workspace_id=workspace_id, company_id=company_id, category=category
        )

    async def update_source_monitoring(
        self,
        workspace_id: str,
        source_id: str,
        monitoring_enabled: bool | None = None,
        frequency_seconds: int | None = None,
        priority: int | None = None,
    ) -> Source:
        """Update monitoring policy with workspace verification."""
        source = await self.get_source(workspace_id, source_id)

        cur_mon = source.monitoring
        updated_mon = MonitoringPolicy(
            enabled=monitoring_enabled
            if monitoring_enabled is not None
            else cur_mon.enabled,
            frequency=frequency_seconds
            if frequency_seconds is not None
            else cur_mon.frequency,
            priority=priority if priority is not None else cur_mon.priority,
            volatility_score=cur_mon.volatility_score,
            last_checked_at=cur_mon.last_checked_at,
            last_changed_at=cur_mon.last_changed_at,
        )

        updated_source = Source(
            identity=source.identity,
            classification=source.classification,
            location=source.location,
            acquisition=source.acquisition,
            monitoring=updated_mon,
            reliability=source.reliability,
            governance=source.governance,
            lifecycle=source.lifecycle,
        )
        return await self._source_repo.update(updated_source)
