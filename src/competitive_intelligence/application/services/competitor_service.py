"""Application use-case service for Tracked Competitors."""

from competitive_intelligence.application.interfaces import (
    BaseCompetitorRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.application.models import TrackedCompetitor
from competitive_intelligence.core.errors import (
    DuplicateEntityError,
    EntityNotFoundError,
    ValidationError,
    WorkspaceIsolationError,
)


class CompetitorService:
    """Manages competitor lifecycle and workspace isolation."""

    def __init__(
        self,
        competitor_repo: BaseCompetitorRepository,
        workspace_repo: BaseWorkspaceRepository | None = None,
    ) -> None:
        self._competitor_repo = competitor_repo
        self._workspace_repo = workspace_repo

    async def register_competitor(
        self,
        workspace_id: str,
        name: str,
        domain: str,
        description: str | None = None,
    ) -> TrackedCompetitor:
        """Register a new competitor to track in a workspace."""
        if not name or not name.strip():
            raise ValidationError("Competitor name cannot be empty")
        if not domain or not domain.strip():
            raise ValidationError("Competitor domain cannot be empty")

        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        competitor = TrackedCompetitor(
            workspace_id=workspace_id,
            name=name.strip(),
            domain=domain.strip().lower(),
            description=description.strip() if description else None,
            is_active=True,
        )
        try:
            return await self._competitor_repo.create(competitor)
        except Exception as err:
            raise DuplicateEntityError(
                f"Competitor with domain '{domain}' already exists in workspace '{workspace_id}'"
            ) from err

    async def get_competitor(self, workspace_id: str, competitor_id: str) -> TrackedCompetitor:
        """Get competitor by ID and enforce workspace isolation."""
        comp = await self._competitor_repo.get_by_id(competitor_id)
        if comp is None:
            raise EntityNotFoundError(f"Competitor '{competitor_id}' not found")
        if comp.workspace_id != workspace_id:
            raise WorkspaceIsolationError(
                f"Competitor '{competitor_id}' does not belong to workspace '{workspace_id}'"
            )
        return comp

    async def list_competitors(
        self, workspace_id: str, is_active: bool | None = None
    ) -> list[TrackedCompetitor]:
        """List competitors in workspace."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        return await self._competitor_repo.list_by_workspace(
            workspace_id=workspace_id, is_active=is_active
        )

    async def update_competitor(
        self,
        workspace_id: str,
        competitor_id: str,
        name: str | None = None,
        description: str | None = None,
        is_active: bool | None = None,
    ) -> TrackedCompetitor:
        """Update competitor details with workspace boundary verification."""
        comp = await self.get_competitor(workspace_id, competitor_id)

        updated_name = name.strip() if name is not None else comp.name
        updated_desc = description.strip() if description is not None else comp.description
        updated_active = is_active if is_active is not None else comp.is_active

        updated_comp = TrackedCompetitor(
            competitor_id=comp.competitor_id,
            workspace_id=comp.workspace_id,
            name=updated_name,
            domain=comp.domain,
            description=updated_desc,
            is_active=updated_active,
            created_at=comp.created_at,
        )
        return await self._competitor_repo.update(updated_comp)
