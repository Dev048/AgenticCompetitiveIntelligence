"""Application use-case service for Workspaces and Users."""

from competitive_intelligence.application.interfaces import (
    BaseUserRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.application.models import User, Workspace
from competitive_intelligence.core.errors import (
    DuplicateEntityError,
    EntityNotFoundError,
    ValidationError,
)


class WorkspaceService:
    """Coordinates workspace provisioning, user associations, and boundary verification."""

    def __init__(
        self,
        workspace_repo: BaseWorkspaceRepository,
        user_repo: BaseUserRepository | None = None,
    ) -> None:
        self._workspace_repo = workspace_repo
        self._user_repo = user_repo

    async def create_workspace(self, name: str, owner_user_id: str) -> Workspace:
        """Create a new workspace for an owner user."""
        if not name or not name.strip():
            raise ValidationError("Workspace name cannot be empty")
        if not owner_user_id or not owner_user_id.strip():
            raise ValidationError("Owner user ID cannot be empty")

        # Ensure owner user exists if user_repo is provided
        if self._user_repo is not None:
            user = await self._user_repo.get_by_id(owner_user_id)
            if user is None:
                # Auto-provision user if not found
                user = User(user_id=owner_user_id, email=f"{owner_user_id}@workspace.local")
                try:
                    await self._user_repo.create(user)
                except Exception:
                    pass

        workspace = Workspace(name=name.strip(), owner_user_id=owner_user_id.strip())
        try:
            return await self._workspace_repo.create(workspace)
        except Exception as err:
            raise DuplicateEntityError(f"Workspace could not be created: {err}") from err

    async def get_workspace(self, workspace_id: str) -> Workspace:
        """Fetch workspace by ID or raise EntityNotFoundError."""
        ws = await self._workspace_repo.get_by_id(workspace_id)
        if ws is None:
            raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")
        return ws

    async def list_user_workspaces(self, owner_user_id: str | None = None) -> list[Workspace]:
        """List all workspaces, optionally filtered by owner user."""
        if owner_user_id is not None:
            return await self._workspace_repo.list_by_user(owner_user_id)
        return await self._workspace_repo.list_all()

    async def verify_workspace_exists(self, workspace_id: str) -> Workspace:
        """Guard method used by other services and dependencies to enforce workspace boundary."""
        return await self.get_workspace(workspace_id)
