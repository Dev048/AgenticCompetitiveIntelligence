"""Application use-case service for Decision Support and Prescriptions."""

from typing import Any

from competitive_intelligence.application.interfaces import (
    BaseDecisionRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.core.errors import (
    EntityNotFoundError,
    ValidationError,
    WorkspaceIsolationError,
)
from competitive_intelligence.decision_support.models import (
    DecisionContext,
    DecisionOption,
    Recommendation,
    RecommendationStatus,
)


class DecisionService:
    """Coordinates decision context registration, candidate options, and 10-layer provenance."""

    def __init__(
        self,
        decision_repo: BaseDecisionRepository,
        workspace_repo: BaseWorkspaceRepository | None = None,
    ) -> None:
        self._decision_repo = decision_repo
        self._workspace_repo = workspace_repo

    async def create_decision_context(
        self, workspace_id: str, context: DecisionContext
    ) -> DecisionContext:
        """Register a new decision context within a workspace."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        # Guarantee workspace_id consistency
        if context.workspace_id != workspace_id:
            context = context.model_copy(update={"workspace_id": workspace_id})

        return await self._decision_repo.create_context(context)

    async def get_decision_context(
        self, workspace_id: str, context_id: str
    ) -> DecisionContext:
        """Fetch decision context by ID with workspace verification."""
        ctx = await self._decision_repo.get_context_by_id(context_id)
        if ctx is None:
            raise EntityNotFoundError(f"Decision context '{context_id}' not found")
        if ctx.workspace_id != workspace_id:
            raise WorkspaceIsolationError(
                f"Decision context '{context_id}' does not belong to workspace '{workspace_id}'"
            )
        return ctx

    async def list_options_for_context(
        self, workspace_id: str, context_id: str
    ) -> list[DecisionOption]:
        """List candidate decision options evaluated under context."""
        # Verify context belongs to workspace
        await self.get_decision_context(workspace_id, context_id)
        return await self._decision_repo.list_options_for_context(context_id)

    async def list_recommendations_for_context(
        self, workspace_id: str, context_id: str
    ) -> list[Recommendation]:
        """List recommendations generated for a decision context."""
        await self.get_decision_context(workspace_id, context_id)
        return await self._decision_repo.list_recommendations_for_context(context_id)

    async def get_recommendation(
        self, workspace_id: str, recommendation_id: str
    ) -> Recommendation:
        """Fetch recommendation by ID with workspace verification."""
        rec = await self._decision_repo.get_recommendation_by_id(recommendation_id)
        if rec is None:
            raise EntityNotFoundError(f"Recommendation '{recommendation_id}' not found")
        if rec.workspace_id != workspace_id:
            raise WorkspaceIsolationError(
                f"Recommendation '{recommendation_id}' does not belong to workspace '{workspace_id}'"
            )
        return rec

    async def get_recommendation_provenance(
        self, workspace_id: str, recommendation_id: str
    ) -> dict[str, Any]:
        """Reconstruct complete 10-layer grounded provenance tree."""
        # Verify recommendation exists and belongs to workspace
        await self.get_recommendation(workspace_id, recommendation_id)
        return await self._decision_repo.reconstruct_recommendation_provenance(recommendation_id)

    async def list_decision_contexts(self, workspace_id: str) -> list[DecisionContext]:
        """List all decision contexts belonging to a workspace."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")
        return await self._decision_repo.list_contexts_by_workspace(workspace_id)

    async def list_recommendations_by_workspace(
        self, workspace_id: str
    ) -> list[Recommendation]:
        """List all recommendations belonging to a workspace."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")
        return await self._decision_repo.list_recommendations_by_workspace(workspace_id)

    async def update_recommendation_status(
        self,
        workspace_id: str,
        recommendation_id: str,
        status: RecommendationStatus,
    ) -> Recommendation:
        """Update recommendation status with human review audit."""
        await self.get_recommendation(workspace_id, recommendation_id)
        return await self._decision_repo.update_recommendation_status(
            recommendation_id=recommendation_id, status=status
        )
