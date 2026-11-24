"""Application use-case service for Grounded Intelligence Graph queries."""

from competitive_intelligence.application.interfaces import (
    BaseIntelligenceRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.core.errors import (
    EntityNotFoundError,
    WorkspaceIsolationError,
)
from competitive_intelligence.intelligence.models import (
    EvidenceReference,
    InferredSignal,
    Insight,
    IntelligenceEvent,
)


class IntelligenceService:
    """Manages queries across events, signals, and insights with strict workspace isolation."""

    def __init__(
        self,
        intelligence_repo: BaseIntelligenceRepository,
        workspace_repo: BaseWorkspaceRepository | None = None,
    ) -> None:
        self._intelligence_repo = intelligence_repo
        self._workspace_repo = workspace_repo

    async def list_events(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 100,
    ) -> list[IntelligenceEvent]:
        """List intelligence events in workspace."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        return await self._intelligence_repo.list_events(
            workspace_id=workspace_id, company_id=company_id, limit=limit
        )

    async def get_event(self, workspace_id: str, event_id: str) -> IntelligenceEvent:
        """Get event by ID with workspace verification."""
        event = await self._intelligence_repo.get_event_by_id(event_id)
        if event is None:
            raise EntityNotFoundError(f"Event '{event_id}' not found")
        if event.workspace_id != workspace_id:
            raise WorkspaceIsolationError(
                f"Event '{event_id}' does not belong to workspace '{workspace_id}'"
            )
        return event

    async def list_signals(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 100,
    ) -> list[InferredSignal]:
        """List inferred signals in workspace."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        return await self._intelligence_repo.list_signals(
            workspace_id=workspace_id, company_id=company_id, limit=limit
        )

    async def get_signal(self, workspace_id: str, signal_id: str) -> InferredSignal:
        """Get signal by ID with workspace verification."""
        signal = await self._intelligence_repo.get_signal_by_id(signal_id)
        if signal is None:
            raise EntityNotFoundError(f"Signal '{signal_id}' not found")
        if signal.workspace_id != workspace_id:
            raise WorkspaceIsolationError(
                f"Signal '{signal_id}' does not belong to workspace '{workspace_id}'"
            )
        return signal

    async def list_insights(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 100,
    ) -> list[Insight]:
        """List strategic insights in workspace."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        return await self._intelligence_repo.list_insights(
            workspace_id=workspace_id, company_id=company_id, limit=limit
        )

    async def get_insight(self, workspace_id: str, insight_id: str) -> Insight:
        """Get insight by ID with workspace verification."""
        insight = await self._intelligence_repo.get_insight_by_id(insight_id)
        if insight is None:
            raise EntityNotFoundError(f"Insight '{insight_id}' not found")
        if insight.workspace_id != workspace_id:
            raise WorkspaceIsolationError(
                f"Insight '{insight_id}' does not belong to workspace '{workspace_id}'"
            )
        return insight
