"""Abstract interfaces for Application repositories and services."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from competitive_intelligence.application.models import (
    TrackedCompetitor,
    User,
    Workspace,
)
from competitive_intelligence.decision_support.models import (
    DecisionContext,
    DecisionOption,
    Recommendation,
    RecommendationStatus,
)
from competitive_intelligence.intelligence.models import (
    EvidenceReference,
    InferredSignal,
    Insight,
    IntelligenceEvent,
)
from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseDocumentRepository as BaseDocumentRepository,
)
from competitive_intelligence.knowledge_pipeline.models import (
    Document,
    DocumentChunk,
    DocumentVersion,
)
from competitive_intelligence.source_acquisition.models import Source, SourceCategory


class BaseUserRepository(ABC):
    """Abstract repository for User accounts."""

    @abstractmethod
    async def create(self, user: User) -> User:
        """Create new user account."""
        pass

    @abstractmethod
    async def get_by_id(self, user_id: str) -> User | None:
        """Fetch user by ID."""
        pass

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None:
        """Fetch user by email."""
        pass


class BaseWorkspaceRepository(ABC):
    """Abstract repository for Workspaces."""

    @abstractmethod
    async def create(self, workspace: Workspace) -> Workspace:
        """Create new workspace."""
        pass

    @abstractmethod
    async def get_by_id(self, workspace_id: str) -> Workspace | None:
        """Fetch workspace by ID."""
        pass

    @abstractmethod
    async def list_by_user(self, owner_user_id: str) -> list[Workspace]:
        """List workspaces owned by user."""
        pass

    @abstractmethod
    async def list_all(self, limit: int = 100) -> list[Workspace]:
        """List all workspaces."""
        pass


class BaseCompetitorRepository(ABC):
    """Abstract repository for Tracked Competitors."""

    @abstractmethod
    async def create(self, competitor: TrackedCompetitor) -> TrackedCompetitor:
        """Register tracked competitor."""
        pass

    @abstractmethod
    async def get_by_id(self, competitor_id: str) -> TrackedCompetitor | None:
        """Fetch competitor by ID."""
        pass

    @abstractmethod
    async def list_by_workspace(
        self, workspace_id: str, is_active: bool | None = None
    ) -> list[TrackedCompetitor]:
        """List competitors in workspace."""
        pass

    @abstractmethod
    async def update(self, competitor: TrackedCompetitor) -> TrackedCompetitor:
        """Update competitor details."""
        pass


class BaseSourceRepository(ABC):
    """Abstract repository for Monitored Sources."""

    @abstractmethod
    async def create(self, source: Source) -> Source:
        """Register monitored source."""
        pass

    @abstractmethod
    async def get_by_id(self, source_id: str) -> Source | None:
        """Fetch source by ID."""
        pass

    @abstractmethod
    async def list_by_workspace(
        self,
        workspace_id: str,
        company_id: str | None = None,
        category: SourceCategory | None = None,
    ) -> list[Source]:
        """List sources in workspace."""
        pass

    @abstractmethod
    async def update(self, source: Source) -> Source:
        """Update source configuration."""
        pass

    @abstractmethod
    async def get_due_sources(
        self,
        limit: int = 50,
        as_of: datetime | None = None,
        workspace_id: str | None = None,
    ) -> list[Source]:
        """Retrieve sources due for monitoring whose frequency interval has elapsed."""
        pass

    @abstractmethod
    async def update_fetch_status(
        self,
        source_id: str,
        last_fetch_status: str,
        consecutive_failure_count: int,
        last_checked_at: datetime | None = None,
        last_changed_at: datetime | None = None,
    ) -> bool:
        """Update source fetch and monitoring telemetry."""
        pass


from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseDocumentRepository,
)


class BaseIntelligenceRepository(ABC):
    """Abstract repository for Grounded Intelligence Graph entities."""

    @abstractmethod
    async def create_evidence(self, evidence: EvidenceReference) -> EvidenceReference:
        """Persist evidence reference."""
        pass

    @abstractmethod
    async def get_evidence_by_id(self, evidence_id: str) -> EvidenceReference | None:
        """Fetch evidence reference by ID."""
        pass

    @abstractmethod
    async def list_evidence_for_document(self, document_id: str) -> list[EvidenceReference]:
        """List evidence references derived from a document."""
        pass

    @abstractmethod
    async def create_event(self, event: IntelligenceEvent) -> IntelligenceEvent:
        """Persist intelligence event."""
        pass

    @abstractmethod
    async def get_event_by_id(self, event_id: str) -> IntelligenceEvent | None:
        """Fetch intelligence event by ID."""
        pass

    @abstractmethod
    async def list_events(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 100,
    ) -> list[IntelligenceEvent]:
        """List intelligence events in workspace."""
        pass

    @abstractmethod
    async def create_signal(self, signal: InferredSignal) -> InferredSignal:
        """Persist inferred signal."""
        pass

    @abstractmethod
    async def get_signal_by_id(self, signal_id: str) -> InferredSignal | None:
        """Fetch inferred signal by ID."""
        pass

    @abstractmethod
    async def list_signals(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 100,
    ) -> list[InferredSignal]:
        """List inferred signals in workspace."""
        pass

    @abstractmethod
    async def create_insight(self, insight: Insight) -> Insight:
        """Persist synthesized insight."""
        pass

    @abstractmethod
    async def get_insight_by_id(self, insight_id: str) -> Insight | None:
        """Fetch insight by ID."""
        pass

    @abstractmethod
    async def list_insights(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 100,
    ) -> list[Insight]:
        """List insights in workspace."""
        pass


class BaseDecisionRepository(ABC):
    """Abstract repository for Decision Contexts, Options, and Recommendations."""

    @abstractmethod
    async def create_context(self, context: DecisionContext) -> DecisionContext:
        """Persist decision context."""
        pass

    @abstractmethod
    async def get_context_by_id(self, context_id: str) -> DecisionContext | None:
        """Fetch decision context by ID."""
        pass

    @abstractmethod
    async def create_option(self, option: DecisionOption) -> DecisionOption:
        """Persist candidate decision option."""
        pass

    @abstractmethod
    async def get_option_by_id(self, option_id: str) -> DecisionOption | None:
        """Fetch decision option by ID."""
        pass

    @abstractmethod
    async def list_options_for_context(self, context_id: str) -> list[DecisionOption]:
        """List candidate options evaluated under a context."""
        pass

    @abstractmethod
    async def create_recommendation(self, recommendation: Recommendation) -> Recommendation:
        """Persist actionable recommendation."""
        pass

    @abstractmethod
    async def get_recommendation_by_id(self, recommendation_id: str) -> Recommendation | None:
        """Fetch recommendation by ID."""
        pass

    @abstractmethod
    async def list_recommendations_for_context(
        self, context_id: str
    ) -> list[Recommendation]:
        """List recommendations for a decision context."""
        pass

    @abstractmethod
    async def reconstruct_recommendation_provenance(
        self, recommendation_id: str
    ) -> dict[str, Any]:
        """Reconstruct the complete 10-layer grounded provenance tree."""
        pass

    @abstractmethod
    async def list_contexts_by_workspace(
        self, workspace_id: str
    ) -> list[DecisionContext]:
        """List all decision contexts in a workspace."""
        pass

    @abstractmethod
    async def list_recommendations_by_workspace(
        self, workspace_id: str
    ) -> list[Recommendation]:
        """List all recommendations in a workspace."""
        pass

    @abstractmethod
    async def update_recommendation_status(
        self, recommendation_id: str, status: RecommendationStatus
    ) -> Recommendation:
        """Update recommendation status (e.g. accepted/rejected by human review)."""
        pass
