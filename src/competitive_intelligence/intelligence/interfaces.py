"""Abstract interfaces for Competitive Intelligence subsystem.

Defines the contract for BaseIntelligenceAnalyzer and event sequence trackers,
transforming document version changes and retrieved search evidence into structured
IntelligenceEvents without exposing underlying databases, ML models, or LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Any

from competitive_intelligence.intelligence.models import (
    Alert,
    DetectedEvent,
    EventSequence,
    GeneratedInsight,
    InferredSignal,
    IntelligenceEvent,
    IntelligenceReport,
    RawEvidence,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChangeResult,
    DocumentVersion,
)
from competitive_intelligence.retrieval_engine.models import (
    RetrievalResult,
)


class BaseIntelligenceAnalyzer(ABC):
    """Abstract interface defining the contract for Intelligence Event Analyzers."""

    @abstractmethod
    async def analyze_document_change(
        self,
        change: ChangeResult,
        current_version: DocumentVersion,
        previous_version: DocumentVersion | None,
        context_evidence: list[RetrievalResult] | None = None,
    ) -> list[IntelligenceEvent]:
        """Analyze a discrete document version change and extract structured IntelligenceEvents.

        Args:
            change: Result of comparing current_version against previous_version.
            current_version: The latest immutable document version snapshot.
            previous_version: The prior version snapshot, if one exists.
            context_evidence: Optional supporting retrieval evidence chunks.

        Returns:
            List of structured, evidence-grounded IntelligenceEvent instances.
        """
        pass

    @abstractmethod
    async def analyze_retrieved_evidence(
        self,
        query: str,
        company_id: str,
        workspace_id: str,
        evidence: list[RetrievalResult],
    ) -> list[IntelligenceEvent]:
        """Analyze retrieved evidence chunks from search and synthesize structured IntelligenceEvents.

        Args:
            query: The research query or monitoring inquiry text.
            company_id: Identifier of the target company.
            workspace_id: Tenant / workspace boundary.
            evidence: List of scored retrieval results.

        Returns:
            List of structured, evidence-grounded IntelligenceEvent instances.
        """
        pass


class BaseEventSequenceTracker(ABC):
    """Abstract interface for managing optional historical state sequences across events."""

    @abstractmethod
    def append_event(
        self,
        sequence: EventSequence | None,
        event: IntelligenceEvent,
    ) -> EventSequence:
        """Associate an event with a historical sequence, updating the sequence's state chain.

        Args:
            sequence: Existing event sequence, or None to initialize a new sequence.
            event: IntelligenceEvent to append.

        Returns:
            Updated EventSequence containing the new event_id and latest state.
        """
        pass


class BaseStateComparator(ABC):
    """Abstract interface for comparing state values across domain transitions."""

    @abstractmethod
    def is_reversal(self, initial_state: Any, current_state: Any) -> bool:
        """Return True if current_state restores or reverts initial_state."""
        pass

    @abstractmethod
    def compare_direction(self, state_a: Any, state_b: Any) -> int | None:
        """Compare two states: returns 1 if increasing, -1 if decreasing, 0 if equal, None if unorderable."""
        pass


class BaseSignalConfidenceStrategy(ABC):
    """Abstract interface for computing signal confidence from supporting events."""

    @abstractmethod
    def compute_confidence(
        self,
        events: list[IntelligenceEvent],
        signal_type: Any,
        relationships: Any | None = None,
    ) -> Any:
        """Compute signal-level confidence score, qualitative level, and rationale."""
        pass


class BaseEventRelationshipStrategy(ABC):
    """Abstract strategy for evaluating pairwise relationships between IntelligenceEvents."""

    @abstractmethod
    def evaluate_relationship(
        self,
        event_a: IntelligenceEvent,
        event_b: IntelligenceEvent,
    ) -> Any:
        """Determine the relationship between two events."""
        pass


class BaseSignalGroupingStrategy(ABC):
    """Abstract strategy for clustering events into candidate signal groups."""

    @abstractmethod
    def group_events(
        self,
        events: list[IntelligenceEvent],
    ) -> dict[str, list[IntelligenceEvent]]:
        """Group events by company and subject/cluster key."""
        pass


class BaseSignalDetector(ABC):
    """Abstract interface for detecting InferredSignals from historical IntelligenceEvents."""

    @abstractmethod
    async def detect_signals(
        self,
        workspace_id: str,
        company_id: str,
        events: list[IntelligenceEvent],
        existing_signals: list[InferredSignal] | None = None,
    ) -> list[InferredSignal]:
        """Synthesize high-level InferredSignals from grounded IntelligenceEvents."""
        pass


from competitive_intelligence.intelligence.insights.interfaces import (
    BaseInsightValidator,
    BaseStrategicAnalyzer,
    BaseStrategicInterpreter,
)


__all__ = [
    "BaseAlertManager",
    "BaseChangeDetector",
    "BaseEventRelationshipStrategy",
    "BaseEventSequenceTracker",
    "BaseInsightGenerator",
    "BaseInsightValidator",
    "BaseIntelligenceAnalyzer",
    "BaseReportGenerator",
    "BaseSignalConfidenceStrategy",
    "BaseSignalDetector",
    "BaseSignalExtractor",
    "BaseSignalGroupingStrategy",
    "BaseStateComparator",
    "BaseStrategicAnalyzer",
    "BaseStrategicInterpreter",
]


# ==============================================================================
# Downstream Subsystem Placeholder Interfaces
# ==============================================================================


class BaseChangeDetector(ABC):
    """Abstract interface for detecting discrete factual changes between document versions."""

    @abstractmethod
    def detect_changes(
        self,
        previous_version: Any | None,
        current_version: Any,
        company_id: str,
    ) -> list[DetectedEvent]:
        """Compare versions deterministically and identify discrete factual events."""
        pass


class BaseSignalExtractor(ABC):
    """Abstract interface for inferring strategic signals from correlated events."""

    @abstractmethod
    async def extract_signals(
        self,
        events: list[DetectedEvent],
        context_evidence: list[RawEvidence],
    ) -> list[InferredSignal]:
        """Aggregate cross-source events into high-level strategic signals."""
        pass


class BaseInsightGenerator(ABC):
    """Abstract interface for synthesizing intelligence insights backed by evidence."""

    @abstractmethod
    async def generate_insights(
        self,
        workspace_id: str,
        signals: list[InferredSignal],
        events: list[DetectedEvent],
    ) -> list[GeneratedInsight]:
        """Synthesize actionable insights with linked provenance."""
        pass


class BaseAlertManager(ABC):
    """Abstract interface for evaluating and dispatching alerts."""

    @abstractmethod
    async def process_alerts(
        self,
        workspace_id: str,
        events: list[DetectedEvent],
        insights: list[GeneratedInsight],
    ) -> list[Alert]:
        """Generate alerts for high-severity events and insights."""
        pass


class BaseReportGenerator(ABC):
    """Abstract interface for assembling executive markdown reports."""

    @abstractmethod
    async def build_report(
        self,
        workspace_id: str,
        company_ids: list[str],
        insights: list[GeneratedInsight],
    ) -> IntelligenceReport:
        """Assemble structured intelligence report."""
        pass
