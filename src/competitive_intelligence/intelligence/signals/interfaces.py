"""Abstract interfaces for Event History and Signal Detection."""

from abc import ABC, abstractmethod
from typing import Any

from competitive_intelligence.intelligence.models import (
    EventRelationship,
    InferredSignal,
    IntelligenceEvent,
    SignalConfidenceAssessment,
    SignalType,
)


class BaseStateComparator(ABC):
    """Abstract interface for comparing state values across domain transitions."""

    @abstractmethod
    def is_reversal(self, initial_state: Any, current_state: Any) -> bool:
        """Return True if current_state restores or reverts initial_state."""
        pass

    @abstractmethod
    def compare_direction(self, state_a: Any, state_b: Any) -> int | None:
        """Compare two states: returns 1 if increasing/expanding, -1 if decreasing, 0 if equal, None if unorderable."""
        pass


class BaseSignalConfidenceStrategy(ABC):
    """Abstract interface for computing signal confidence from supporting events."""

    @abstractmethod
    def compute_confidence(
        self,
        events: list[IntelligenceEvent],
        signal_type: SignalType,
        relationships: list[EventRelationship] | None = None,
    ) -> SignalConfidenceAssessment:
        """Compute signal-level confidence score, qualitative level, and rationale."""
        pass


class BaseEventRelationshipStrategy(ABC):
    """Abstract strategy for evaluating pairwise relationships between IntelligenceEvents."""

    @abstractmethod
    def evaluate_relationship(
        self,
        event_a: IntelligenceEvent,
        event_b: IntelligenceEvent,
    ) -> EventRelationship:
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
