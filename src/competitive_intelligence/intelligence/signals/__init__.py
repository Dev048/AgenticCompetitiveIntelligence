"""Signals package for Event History and Signal Detection."""

from competitive_intelligence.intelligence.signals.comparators import (
    ExactMatchStateComparator,
    NumericStateComparator,
)
from competitive_intelligence.intelligence.signals.detector import (
    DeterministicEventRelationshipStrategy,
    DeterministicSignalConfidenceStrategy,
    DeterministicSignalDetector,
    DeterministicSignalGroupingStrategy,
)
from competitive_intelligence.intelligence.signals.interfaces import (
    BaseEventRelationshipStrategy,
    BaseSignalConfidenceStrategy,
    BaseSignalDetector,
    BaseSignalGroupingStrategy,
    BaseStateComparator,
)
from competitive_intelligence.intelligence.signals.models import (
    EventCluster,
    SignalPolicy,
)
from competitive_intelligence.intelligence.signals.tracker import (
    DeterministicEventSequenceTracker,
)

__all__ = [
    "BaseEventRelationshipStrategy",
    "BaseSignalConfidenceStrategy",
    "BaseSignalDetector",
    "BaseSignalGroupingStrategy",
    "BaseStateComparator",
    "DeterministicEventRelationshipStrategy",
    "DeterministicEventSequenceTracker",
    "DeterministicSignalConfidenceStrategy",
    "DeterministicSignalDetector",
    "DeterministicSignalGroupingStrategy",
    "EventCluster",
    "ExactMatchStateComparator",
    "NumericStateComparator",
    "SignalPolicy",
]
