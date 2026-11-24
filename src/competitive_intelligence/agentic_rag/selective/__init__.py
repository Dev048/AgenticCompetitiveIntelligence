"""Selective Agentic RAG Escalation Subsystem.

Provides deterministic escalation policy, controller, models, and evaluation
for selective routing between AdaptiveRetriever and AgenticRetriever.
"""

from competitive_intelligence.agentic_rag.selective.controller import (
    SelectiveAgenticRetriever,
)
from competitive_intelligence.agentic_rag.selective.evaluation import (
    SelectiveAgenticEvaluator,
    SelectiveBenchmarkReport,
    SelectiveComparativeRecord,
    SelectiveRoutingSummary,
    SelectiveSystemMetrics,
)
from competitive_intelligence.agentic_rag.selective.models import (
    EscalationAssessment,
    EscalationPolicyConfig,
    EscalationSignalType,
    ResultSelectionStrategy,
    SelectiveExecutionTrace,
    SelectiveRetrieverConfig,
)
from competitive_intelligence.agentic_rag.selective.policy import (
    DeterministicEscalationPolicy,
)

__all__ = [
    "DeterministicEscalationPolicy",
    "EscalationAssessment",
    "EscalationPolicyConfig",
    "EscalationSignalType",
    "ResultSelectionStrategy",
    "SelectiveAgenticEvaluator",
    "SelectiveAgenticRetriever",
    "SelectiveBenchmarkReport",
    "SelectiveComparativeRecord",
    "SelectiveExecutionTrace",
    "SelectiveRetrieverConfig",
    "SelectiveRoutingSummary",
    "SelectiveSystemMetrics",
]

