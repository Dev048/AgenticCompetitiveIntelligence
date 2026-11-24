"""Evaluation and benchmark infrastructure for the Competitive Intelligence Platform."""

from competitive_intelligence.evaluation.system_corpus import (
    CorpusDocument,
    GroundTruthScenario,
    OutcomeClassification,
    ScenarioResult,
    SystemCorpus,
    create_system_evaluation_corpus,
)

__all__ = [
    "CorpusDocument",
    "GroundTruthScenario",
    "OutcomeClassification",
    "ScenarioResult",
    "SystemCorpus",
    "create_system_evaluation_corpus",
]
