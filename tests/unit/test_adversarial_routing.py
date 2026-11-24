"""Unit tests for held-out and adversarial routing evaluation dataset."""

import pytest

from competitive_intelligence.retrieval_engine.evaluation.adversarial_dataset import (
    get_adversarial_routing_dataset,
)
from competitive_intelligence.retrieval_engine.evaluation.routing_evaluator import (
    RoutingEvaluator,
)
from competitive_intelligence.retrieval_engine.evaluation.routing_models import (
    RoutingEvaluationDataset,
)
from competitive_intelligence.retrieval_engine.routing.models import (
    RouterConfiguration,
)
from competitive_intelligence.retrieval_engine.routing.router import (
    DeterministicRetrievalRouter,
)


@pytest.fixture
def default_router() -> DeterministicRetrievalRouter:
    config = RouterConfiguration(
        known_entities=[
            "Snowflake", "Databricks", "Microsoft", "OpenAI", "Acme", "Stripe",
            "Google", "Pinecone", "Qdrant", "Anthropic", "Apple", "Amazon", "Meta", "Tesla", "NVIDIA",
        ],
        short_lookup_candidate_pool_size=10,
        numeric_candidate_pool_size=20,
        comparison_candidate_pool_size=30,
        conceptual_candidate_pool_size=20,
        fallback_candidate_pool_size=20,
        exact_lookup_top_k=5,
        numeric_top_k=5,
        default_top_k=10,
        short_lookup_max_tokens=5,
    )
    return DeterministicRetrievalRouter(config=config)


@pytest.fixture
def adversarial_dataset() -> RoutingEvaluationDataset:
    return get_adversarial_routing_dataset()


def test_adversarial_dataset_structure(adversarial_dataset: RoutingEvaluationDataset) -> None:
    """Verify adversarial dataset size and category variety."""
    assert len(adversarial_dataset.queries) == 38
    assert len({q.category for q in adversarial_dataset.queries}) >= 8


def test_adversarial_routing_evaluation(
    default_router: DeterministicRetrievalRouter,
    adversarial_dataset: RoutingEvaluationDataset,
) -> None:
    """Test adversarial evaluation produces high generalization metrics."""
    evaluator = RoutingEvaluator()
    report = evaluator.evaluate(default_router, adversarial_dataset)

    assert report.total_queries == 38
    assert report.strategy_accuracy > 0.90
    assert report.reranker_accuracy > 0.90
    assert report.candidate_pool_accuracy > 0.90
    assert report.overall_accuracy > 0.85
    assert report.estimated_compute_savings_pct > 40.0


def test_embedded_year_pattern_inspection(
    default_router: DeterministicRetrievalRouter,
) -> None:
    """Explicitly verify and document behavior of embedded year patterns in identifiers."""
    # Hyphenated identifiers: '-' is \W (non-word char), so '\b2023\b' matches as a temporal word
    decision_cve = default_router.route("CVE-2023-9912")
    assert decision_cve.features.has_temporal is True
    assert decision_cve.rule_name == "ComplexIdentifierOrVersionRule"

    decision_rfc = default_router.route("RFC-2024-STD")
    assert decision_rfc.features.has_temporal is True
    assert decision_rfc.rule_name == "ComplexIdentifierOrVersionRule"

    # Underscore identifiers: '_' is \w (word char), so '\b2022\b' does NOT match as a standalone word
    decision_build = default_router.route("BUILD_2022_FINAL")
    assert decision_build.features.has_temporal is False
    assert decision_build.rule_name == "ShortExactLookupRule"
