"""Unit tests for query routing evaluation dataset, evaluator metrics, and cost analysis."""

import pytest

from competitive_intelligence.retrieval_engine.evaluation.routing_dataset import (
    get_routing_evaluation_dataset,
)
from competitive_intelligence.retrieval_engine.evaluation.routing_evaluator import (
    RoutingEvaluator,
)
from competitive_intelligence.retrieval_engine.evaluation.routing_models import (
    RoutingCostEstimateConfig,
    RoutingEvaluationDataset,
    RoutingEvaluationQuery,
)
from competitive_intelligence.retrieval_engine.models import RetrievalStrategy
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
def routing_dataset() -> RoutingEvaluationDataset:
    return get_routing_evaluation_dataset()


def test_routing_dataset_composition(routing_dataset: RoutingEvaluationDataset) -> None:
    """Verify routing dataset composition, size, and category distribution."""
    assert len(routing_dataset.queries) == 36

    categories = {q.category for q in routing_dataset.queries}
    expected_categories = {
        "exact_lookup",
        "conceptual",
        "numeric_entity",
        "complex_identifier",
        "multi_entity_comparison",
        "ambiguous_fallback",
    }
    assert categories == expected_categories

    # Exactly 6 queries per category
    for cat in expected_categories:
        count = sum(1 for q in routing_dataset.queries if q.category == cat)
        assert count == 6, f"Expected 6 queries for category {cat}, got {count}"


def test_end_to_end_routing_evaluation(
    default_router: DeterministicRetrievalRouter,
    routing_dataset: RoutingEvaluationDataset,
) -> None:
    """Test full routing evaluation and assert 100% accuracy on the labeled dataset."""
    evaluator = RoutingEvaluator()
    report = evaluator.evaluate(default_router, routing_dataset)

    assert report.total_queries == 36
    assert report.overall_accuracy == 1.0
    assert report.strategy_accuracy == 1.0
    assert report.reranker_accuracy == 1.0
    assert report.candidate_pool_accuracy == 1.0
    assert report.final_top_k_accuracy == 1.0
    assert report.rule_accuracy == 1.0


def test_confusion_matrix_generation(
    default_router: DeterministicRetrievalRouter,
    routing_dataset: RoutingEvaluationDataset,
) -> None:
    """Test confusion matrix accurately records expected vs actual strategy counts."""
    evaluator = RoutingEvaluator()
    report = evaluator.evaluate(default_router, routing_dataset)

    matrix = report.confusion_matrix
    assert matrix["sparse_bm25"]["sparse_bm25"] == 6
    assert matrix["dense_vector"]["dense_vector"] == 6
    assert matrix["hybrid_rrf"]["hybrid_rrf"] == 24

    # No cross-strategy errors
    assert matrix["sparse_bm25"]["dense_vector"] == 0
    assert matrix["dense_vector"]["sparse_bm25"] == 0
    assert matrix["hybrid_rrf"]["sparse_bm25"] == 0


def test_cost_efficiency_analysis(
    default_router: DeterministicRetrievalRouter,
    routing_dataset: RoutingEvaluationDataset,
) -> None:
    """Test cost analysis accurately computes savings over fixed 30-pool Hybrid+Reranker."""
    custom_costs = RoutingCostEstimateConfig(
        bm25_cost_ms=1.0,
        semantic_cost_ms=400.0,
        hybrid_no_rerank_cost_ms=450.0,
        hybrid_rerank_20_cost_ms=8000.0,
        hybrid_rerank_30_cost_ms=11800.0,
        baseline_all_hybrid_rerank_cost_ms=11800.0,
    )
    evaluator = RoutingEvaluator(cost_config=custom_costs)
    report = evaluator.evaluate(default_router, routing_dataset)

    # 6 BM25 queries (6 * 1 = 6)
    # 6 Semantic queries (6 * 400 = 2400)
    # 18 Hybrid+Rerank Pool=20 queries (18 * 8000 = 144000)
    # 6 Hybrid+Rerank Pool=30 queries (6 * 11800 = 70800)
    # Total = 6 + 2400 + 144000 + 70800 = 217206
    # Avg = 217206 / 36 = 6033.5 ms
    expected_avg = (6 * 1.0 + 6 * 400.0 + 18 * 8000.0 + 6 * 11800.0) / 36.0
    assert abs(report.average_routed_estimated_cost - expected_avg) < 0.1
    assert report.baseline_fixed_estimated_cost == 11800.0

    # Savings percentage: (1 - 6033.5 / 11800) * 100 ~ 48.87%
    assert report.estimated_compute_savings_pct > 45.0
    assert report.estimated_compute_savings_pct < 55.0


def test_deterministic_repeated_evaluation(
    default_router: DeterministicRetrievalRouter,
    routing_dataset: RoutingEvaluationDataset,
) -> None:
    """Test routing evaluation is 100% deterministic across repeated runs."""
    evaluator = RoutingEvaluator()
    report1 = evaluator.evaluate(default_router, routing_dataset)
    report2 = evaluator.evaluate(default_router, routing_dataset)

    assert report1.overall_accuracy == report2.overall_accuracy
    assert report1.average_routed_estimated_cost == report2.average_routed_estimated_cost
    assert report1.confusion_matrix == report2.confusion_matrix


def test_empty_dataset_raises_error(default_router: DeterministicRetrievalRouter) -> None:
    """Test evaluator raises ValueError on empty dataset."""
    empty_ds = RoutingEvaluationDataset(dataset_id="empty", queries=[])
    evaluator = RoutingEvaluator()

    with pytest.raises(ValueError, match="contains 0 queries"):
        evaluator.evaluate(default_router, empty_ds)
