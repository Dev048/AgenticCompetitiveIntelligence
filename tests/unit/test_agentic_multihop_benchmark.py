"""Unit tests for Multi-Hop and Adversarial Benchmark dataset, evaluator, and divergence classification."""

import pytest

from competitive_intelligence.agentic_rag.evaluation.multihop_dataset import (
    get_multihop_evaluation_dataset,
)
from competitive_intelligence.agentic_rag.evaluation.multihop_evaluator import (
    MultiHopAgenticEvaluator,
)
from competitive_intelligence.agentic_rag.evaluation.multihop_models import (
    BenchmarkCategory,
    DivergenceClassification,
    MultiHopComparativeReport,
    MultiHopEvaluationDataset,
    MultiHopEvaluationQuery,
)
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.models import (
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
)


class MockRetriever(BaseRetriever):
    """Configurable mock retriever for unit tests."""

    def __init__(self, responses: dict[str, list[RetrievalResult]] | None = None) -> None:
        self.responses = responses or {}
        self.executed_queries: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.executed_queries.append(query)
        return self.responses.get(query.query_text, [])


def make_test_result(chunk_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"Text for {chunk_id}",
        document_id="doc_1",
        version_id="v1",
        workspace_id="ws_multihop_eval",
        company_id="snowflake",
    )


# ==============================================================================
# 1. Dataset Integrity & Ground Truth Tests
# ==============================================================================


def test_multihop_dataset_schema_and_ground_truth_validity() -> None:
    """Verify that dataset contains 115 chunks, 50 queries, and 0 invalid ground-truth references."""
    dataset = get_multihop_evaluation_dataset()

    assert isinstance(dataset, MultiHopEvaluationDataset)
    assert len(dataset.corpus) == 115
    assert len(dataset.queries) == 50

    # Validate ground truth references
    errors = dataset.validate_ground_truth()
    assert len(errors) == 0, f"Ground truth validation errors: {errors}"


def test_multihop_dataset_covers_all_nine_categories() -> None:
    """Verify all 9 benchmark categories are populated with queries."""
    dataset = get_multihop_evaluation_dataset()

    category_counts: dict[BenchmarkCategory, int] = {}
    for q in dataset.queries:
        category_counts[q.category] = category_counts.get(q.category, 0) + 1

    # Verify all 9 enum categories are present
    assert len(category_counts) == 9
    for cat in BenchmarkCategory:
        assert cat in category_counts
        assert category_counts[cat] >= 5, f"Category {cat.value} has fewer than 5 queries"


# ==============================================================================
# 2. Evaluator Calculations & Percentiles
# ==============================================================================


def test_evaluator_percentile_calculations() -> None:
    """Test percentile calculations (mean, median, p95)."""
    dataset = get_multihop_evaluation_dataset()
    evaluator = MultiHopAgenticEvaluator(dataset=dataset)

    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    mean_val, med_val, p95_val = evaluator._compute_percentiles(values)

    assert mean_val == 55.0
    assert med_val == 55.0
    assert p95_val > 90.0

    # Empty list edge case
    assert evaluator._compute_percentiles([]) == (0.0, 0.0, 0.0)


# ==============================================================================
# 3. Divergence Classification Tests
# ==============================================================================


def test_evaluator_divergence_classification_rules() -> None:
    """Test deterministic divergence classification under all conditions."""
    dataset = get_multihop_evaluation_dataset()
    evaluator = MultiHopAgenticEvaluator(dataset=dataset)

    # 1. Negative case
    neg_q = MultiHopEvaluationQuery(
        query_id="q_neg",
        query_text="Non-existent query",
        category=BenchmarkCategory.ADVERSARIAL_NEGATIVE,
        is_negative_case=True,
    )
    div, _ = evaluator._classify_divergence(
        query=neg_q, ad_r10=0.0, ag_r10=0.0, ad_mrr=0.0, ag_mrr=0.0, llm_calls=0, latency_ms=5.0
    )
    assert div == DivergenceClassification.NEUTRAL_NEGATIVE

    # 2. Both failed case
    std_q = MultiHopEvaluationQuery(
        query_id="q_std",
        query_text="Standard query",
        category=BenchmarkCategory.MULTI_ENTITY,
        relevant_chunk_ids={"chunk_001": 3},
    )
    div, _ = evaluator._classify_divergence(
        query=std_q, ad_r10=0.0, ag_r10=0.0, ad_mrr=0.0, ag_mrr=0.0, llm_calls=2, latency_ms=50.0
    )
    assert div == DivergenceClassification.BOTH_FAILED

    # 3. Agentic success case (Recall gain >= 15%)
    div, _ = evaluator._classify_divergence(
        query=std_q, ad_r10=0.50, ag_r10=1.00, ad_mrr=0.50, ag_mrr=1.00, llm_calls=2, latency_ms=50.0
    )
    assert div == DivergenceClassification.AGENTIC_SUCCESS

    # 4. Adaptive better case (Adaptive Recall gain >= 15%)
    div, _ = evaluator._classify_divergence(
        query=std_q, ad_r10=1.00, ag_r10=0.50, ad_mrr=1.00, ag_mrr=0.50, llm_calls=2, latency_ms=50.0
    )
    assert div == DivergenceClassification.ADAPTIVE_BETTER

    # 5. Agentic overhead case (Equal recall, extra latency/calls)
    div, _ = evaluator._classify_divergence(
        query=std_q, ad_r10=1.00, ag_r10=1.00, ad_mrr=1.00, ag_mrr=1.00, llm_calls=2, latency_ms=60.0
    )
    assert div == DivergenceClassification.AGENTIC_OVERHEAD


# ==============================================================================
# 4. End-to-End Evaluation Execution
# ==============================================================================


@pytest.mark.anyio
async def test_multihop_evaluator_evaluate_query() -> None:
    """Test single query evaluation through both retrievers."""
    dataset = get_multihop_evaluation_dataset()
    evaluator = MultiHopAgenticEvaluator(dataset=dataset)

    query = dataset.queries[0]
    r1 = make_test_result("chunk_001", score=0.95)
    r2 = make_test_result("chunk_010", score=0.90)

    ad_retriever = MockRetriever({query.query_text: [r1]})
    ag_inner = MockRetriever({query.query_text: [r1, r2]})
    ag_retriever = AgenticRetriever(executor_retriever=ag_inner, llm_client=MockLLMClient())

    record = await evaluator.evaluate_query(
        query=query,
        adaptive_retriever=ad_retriever,
        agentic_retriever=ag_retriever,
        top_k=10,
    )

    assert record.query_id == query.query_id
    assert record.adaptive_recall_at_10 > 0
    assert record.agentic_recall_at_10 > 0
    assert isinstance(record.classification, DivergenceClassification)


@pytest.mark.anyio
async def test_multihop_evaluator_evaluate_all() -> None:
    """Test full multi-hop dataset evaluation and report generation."""
    # Create a small 2-query subset for fast unit testing
    dataset = get_multihop_evaluation_dataset()
    small_dataset = MultiHopEvaluationDataset(
        dataset_id="test_small",
        corpus=dataset.corpus[:10],
        queries=dataset.queries[:2],
    )

    evaluator = MultiHopAgenticEvaluator(dataset=small_dataset)

    mock_base = MockRetriever()
    agentic = AgenticRetriever(executor_retriever=mock_base, llm_client=MockLLMClient())

    report: MultiHopComparativeReport = await evaluator.evaluate_all(
        adaptive_retriever=mock_base,
        agentic_retriever=agentic,
        top_k=5,
    )

    assert report.total_queries == 2
    assert len(report.query_records) == 2
    assert "recall_at_10" in report.overall_adaptive_metrics
    assert "recall_at_10" in report.overall_agentic_metrics
    assert len(report.category_metrics) > 0
