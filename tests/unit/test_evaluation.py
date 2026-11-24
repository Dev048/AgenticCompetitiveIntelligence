import math

import pytest

from competitive_intelligence.knowledge_pipeline.models import DocumentChunk
from competitive_intelligence.retrieval_engine.evaluation.baseline_dataset import (
    get_baseline_dataset,
)
from competitive_intelligence.retrieval_engine.evaluation.evaluator import RetrievalEvaluator
from competitive_intelligence.retrieval_engine.evaluation.metrics import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from competitive_intelligence.retrieval_engine.evaluation.models import (
    EvaluationDataset,
    EvaluationQuery,
)
from competitive_intelligence.retrieval_engine.models import RetrievalResult


def create_mock_result(chunk_id: str, score: float = 1.0) -> RetrievalResult:
    """Helper to create dummy RetrievalResults for testing."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text="Dummy text",
        document_id="doc1",
        version_id="v1",
        metadata={},
    )


def test_calculate_recall_at_k() -> None:
    """Test that Recall@K calculates hits using grades >= 2."""
    relevant_ids = {"c1": 3, "c2": 2, "c3": 1, "c4": 0}

    results = [
        create_mock_result("c1"),
        create_mock_result("c5"),
        create_mock_result("c2"),
    ]

    assert calculate_recall_at_k(results, relevant_ids, k=1) == 0.5
    assert calculate_recall_at_k(results, relevant_ids, k=2) == 0.5
    assert calculate_recall_at_k(results, relevant_ids, k=3) == 1.0


def test_calculate_mrr() -> None:
    """Test that MRR handles grades >= 2 as relevant."""
    relevant_ids = {"c1": 3, "c2": 2, "c3": 1}

    results = [
        create_mock_result("c5"),
        create_mock_result("c3"),  # grade 1, ignored
        create_mock_result("c1"),  # grade 3, relevant
    ]

    assert calculate_mrr(results, relevant_ids) == 1.0 / 3.0


def test_calculate_ndcg_at_k() -> None:
    """Test NDCG calculation utilizing full 0-3 grading scale."""
    relevant_ids = {"c1": 3, "c2": 2, "c3": 1}

    results = [
        create_mock_result("c3"),
        create_mock_result("c5"),
        create_mock_result("c1"),
    ]

    ndcg_1 = calculate_ndcg_at_k(results, relevant_ids, k=1)
    assert math.isclose(ndcg_1, 1.0 / 3.0)

    idcg_2 = 3.0 + 2.0 / math.log2(3)
    ndcg_2 = calculate_ndcg_at_k(results, relevant_ids, k=2)
    assert math.isclose(ndcg_2, 1.0 / idcg_2)

    idcg_3 = idcg_2 + 1.0 / math.log2(4)
    dcg_3 = 1.0 + 3.0 / math.log2(4)
    ndcg_3 = calculate_ndcg_at_k(results, relevant_ids, k=3)
    assert math.isclose(ndcg_3, dcg_3 / idcg_3)


class MockRetriever:
    """A dummy retriever for evaluation pipeline testing."""

    def __init__(self, results: list[RetrievalResult], delay: float = 0.0) -> None:
        self.results = results
        self.delay = delay
        self.called_count = 0

    def retrieve(self, query_text: str) -> list[RetrievalResult]:
        self.called_count += 1
        import time

        if self.delay > 0:
            time.sleep(self.delay)
        return self.results


@pytest.mark.anyio
async def test_retrieval_evaluator() -> None:
    """Test evaluator pipeline, metric macro-averaging, and latency measurement."""
    dataset = EvaluationDataset(
        dataset_id="test",
        queries=[
            EvaluationQuery(
                query_id="q1",
                query_text="query 1",
                relevant_chunk_ids={"c1": 3, "c2": 2},
            ),
            EvaluationQuery(
                query_id="q2",
                query_text="query 2",
                relevant_chunk_ids={"c3": 3},
            ),
        ],
        corpus=[],
    )

    results = [
        create_mock_result("c1"),
        create_mock_result("c3"),
    ]

    retriever = MockRetriever(results, delay=0.01)
    evaluator = RetrievalEvaluator(dataset)

    metrics = await evaluator.evaluate(
        test_dataset_path="", retriever=retriever, k_values=[1, 2], warm_up=True
    )

    assert retriever.called_count == 3  # 1 warm up + 2 queries
    assert metrics.sample_size == 2

    assert metrics.recall_at_k[1] == 0.25
    assert metrics.recall_at_k[2] == 0.75
    assert metrics.mrr == 0.75
    assert metrics.average_latency_ms >= 10.0


def test_baseline_dataset() -> None:
    """Verify baseline dataset contents and query-chunk references."""
    dataset = get_baseline_dataset()
    assert len(dataset.corpus) >= 50
    assert len(dataset.queries) >= 20

    corpus_ids = {c.chunk_id for c in dataset.corpus}
    for q in dataset.queries:
        for c_id in q.relevant_chunk_ids:
            assert c_id in corpus_ids


def test_baseline_dataset_v2() -> None:
    """Verify v2 baseline dataset exact dimensions and strict rules."""
    from competitive_intelligence.retrieval_engine.evaluation.baseline_dataset_v2 import get_baseline_dataset_v2
    dataset = get_baseline_dataset_v2()
    
    # 1. Corpus size (200-300 chunks) and Query count (40-60 queries)
    assert 200 <= len(dataset.corpus) <= 300, f"Expected 200-300 chunks, got {len(dataset.corpus)}"
    assert 40 <= len(dataset.queries) <= 60, f"Expected 40-60 queries, got {len(dataset.queries)}"
    
    corpus_ids = {c.chunk_id for c in dataset.corpus}
    
    for q in dataset.queries:
        # 2. All relevant_chunk_ids exist in corpus
        for c_id in q.relevant_chunk_ids:
            assert c_id in corpus_ids, f"Missing chunk {c_id} in corpus for query {q.query_id}"
            
        # 3. Each query has at least one grade >= 2 item
        has_high_grade = any(grade >= 2 for grade in q.relevant_chunk_ids.values())
        assert has_high_grade, f"Query {q.query_id} has no highly relevant items"
        
        # 4. No query has all corpus items relevant
        assert len(q.relevant_chunk_ids) < len(dataset.corpus)

    # 5. Check source-type distribution
    sources = set(c.metadata.source_category for c in dataset.corpus)
    expected_sources = {"pricing", "blogs", "careers", "github", "news", "reddit"}
    assert sources == expected_sources, f"Missing expected source categories. Got {sources}"
    
    # 6. No accidental duplicate chunk IDs
    assert len(corpus_ids) == len(dataset.corpus), "Corpus contains duplicate chunk IDs"
