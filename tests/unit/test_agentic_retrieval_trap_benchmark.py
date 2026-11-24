"""Unit tests for Hard-Case Retrieval-Trap Agentic RAG benchmark."""

from datetime import datetime, timezone
import pytest
from typing import Any

from competitive_intelligence.agentic_rag.evaluation.trap_dataset import (
    get_retrieval_trap_dataset,
)
from competitive_intelligence.agentic_rag.evaluation.trap_evaluator import (
    TrapAgenticEvaluator,
)
from competitive_intelligence.agentic_rag.evaluation.multihop_models import (
    DivergenceClassification,
)
from competitive_intelligence.agentic_rag.evaluation.trap_models import (
    TrapBenchmarkCategory,
    TrapBenchmarkReport,
    TrapEvaluationDataset,
    TrapEvaluationQuery,
    TrapQueryExecutionRecord,
)
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.models import (
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.knowledge_pipeline.models import ChunkMetadata, DocumentChunk
from competitive_intelligence.knowledge_pipeline.normalizer import compute_content_hash
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
)


def _make_test_chunk(chunk_id: str, content: str) -> DocumentChunk:
    meta = ChunkMetadata(
        source_id="src_test",
        document_id="doc_test",
        version_id="v1",
        company_id="test_co",
        workspace_id="ws_test",
        source_url="https://test.internal/doc",
        source_category="test",
        collected_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        start_char=0,
        end_char=len(content),
        chunk_index=0,
        total_chunks=1,
    )
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc_test",
        version_id="v1",
        content=content,
        content_hash=compute_content_hash(content),
        metadata=meta,
        created_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def test_retrieval_trap_dataset_integrity() -> None:
    """Verify that the retrieval-trap dataset has valid structure, frozen IDs, and zero dangling references."""
    dataset = get_retrieval_trap_dataset()

    # 1. Size checks
    assert len(dataset.corpus) >= 200, f"Corpus size {len(dataset.corpus)} should be >= 200 chunks"
    assert len(dataset.queries) == 50, f"Query count {len(dataset.queries)} should be exactly 50 queries"

    # 2. No duplicate query IDs
    q_ids = [q.query_id for q in dataset.queries]
    assert len(q_ids) == len(set(q_ids)), "Duplicate query IDs detected in benchmark dataset"

    # 3. Ground truth reference validation
    errors = dataset.validate_ground_truth()
    assert len(errors) == 0, f"Validation errors found: {errors}"

    # 4. 10 Category coverage
    all_categories = set(TrapBenchmarkCategory)
    covered_categories = {q.category for q in dataset.queries}
    assert all_categories == covered_categories, f"Missing categories: {all_categories - covered_categories}"

    for cat in TrapBenchmarkCategory:
        cat_queries = dataset.get_queries_by_category(cat)
        assert len(cat_queries) >= 5, f"Category {cat.value} has fewer than 5 queries ({len(cat_queries)})"

    # 5. Non-negative queries must specify required_chunk_ids
    for q in dataset.queries:
        if not q.is_negative_case:
            assert len(q.required_chunk_ids) >= 1, f"Non-negative query {q.query_id} has no required_chunk_ids"
            for cid in q.required_chunk_ids:
                assert cid in q.relevant_chunk_ids, f"Required chunk {cid} not in relevant_chunk_ids for {q.query_id}"
                assert q.relevant_chunk_ids[cid] >= 2, f"Required chunk {cid} has low relevance score for {q.query_id}"
        else:
            assert len(q.relevant_chunk_ids) == 0, f"Negative query {q.query_id} should have empty relevant_chunk_ids"


def test_divergence_classification_rules() -> None:
    """Test deterministic divergence classification rules for all benchmark conditions."""
    evaluator = TrapAgenticEvaluator(
        dataset=TrapEvaluationDataset(
            dataset_id="test_ds",
            corpus=[
                _make_test_chunk("chunk_01", "SLA commitment 99.9%"),
                _make_test_chunk("chunk_02", "Credit remedy refund schedule 25%"),
            ],
            queries=[],
        )
    )

    query = TrapEvaluationQuery(
        query_id="q_test",
        query_text="What is the SLA and refund credit?",
        category=TrapBenchmarkCategory.RETRIEVAL_TRAP,
        relevant_chunk_ids={"chunk_01": 3, "chunk_02": 3},
        required_chunk_ids=["chunk_01", "chunk_02"],
    )

    # 1. Negative query -> NEUTRAL_NEGATIVE
    neg_query = TrapEvaluationQuery(
        query_id="q_neg",
        query_text="Unanswerable query",
        category=TrapBenchmarkCategory.NEGATIVE_ADVERSARIAL,
        relevant_chunk_ids={},
        required_chunk_ids=[],
        is_negative_case=True,
    )
    div, _ = evaluator._classify_divergence(
        query=neg_query,
        ad_chunks=["chunk_01"],
        ag_chunks=["chunk_01"],
        ad_r10=0.0,
        ag_r10=0.0,
        ad_mrr=0.0,
        ag_mrr=0.0,
        adaptive_missed_required=False,
        agentic_recovered_missed=False,
        llm_calls=1,
        latency_ms=5.0,
    )
    assert div == DivergenceClassification.NEUTRAL_NEGATIVE

    # 2. Both failed -> BOTH_FAILED
    div, _ = evaluator._classify_divergence(
        query=query,
        ad_chunks=["chunk_99"],
        ag_chunks=["chunk_99"],
        ad_r10=0.0,
        ag_r10=0.0,
        ad_mrr=0.0,
        ag_mrr=0.0,
        adaptive_missed_required=True,
        agentic_recovered_missed=False,
        llm_calls=2,
        latency_ms=10.0,
    )
    assert div == DivergenceClassification.BOTH_FAILED

    # 3. Adaptive missed required evidence, Agentic recovered it -> AGENTIC_SUCCESS
    div, _ = evaluator._classify_divergence(
        query=query,
        ad_chunks=["chunk_01"],
        ag_chunks=["chunk_01", "chunk_02"],
        ad_r10=0.5,
        ag_r10=1.0,
        ad_mrr=1.0,
        ag_mrr=1.0,
        adaptive_missed_required=True,
        agentic_recovered_missed=True,
        llm_calls=3,
        latency_ms=15.0,
    )
    assert div == DivergenceClassification.AGENTIC_SUCCESS

    # 4. Both retrieved all required evidence -> AGENTIC_OVERHEAD
    div, _ = evaluator._classify_divergence(
        query=query,
        ad_chunks=["chunk_01", "chunk_02"],
        ag_chunks=["chunk_01", "chunk_02"],
        ad_r10=1.0,
        ag_r10=1.0,
        ad_mrr=1.0,
        ag_mrr=1.0,
        adaptive_missed_required=False,
        agentic_recovered_missed=False,
        llm_calls=2,
        latency_ms=12.0,
    )
    assert div == DivergenceClassification.AGENTIC_OVERHEAD

    # 5. Adaptive succeeded, Agentic dropped a required chunk -> ADAPTIVE_BETTER
    div, _ = evaluator._classify_divergence(
        query=query,
        ad_chunks=["chunk_01", "chunk_02"],
        ag_chunks=["chunk_01"],
        ad_r10=1.0,
        ag_r10=0.5,
        ad_mrr=1.0,
        ag_mrr=1.0,
        adaptive_missed_required=False,
        agentic_recovered_missed=False,
        llm_calls=2,
        latency_ms=12.0,
    )
    assert div == DivergenceClassification.ADAPTIVE_BETTER


@pytest.mark.anyio
async def test_trap_evaluator_metrics_calculation() -> None:
    """Verify that TrapAgenticEvaluator computes accurate recovery rates and percentile latencies."""

    chunk_a = _make_test_chunk("chunk_a", "SLA 99.9%")
    chunk_b = _make_test_chunk("chunk_b", "Remedy 20%")

    q1 = TrapEvaluationQuery(
        query_id="q1",
        query_text="SLA and remedy",
        category=TrapBenchmarkCategory.RETRIEVAL_TRAP,
        relevant_chunk_ids={"chunk_a": 3, "chunk_b": 3},
        required_chunk_ids=["chunk_a", "chunk_b"],
    )

    dataset = TrapEvaluationDataset(
        dataset_id="test_trap_eval_ds",
        corpus=[chunk_a, chunk_b],
        queries=[q1],
    )

    class MockAdaptive(BaseRetriever):
        async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
            # Returns only chunk_a (misses chunk_b)
            return [
                RetrievalResult(
                    chunk_id="chunk_a",
                    document_id="doc_test",
                    version_id="v1",
                    source_id="src_test",
                    text="SLA 99.9%",
                    score=0.9,
                )
            ]

    class MockAgentic(BaseRetriever):
        async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
            # Recovers both chunk_a and chunk_b
            return [
                RetrievalResult(
                    chunk_id="chunk_a",
                    document_id="doc_test",
                    version_id="v1",
                    source_id="src_test",
                    text="SLA 99.9%",
                    score=0.9,
                ),
                RetrievalResult(
                    chunk_id="chunk_b",
                    document_id="doc_test",
                    version_id="v1",
                    source_id="src_test",
                    text="Remedy 20%",
                    score=0.85,
                ),
            ]

    evaluator = TrapAgenticEvaluator(dataset=dataset)
    report = await evaluator.evaluate_all(
        adaptive_retriever=MockAdaptive(),
        agentic_retriever=MockAgentic(),
        top_k=10,
    )

    assert report.total_queries == 1
    assert report.trap_specific_metrics["adaptive_miss_count"] == 1
    assert report.trap_specific_metrics["agentic_recovery_count"] == 1
    assert report.trap_specific_metrics["agentic_recovery_rate"] == 1.0
    assert report.trap_specific_metrics["agentic_unique_evidence_count"] == 1
    assert report.divergence_distribution[DivergenceClassification.AGENTIC_SUCCESS.value] == 1


def test_mock_llm_anti_cheating_invariants() -> None:
    """Verify that MockLLM handles generic parsing without query-to-chunk answer keys."""
    from competitive_intelligence.agentic_rag.evaluation import (
        create_trap_benchmark_mock_llm,
    )

    mock_llm = create_trap_benchmark_mock_llm()

    # Query analysis handles generic linguistic patterns
    assert mock_llm.custom_handler is not None

    # Test multi-entity decomposition
    qa_res = mock_llm.custom_handler(
        "Original Query: Compare Snowflake and Databricks pricing\nAvailable Tools:",
        QueryAnalysis,
    )
    assert isinstance(qa_res, QueryAnalysis)
    assert qa_res.decision == RetrievalDecision.DECOMPOSE_AND_RETRIEVE
    assert len(qa_res.sub_queries) == 2

    # Test direct easy query
    qa_direct = mock_llm.custom_handler(
        "Original Query: What is the Pinecone Serverless read unit price?\nAvailable Tools:",
        QueryAnalysis,
    )
    assert isinstance(qa_direct, QueryAnalysis)
    assert qa_direct.decision == RetrievalDecision.DIRECT_RETRIEVAL
