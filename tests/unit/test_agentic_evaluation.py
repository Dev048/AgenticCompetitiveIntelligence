"""Unit tests for Agentic RAG evaluation and benchmarking harness."""

import pytest

from competitive_intelligence.agentic_rag.evaluation.agentic_evaluator import (
    AgenticEvaluator,
)
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.retrieval_engine.evaluation.models import (
    EvaluationDataset,
    EvaluationQuery,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
)


class DummyBenchmarkRetriever(BaseRetriever):
    """Dummy retriever returning fixed chunk IDs for evaluation testing."""

    def __init__(self, returned_chunk_ids: list[str]) -> None:
        self.returned_chunk_ids = returned_chunk_ids

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=cid,
                score=1.0 - (i * 0.1),
                text=f"Text for {cid}",
                document_id="doc_1",
                version_id="ver_1",
                workspace_id=query.metadata_filter.workspace_id,
            )
            for i, cid in enumerate(self.returned_chunk_ids)
        ]


@pytest.fixture
def sample_eval_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        dataset_id="test_eval_ds",
        queries=[
            EvaluationQuery(
                query_id="q1",
                query_text="query 1",
                relevant_chunk_ids={"chunk_a": 3, "chunk_b": 2},
            ),
            EvaluationQuery(
                query_id="q2",
                query_text="query 2",
                relevant_chunk_ids={"chunk_c": 3},
            ),
        ],
        corpus=[],
    )


@pytest.mark.anyio
async def test_agentic_evaluator_metrics(sample_eval_dataset: EvaluationDataset) -> None:
    """Test AgenticEvaluator calculates Recall, MRR, NDCG, and latency."""
    retriever = DummyBenchmarkRetriever(returned_chunk_ids=["chunk_a", "chunk_c"])
    evaluator = AgenticEvaluator(dataset=sample_eval_dataset)

    metrics = await evaluator.evaluate_retriever(
        retriever=retriever,
        retriever_name="TestRetriever",
        k_values=[1, 2],
    )

    assert metrics.sample_size == 2
    assert metrics.retriever_name == "TestRetriever"
    assert metrics.recall_at_k[1] == 0.25  # hit chunk_a for q1 (1/2), 0/1 for q2 -> avg (0.5 + 0.0)/2 = 0.25
    assert metrics.mrr == 0.75  # 1.0 for q1 (rank 1), 0.5 for q2 (rank 2) -> avg 0.75
    assert metrics.average_latency_ms >= 0.0


@pytest.mark.anyio
async def test_agentic_evaluator_comparative_report(sample_eval_dataset: EvaluationDataset) -> None:
    """Test AgenticEvaluator generates side-by-side comparison report."""
    base_retriever = DummyBenchmarkRetriever(returned_chunk_ids=["chunk_a", "chunk_b"])
    agentic_retriever = AgenticRetriever(
        executor_retriever=base_retriever,
        llm_client=MockLLMClient(),
    )

    evaluator = AgenticEvaluator(dataset=sample_eval_dataset)
    report = await evaluator.compare(
        adaptive_retriever=base_retriever,
        agentic_retriever=agentic_retriever,
        k_values=[1, 2],
    )

    assert report.dataset_id == "test_eval_ds"
    assert report.adaptive_metrics.sample_size == 2
    assert report.agentic_metrics.sample_size == 2
    assert report.adaptive_metrics.retriever_name == "AdaptiveRetriever"
    assert report.agentic_metrics.retriever_name == "AgenticRetriever"
