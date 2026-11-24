"""Evaluator for benchmarking AdaptiveRetriever vs AgenticRetriever."""

import time
from typing import Sequence

from competitive_intelligence.agentic_rag.evaluation.models import (
    AgenticBenchmarkMetrics,
    AgenticComparativeReport,
)
from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.retrieval_engine.evaluation.metrics import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from competitive_intelligence.retrieval_engine.evaluation.models import EvaluationDataset
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)


class AgenticEvaluator:
    """Evaluates retrievers across benchmark datasets measuring IR metrics, latency, and agentic overhead."""

    def __init__(self, dataset: EvaluationDataset) -> None:
        self.dataset = dataset

    async def evaluate_retriever(
        self,
        retriever: BaseRetriever,
        retriever_name: str = "retriever",
        k_values: Sequence[int] | None = None,
        top_k: int = 20,
    ) -> AgenticBenchmarkMetrics:
        """Run IR evaluation across all dataset queries for a single retriever."""
        if k_values is None:
            k_values = [5, 10, 20]

        queries = self.dataset.queries
        num_queries = len(queries)
        if num_queries == 0:
            return AgenticBenchmarkMetrics(
                retriever_name=retriever_name,
                sample_size=0,
            )

        total_mrr = 0.0
        total_recall = {k: 0.0 for k in k_values}
        total_ndcg = {k: 0.0 for k in k_values}
        total_latency_ms = 0.0
        total_llm_calls = 0
        total_retrieval_hops = 0
        fallback_queries_count = 0

        for q in queries:
            m_filter = q.metadata_filter or MetadataFilter(workspace_id="benchmark_ws")
            retrieval_query = RetrievalQuery(
                query_text=q.query_text,
                metadata_filter=m_filter,
                top_k=top_k,
            )

            t0 = time.perf_counter()
            results: list[RetrievalResult] = await retriever.retrieve(retrieval_query)
            duration_ms = (time.perf_counter() - t0) * 1000.0
            total_latency_ms += duration_ms

            # Metric calculations
            total_mrr += calculate_mrr(results, q.relevant_chunk_ids)
            for k in k_values:
                total_recall[k] += calculate_recall_at_k(results, q.relevant_chunk_ids, k)
                total_ndcg[k] += calculate_ndcg_at_k(results, q.relevant_chunk_ids, k)

            # Check for agentic statistics
            is_fallback = any(r.metadata.get("agentic_fallback", False) for r in results)
            if is_fallback:
                fallback_queries_count += 1

            if isinstance(retriever, AgenticRetriever) and retriever.last_budget_usage is not None:
                total_llm_calls += retriever.last_budget_usage.llm_calls
                total_retrieval_hops += max(1, retriever.last_budget_usage.retrieval_hops)
            else:
                total_retrieval_hops += 1

        metrics = AgenticBenchmarkMetrics(
            retriever_name=retriever_name,
            sample_size=num_queries,
            mrr=total_mrr / num_queries,
            average_latency_ms=total_latency_ms / num_queries,
            retrieval_calls_per_query=total_retrieval_hops / num_queries,
            llm_calls_per_query=total_llm_calls / num_queries,
            fallback_rate=fallback_queries_count / num_queries,
        )

        for k in k_values:
            metrics.recall_at_k[k] = total_recall[k] / num_queries
            metrics.ndcg_at_k[k] = total_ndcg[k] / num_queries

        return metrics

    async def compare(
        self,
        adaptive_retriever: BaseRetriever,
        agentic_retriever: BaseRetriever,
        k_values: Sequence[int] | None = None,
        top_k: int = 20,
    ) -> AgenticComparativeReport:
        """Run benchmark on both Adaptive and Agentic retrievers and produce a comparative report."""
        adaptive_metrics = await self.evaluate_retriever(
            retriever=adaptive_retriever,
            retriever_name="AdaptiveRetriever",
            k_values=k_values,
            top_k=top_k,
        )

        agentic_metrics = await self.evaluate_retriever(
            retriever=agentic_retriever,
            retriever_name="AgenticRetriever",
            k_values=k_values,
            top_k=top_k,
        )

        return AgenticComparativeReport(
            dataset_id=self.dataset.dataset_id,
            adaptive_metrics=adaptive_metrics,
            agentic_metrics=agentic_metrics,
        )
