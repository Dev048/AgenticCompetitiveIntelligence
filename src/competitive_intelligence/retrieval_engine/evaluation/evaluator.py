import time
from typing import Any

from competitive_intelligence.retrieval_engine.interfaces import BaseRetrievalEvaluator
from competitive_intelligence.retrieval_engine.models import RetrievalEvaluationMetrics
from competitive_intelligence.retrieval_engine.evaluation.models import EvaluationDataset
from competitive_intelligence.retrieval_engine.evaluation.metrics import (
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
)


class RetrievalEvaluator(BaseRetrievalEvaluator):
    """Evaluates a pre-configured retriever against an EvaluationDataset using standard IR metrics.
    
    The evaluator expects a prepared retriever that exposes a `retrieve(query_text: str)` method,
    returning a list of objects that have at least a `chunk_id` attribute (e.g., RetrievalResult 
    or VectorSearchResult).
    """

    def __init__(self, dataset: EvaluationDataset):
        """Initialize the evaluator with a self-contained EvaluationDataset."""
        self.dataset = dataset

    async def evaluate(
        self,
        test_dataset_path: str,
        retriever: Any,
        k_values: list[int] | None = None,
        warm_up: bool = True,
    ) -> RetrievalEvaluationMetrics:
        """Run the evaluation across all queries in the dataset.

        Args:
            test_dataset_path: Path identifier (ignored since dataset is provided via __init__).
            retriever: The initialized and prepared retriever to evaluate.
            k_values: List of K limits to evaluate (e.g., [5, 10, 20]). Defaults to [5, 10, 20, 50].
            warm_up: If True, runs a single dummy query to warm up models before measuring latency.

        Returns:
            RetrievalEvaluationMetrics populated with macro-averaged Recall@K, MRR, NDCG@K,
            and average latency.
        """
        if k_values is None:
            k_values = [5, 10, 20, 50]

        if not self.dataset.queries:
            return RetrievalEvaluationMetrics(sample_size=0)

        # Warm-up pass to exclude initialization costs from latency
        if warm_up:
            try:
                retriever.retrieve(self.dataset.queries[0].query_text)
            except Exception:
                pass  # Ignore warm-up errors

        total_mrr = 0.0
        total_recall = {k: 0.0 for k in k_values}
        total_ndcg = {k: 0.0 for k in k_values}
        total_latency_ms = 0.0
        num_queries = len(self.dataset.queries)

        for query in self.dataset.queries:
            # Measure end-to-end wall-clock latency
            start_time = time.perf_counter()
            results = retriever.retrieve(query.query_text)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            total_latency_ms += latency_ms

            # The results might be VectorSearchResult or RetrievalResult.
            # Both have a chunk_id property.
            
            # MRR is not K-dependent
            total_mrr += calculate_mrr(results, query.relevant_chunk_ids)

            for k in k_values:
                total_recall[k] += calculate_recall_at_k(results, query.relevant_chunk_ids, k)
                total_ndcg[k] += calculate_ndcg_at_k(results, query.relevant_chunk_ids, k)

        # Macro-average across all queries
        metrics = RetrievalEvaluationMetrics(sample_size=num_queries)
        metrics.mrr = total_mrr / num_queries
        metrics.average_latency_ms = total_latency_ms / num_queries
        
        for k in k_values:
            metrics.recall_at_k[k] = total_recall[k] / num_queries
            metrics.ndcg_at_k[k] = total_ndcg[k] / num_queries

        return metrics
