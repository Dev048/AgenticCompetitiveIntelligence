"""Evaluator for Multi-Hop and Adversarial Agentic RAG benchmarking."""

import numpy as np
import time
from typing import Sequence

from competitive_intelligence.agentic_rag.evaluation.multihop_models import (
    BenchmarkCategory,
    CategoryMetrics,
    DivergenceClassification,
    MultiHopComparativeReport,
    MultiHopEvaluationDataset,
    MultiHopEvaluationQuery,
    QueryExecutionRecord,
)
from competitive_intelligence.agentic_rag.models import RetrievalDecision
from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.retrieval_engine.evaluation.metrics import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)


class MultiHopAgenticEvaluator:
    """Runs rigorous side-by-side evaluation of AdaptiveRetriever vs AgenticRetriever."""

    def __init__(self, dataset: MultiHopEvaluationDataset) -> None:
        self.dataset = dataset

    def _compute_percentiles(self, values: list[float]) -> tuple[float, float, float]:
        """Compute mean, median (p50), and p95 for a list of floats."""
        if not values:
            return 0.0, 0.0, 0.0
        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        p95_val = float(np.percentile(arr, 95))
        return mean_val, median_val, p95_val

    def _classify_divergence(
        self,
        query: MultiHopEvaluationQuery,
        ad_r10: float,
        ag_r10: float,
        ad_mrr: float,
        ag_mrr: float,
        llm_calls: int,
        latency_ms: float,
    ) -> tuple[DivergenceClassification, str]:
        """Deterministically classify divergence between Adaptive and Agentic results."""
        if query.is_negative_case or not query.relevant_chunk_ids:
            return (
                DivergenceClassification.NEUTRAL_NEGATIVE,
                "Negative/adversarial case without target evidence in corpus.",
            )

        if ad_r10 == 0.0 and ag_r10 == 0.0:
            return (
                DivergenceClassification.BOTH_FAILED,
                "Neither system retrieved sufficient relevant ground truth evidence.",
            )

        delta_r10 = ag_r10 - ad_r10
        delta_mrr = ag_mrr - ad_mrr

        # Material improvement threshold: >= 15% recall boost or >= 0.20 MRR boost with non-negative recall delta
        if delta_r10 >= 0.15 or (delta_r10 >= 0.0 and delta_mrr >= 0.20):
            return (
                DivergenceClassification.AGENTIC_SUCCESS,
                f"AgenticRetriever achieved materially higher evidence recall/ranking (Recall@10 {ag_r10:.2f} vs {ad_r10:.2f}, MRR {ag_mrr:.2f} vs {ad_mrr:.2f}).",
            )

        # Material degradation threshold
        if delta_r10 <= -0.15 or (delta_r10 <= 0.0 and delta_mrr <= -0.20):
            return (
                DivergenceClassification.ADAPTIVE_BETTER,
                f"AdaptiveRetriever achieved higher evidence recall/ranking (Recall@10 {ad_r10:.2f} vs {ag_r10:.2f}, MRR {ad_mrr:.2f} vs {ag_mrr:.2f}).",
            )

        # Equivalent retrieval quality with added reasoning overhead
        return (
            DivergenceClassification.AGENTIC_OVERHEAD,
            f"Equivalent evidence recall (Recall@10 {ag_r10:.2f} vs {ad_r10:.2f}), but Agentic incurred {llm_calls} LLM calls and {latency_ms:.1f}ms latency.",
        )

    async def evaluate_query(
        self,
        query: MultiHopEvaluationQuery,
        adaptive_retriever: BaseRetriever,
        agentic_retriever: BaseRetriever,
        top_k: int = 20,
    ) -> QueryExecutionRecord:
        """Run single query through both retrievers and record execution traces."""
        m_filter = query.metadata_filter or MetadataFilter(workspace_id="ws_multihop_eval")
        retrieval_query = RetrievalQuery(
            query_text=query.query_text,
            metadata_filter=m_filter,
            top_k=top_k,
        )

        # 1. Execute AdaptiveRetriever
        t0 = time.perf_counter()
        ad_results: list[RetrievalResult] = await adaptive_retriever.retrieve(retrieval_query)
        ad_lat = (time.perf_counter() - t0) * 1000.0

        ad_r5 = calculate_recall_at_k(ad_results, query.relevant_chunk_ids, 5)
        ad_r10 = calculate_recall_at_k(ad_results, query.relevant_chunk_ids, 10)
        ad_r20 = calculate_recall_at_k(ad_results, query.relevant_chunk_ids, 20)
        ad_mrr = calculate_mrr(ad_results, query.relevant_chunk_ids)
        ad_ndcg10 = calculate_ndcg_at_k(ad_results, query.relevant_chunk_ids, 10)
        ad_chunks = [r.chunk_id for r in ad_results]

        # 2. Execute AgenticRetriever
        t1 = time.perf_counter()
        ag_results: list[RetrievalResult] = await agentic_retriever.retrieve(retrieval_query)
        ag_lat = (time.perf_counter() - t1) * 1000.0

        ag_r5 = calculate_recall_at_k(ag_results, query.relevant_chunk_ids, 5)
        ag_r10 = calculate_recall_at_k(ag_results, query.relevant_chunk_ids, 10)
        ag_r20 = calculate_recall_at_k(ag_results, query.relevant_chunk_ids, 20)
        ag_mrr = calculate_mrr(ag_results, query.relevant_chunk_ids)
        ag_ndcg10 = calculate_ndcg_at_k(ag_results, query.relevant_chunk_ids, 10)
        ag_chunks = [r.chunk_id for r in ag_results]

        # 3. Extract Agentic Control Flow Metadata
        ag_decision = "direct_retrieval"
        ag_subqueries: list[str] = []
        ag_hops = 1
        ag_verifications_count = 0
        ag_corrective: str | None = None
        ag_is_fallback = any(r.metadata.get("agentic_fallback", False) for r in ag_results)
        ag_llm_calls = 0
        ag_retrieval_calls = 1

        if isinstance(agentic_retriever, AgenticRetriever):
            st = agentic_retriever.last_execution_state
            if st is not None:
                if st.query_analysis is not None:
                    ag_decision = st.query_analysis.decision.value
                    ag_subqueries = [sq.sub_query for sq in st.query_analysis.sub_queries]
                ag_hops = st.hops_executed
                ag_verifications_count = len(st.verifications)
                ag_corrective = st.corrective_query
                ag_is_fallback = st.is_fallback

            b_usage = agentic_retriever.last_budget_usage
            if b_usage is not None:
                ag_llm_calls = b_usage.llm_calls
                ag_retrieval_calls = max(1, b_usage.retrieval_hops)

        # 4. Classify Divergence
        classification, rationale = self._classify_divergence(
            query=query,
            ad_r10=ad_r10,
            ag_r10=ag_r10,
            ad_mrr=ad_mrr,
            ag_mrr=ag_mrr,
            llm_calls=ag_llm_calls,
            latency_ms=ag_lat,
        )

        return QueryExecutionRecord(
            query_id=query.query_id,
            query_text=query.query_text,
            category=query.category,
            relevant_chunk_ids=query.relevant_chunk_ids,
            distractor_chunk_ids=query.distractor_chunk_ids,
            adaptive_chunk_ids=ad_chunks,
            adaptive_recall_at_5=ad_r5,
            adaptive_recall_at_10=ad_r10,
            adaptive_recall_at_20=ad_r20,
            adaptive_mrr=ad_mrr,
            adaptive_ndcg_at_10=ad_ndcg10,
            adaptive_latency_ms=ad_lat,
            agentic_chunk_ids=ag_chunks,
            agentic_recall_at_5=ag_r5,
            agentic_recall_at_10=ag_r10,
            agentic_recall_at_20=ag_r20,
            agentic_mrr=ag_mrr,
            agentic_ndcg_at_10=ag_ndcg10,
            agentic_latency_ms=ag_lat,
            agentic_decision=ag_decision,
            agentic_subqueries=ag_subqueries,
            agentic_hops=ag_hops,
            agentic_verifications_count=ag_verifications_count,
            agentic_corrective_query=ag_corrective,
            agentic_is_fallback=ag_is_fallback,
            agentic_llm_calls=ag_llm_calls,
            agentic_retrieval_calls=ag_retrieval_calls,
            classification=classification,
            classification_rationale=rationale,
        )

    async def evaluate_all(
        self,
        adaptive_retriever: BaseRetriever,
        agentic_retriever: BaseRetriever,
        top_k: int = 20,
    ) -> MultiHopComparativeReport:
        """Run full evaluation suite across all benchmark queries and aggregate by category."""
        records: list[QueryExecutionRecord] = []

        for q in self.dataset.queries:
            rec = await self.evaluate_query(
                query=q,
                adaptive_retriever=adaptive_retriever,
                agentic_retriever=agentic_retriever,
                top_k=top_k,
            )
            records.append(rec)

        # Aggregate category-level metrics
        category_metrics_map: dict[str, CategoryMetrics] = {}
        for category in BenchmarkCategory:
            cat_records = [r for r in records if r.category == category]
            if not cat_records:
                continue

            n_cat = len(cat_records)
            ad_lat_mean, ad_lat_med, ad_lat_p95 = self._compute_percentiles([r.adaptive_latency_ms for r in cat_records])
            ag_lat_mean, ag_lat_med, ag_lat_p95 = self._compute_percentiles([r.agentic_latency_ms for r in cat_records])

            decomposed_cnt = sum(1 for r in cat_records if r.agentic_decision == "decompose_and_retrieve")
            direct_cnt = sum(1 for r in cat_records if r.agentic_decision == "direct_retrieval")
            
            # Unnecessary decomposition: categories that should be direct lookup or negative adversarial
            unnecessary_cnt = 0
            for r in cat_records:
                if r.category in [BenchmarkCategory.DIRECT_LOOKUP, BenchmarkCategory.ADVERSARIAL_NEGATIVE]:
                    if r.agentic_decision == "decompose_and_retrieve":
                        unnecessary_cnt += 1

            corrective_cnt = sum(1 for r in cat_records if r.agentic_hops > 1 or r.agentic_corrective_query is not None)
            fallback_cnt = sum(1 for r in cat_records if r.agentic_is_fallback)

            div_counts: dict[str, int] = {}
            for div in DivergenceClassification:
                div_counts[div.value] = sum(1 for r in cat_records if r.classification == div)

            cat_metric = CategoryMetrics(
                category=category,
                query_count=n_cat,
                adaptive_recall_at_5=sum(r.adaptive_recall_at_5 for r in cat_records) / n_cat,
                adaptive_recall_at_10=sum(r.adaptive_recall_at_10 for r in cat_records) / n_cat,
                adaptive_recall_at_20=sum(r.adaptive_recall_at_20 for r in cat_records) / n_cat,
                adaptive_mrr=sum(r.adaptive_mrr for r in cat_records) / n_cat,
                adaptive_ndcg_at_10=sum(r.adaptive_ndcg_at_10 for r in cat_records) / n_cat,
                adaptive_mean_latency_ms=ad_lat_mean,
                adaptive_median_latency_ms=ad_lat_med,
                adaptive_p95_latency_ms=ad_lat_p95,
                agentic_recall_at_5=sum(r.agentic_recall_at_5 for r in cat_records) / n_cat,
                agentic_recall_at_10=sum(r.agentic_recall_at_10 for r in cat_records) / n_cat,
                agentic_recall_at_20=sum(r.agentic_recall_at_20 for r in cat_records) / n_cat,
                agentic_mrr=sum(r.agentic_mrr for r in cat_records) / n_cat,
                agentic_ndcg_at_10=sum(r.agentic_ndcg_at_10 for r in cat_records) / n_cat,
                agentic_mean_latency_ms=ag_lat_mean,
                agentic_median_latency_ms=ag_lat_med,
                agentic_p95_latency_ms=ag_lat_p95,
                decomposition_rate=decomposed_cnt / n_cat,
                direct_retrieval_rate=direct_cnt / n_cat,
                unnecessary_decomposition_rate=unnecessary_cnt / n_cat,
                average_subqueries=sum(len(r.agentic_subqueries) for r in cat_records) / n_cat,
                average_hops=sum(r.agentic_hops for r in cat_records) / n_cat,
                corrective_retrieval_rate=corrective_cnt / n_cat,
                llm_calls_per_query=sum(r.agentic_llm_calls for r in cat_records) / n_cat,
                retrieval_calls_per_query=sum(r.agentic_retrieval_calls for r in cat_records) / n_cat,
                fallback_rate=fallback_cnt / n_cat,
                divergence_counts=div_counts,
            )
            category_metrics_map[category.value] = cat_metric

        # Overall metrics
        total_q = len(records)
        overall_ad_lat_mean, overall_ad_lat_med, overall_ad_lat_p95 = self._compute_percentiles([r.adaptive_latency_ms for r in records])
        overall_ag_lat_mean, overall_ag_lat_med, overall_ag_lat_p95 = self._compute_percentiles([r.agentic_latency_ms for r in records])

        overall_adaptive = {
            "recall_at_5": sum(r.adaptive_recall_at_5 for r in records) / total_q if total_q else 0.0,
            "recall_at_10": sum(r.adaptive_recall_at_10 for r in records) / total_q if total_q else 0.0,
            "recall_at_20": sum(r.adaptive_recall_at_20 for r in records) / total_q if total_q else 0.0,
            "mrr": sum(r.adaptive_mrr for r in records) / total_q if total_q else 0.0,
            "ndcg_at_10": sum(r.adaptive_ndcg_at_10 for r in records) / total_q if total_q else 0.0,
            "mean_latency_ms": overall_ad_lat_mean,
            "median_latency_ms": overall_ad_lat_med,
            "p95_latency_ms": overall_ad_lat_p95,
            "retrieval_calls_per_query": 1.0,
            "llm_calls_per_query": 0.0,
        }

        overall_agentic = {
            "recall_at_5": sum(r.agentic_recall_at_5 for r in records) / total_q if total_q else 0.0,
            "recall_at_10": sum(r.agentic_recall_at_10 for r in records) / total_q if total_q else 0.0,
            "recall_at_20": sum(r.agentic_recall_at_20 for r in records) / total_q if total_q else 0.0,
            "mrr": sum(r.agentic_mrr for r in records) / total_q if total_q else 0.0,
            "ndcg_at_10": sum(r.agentic_ndcg_at_10 for r in records) / total_q if total_q else 0.0,
            "mean_latency_ms": overall_ag_lat_mean,
            "median_latency_ms": overall_ag_lat_med,
            "p95_latency_ms": overall_ag_lat_p95,
            "decomposition_rate": sum(1 for r in records if r.agentic_decision == "decompose_and_retrieve") / total_q if total_q else 0.0,
            "direct_retrieval_rate": sum(1 for r in records if r.agentic_decision == "direct_retrieval") / total_q if total_q else 0.0,
            "average_subqueries": sum(len(r.agentic_subqueries) for r in records) / total_q if total_q else 0.0,
            "average_hops": sum(r.agentic_hops for r in records) / total_q if total_q else 0.0,
            "corrective_retrieval_rate": sum(1 for r in records if r.agentic_hops > 1 or r.agentic_corrective_query is not None) / total_q if total_q else 0.0,
            "retrieval_calls_per_query": sum(r.agentic_retrieval_calls for r in records) / total_q if total_q else 0.0,
            "llm_calls_per_query": sum(r.agentic_llm_calls for r in records) / total_q if total_q else 0.0,
            "fallback_rate": sum(1 for r in records if r.agentic_is_fallback) / total_q if total_q else 0.0,
        }

        overall_divergence: dict[str, int] = {}
        for div in DivergenceClassification:
            overall_divergence[div.value] = sum(1 for r in records if r.classification == div)

        return MultiHopComparativeReport(
            dataset_id=self.dataset.dataset_id,
            total_corpus_size=len(self.dataset.corpus),
            total_queries=total_q,
            overall_adaptive_metrics=overall_adaptive,
            overall_agentic_metrics=overall_agentic,
            category_metrics=category_metrics_map,
            divergence_distribution=overall_divergence,
            query_records=records,
        )
