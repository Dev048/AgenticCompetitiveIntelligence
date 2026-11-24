"""Evaluation engine and metrics models for Selective Agentic RAG Escalation."""

import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.evaluation.trap_evaluator import (
    create_trap_benchmark_mock_llm,
)
from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.agentic_rag.selective.controller import (
    SelectiveAgenticRetriever,
)
from competitive_intelligence.agentic_rag.selective.models import (
    EscalationPolicyConfig,
    SelectiveRetrieverConfig,
)
from competitive_intelligence.agentic_rag.selective.policy import (
    DeterministicEscalationPolicy,
)
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


class SelectiveSystemMetrics(BaseModel):
    """Aggregate IR and efficiency metrics for a retrieval system."""

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_llm_calls: int = 0
    mean_llm_calls_per_query: float = 0.0
    mean_hops_per_query: float = 1.0


class SelectiveRoutingSummary(BaseModel):
    """Core routing precision, recall, and overhead reduction statistics."""

    total_queries: int = 0
    escalated_count: int = 0
    escalation_rate: float = 0.0
    escalation_precision: float = 0.0
    escalation_recall: float = 0.0
    unnecessary_escalation_rate: float = 0.0
    missed_escalation_rate: float = 0.0

    adaptive_misses: int = 0
    agentic_recoveries: int = 0
    selective_recoveries: int = 0
    agentic_recovery_retained: int = 0
    agentic_recovery_rejected: int = 0
    unnecessary_invocations: int = 0

    llm_overhead_reduction_pct: float = 0.0
    latency_overhead_reduction_pct: float = 0.0


class SelectiveComparativeRecord(BaseModel):
    """Per-query 3-way comparative execution record."""

    query_id: str
    query_text: str
    category: str
    ground_truth_count: int

    adaptive_recall_at_10: float
    adaptive_ndcg_at_10: float
    adaptive_mrr: float
    adaptive_latency_ms: float

    agentic_recall_at_10: float
    agentic_ndcg_at_10: float
    agentic_mrr: float
    agentic_latency_ms: float
    agentic_llm_calls: int
    agentic_hops: int

    selective_escalated: bool
    selective_reasons: list[str] = Field(default_factory=list)
    selective_recall_at_10: float
    selective_ndcg_at_10: float
    selective_mrr: float
    selective_latency_ms: float
    selective_llm_calls: int
    selective_hops: int
    selective_final_source: str

    is_adaptive_miss: bool
    is_agentic_recovery: bool
    is_selective_recovery: bool
    is_unnecessary_escalation: bool
    is_missed_escalation: bool


class SelectiveBenchmarkReport(BaseModel):
    """Full 3-way comparative evaluation report."""

    dataset_name: str
    total_queries: int
    adaptive_metrics: SelectiveSystemMetrics
    agentic_metrics: SelectiveSystemMetrics
    selective_metrics: SelectiveSystemMetrics
    routing_summary: SelectiveRoutingSummary
    category_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    query_records: list[SelectiveComparativeRecord] = Field(default_factory=list)


class SelectiveAgenticEvaluator:
    """Evaluator comparing Adaptive baseline, Always-Agentic, and Selective Agentic retrieval."""

    def __init__(
        self,
        adaptive_retriever: BaseRetriever,
        llm_client: Any | None = None,
        policy_config: EscalationPolicyConfig | None = None,
        selective_config: SelectiveRetrieverConfig | None = None,
    ) -> None:
        """Initialize evaluator with underlying retrievers and configurations."""
        self.adaptive_retriever = adaptive_retriever
        self.mock_llm = llm_client or create_trap_benchmark_mock_llm()

        self.agentic_retriever = AgenticRetriever(
            executor_retriever=adaptive_retriever,
            llm_client=self.mock_llm,
            max_hops=3,
            max_llm_calls=5,
        )

        p_config = policy_config or EscalationPolicyConfig()
        s_config = selective_config or SelectiveRetrieverConfig(policy_config=p_config)
        self.policy = DeterministicEscalationPolicy(p_config)

        self.selective_retriever = SelectiveAgenticRetriever(
            adaptive_retriever=adaptive_retriever,
            agentic_retriever=self.agentic_retriever,
            policy=self.policy,
            config=s_config,
        )

    async def evaluate_dataset(
        self,
        dataset: Any,
        dataset_name: str = "Benchmark",
    ) -> SelectiveBenchmarkReport:
        """Run 3-way comparative evaluation across all queries in dataset."""
        records: list[SelectiveComparativeRecord] = []

        ad_r5_list, ad_r10_list, ad_r20_list, ad_mrr_list, ad_ndcg_list, ad_lat_list = [], [], [], [], [], []
        ag_r5_list, ag_r10_list, ag_r20_list, ag_mrr_list, ag_ndcg_list, ag_lat_list, ag_llm_list, ag_hop_list = [], [], [], [], [], [], [], []
        se_r5_list, se_r10_list, se_r20_list, se_mrr_list, se_ndcg_list, se_lat_list, se_llm_list, se_hop_list = [], [], [], [], [], [], [], []

        for query_item in dataset.queries:
            q_id: str = getattr(query_item, "query_id", "")
            q_text: str = getattr(query_item, "query_text", "")
            cat_val: Any = getattr(query_item, "category", "")
            category: str = cat_val.value if hasattr(cat_val, "value") else str(cat_val)
            relevant_ids: dict[str, int] = getattr(query_item, "relevant_chunk_ids", {})
            gt_count = len(relevant_ids)

            company_val = getattr(query_item, "company_filter", None)
            comp_ids = [company_val] if company_val else None

            retrieval_query = RetrievalQuery(
                query_text=q_text,
                top_k=20,
                metadata_filter=MetadataFilter(
                    workspace_id="ws-benchmark",
                    company_ids=comp_ids,
                ),
            )

            # 1. Run Adaptive Baseline
            t0 = time.perf_counter()
            adaptive_results: list[RetrievalResult] = await self.adaptive_retriever.retrieve(retrieval_query)
            ad_lat = (time.perf_counter() - t0) * 1000.0

            ad_r5 = calculate_recall_at_k(adaptive_results, relevant_ids, k=5)
            ad_r10 = calculate_recall_at_k(adaptive_results, relevant_ids, k=10)
            ad_r20 = calculate_recall_at_k(adaptive_results, relevant_ids, k=20)
            ad_mrr = calculate_mrr(adaptive_results, relevant_ids)
            ad_ndcg = calculate_ndcg_at_k(adaptive_results, relevant_ids, k=10)

            ad_r5_list.append(ad_r5)
            ad_r10_list.append(ad_r10)
            ad_r20_list.append(ad_r20)
            ad_mrr_list.append(ad_mrr)
            ad_ndcg_list.append(ad_ndcg)
            ad_lat_list.append(ad_lat)

            # 2. Run Always-Agentic
            t0 = time.perf_counter()
            agentic_results: list[RetrievalResult] = await self.agentic_retriever.retrieve(retrieval_query)
            ag_lat = (time.perf_counter() - t0) * 1000.0

            ag_r5 = calculate_recall_at_k(agentic_results, relevant_ids, k=5)
            ag_r10 = calculate_recall_at_k(agentic_results, relevant_ids, k=10)
            ag_r20 = calculate_recall_at_k(agentic_results, relevant_ids, k=20)
            ag_mrr = calculate_mrr(agentic_results, relevant_ids)
            ag_ndcg = calculate_ndcg_at_k(agentic_results, relevant_ids, k=10)

            ag_llm = 0
            if self.agentic_retriever.last_budget_usage:
                ag_llm = getattr(self.agentic_retriever.last_budget_usage, "llm_calls", 0)
            ag_hops = 1
            if self.agentic_retriever.last_execution_state:
                ag_hops = self.agentic_retriever.last_execution_state.hops_executed

            ag_r5_list.append(ag_r5)
            ag_r10_list.append(ag_r10)
            ag_r20_list.append(ag_r20)
            ag_mrr_list.append(ag_mrr)
            ag_ndcg_list.append(ag_ndcg)
            ag_lat_list.append(ag_lat)
            ag_llm_list.append(ag_llm)
            ag_hop_list.append(ag_hops)

            # 3. Run Selective Agentic Retriever
            t0 = time.perf_counter()
            selective_results: list[RetrievalResult] = await self.selective_retriever.retrieve(retrieval_query)
            se_lat = (time.perf_counter() - t0) * 1000.0

            se_r5 = calculate_recall_at_k(selective_results, relevant_ids, k=5)
            se_r10 = calculate_recall_at_k(selective_results, relevant_ids, k=10)
            se_r20 = calculate_recall_at_k(selective_results, relevant_ids, k=20)
            se_mrr = calculate_mrr(selective_results, relevant_ids)
            se_ndcg = calculate_ndcg_at_k(selective_results, relevant_ids, k=10)

            trace = self.selective_retriever.last_trace
            se_escalated = trace.escalated if trace else False
            se_reasons = trace.assessment.reasons if trace else []
            se_llm = trace.llm_calls if trace else 0
            se_hops = trace.retrieval_hops if trace else 1
            se_source = trace.final_source if trace else "adaptive"

            se_r5_list.append(se_r5)
            se_r10_list.append(se_r10)
            se_r20_list.append(se_r20)
            se_mrr_list.append(se_mrr)
            se_ndcg_list.append(se_ndcg)
            se_lat_list.append(se_lat)
            se_llm_list.append(se_llm)
            se_hop_list.append(se_hops)

            # Classification of routing outcomes (reconciling required_chunk_ids when defined)
            required_ids: list[str] = getattr(query_item, "required_chunk_ids", [])
            ad_cids = [r.chunk_id for r in adaptive_results[:10]]
            ag_cids = [r.chunk_id for r in agentic_results[:10]]
            se_cids = [r.chunk_id for r in selective_results[:10]]

            if required_ids:
                missing_required = [cid for cid in required_ids if cid not in ad_cids]
                is_ad_miss = len(missing_required) > 0
                is_ag_recovery = is_ad_miss and any(cid in ag_cids for cid in missing_required)
                is_se_recovery = is_ad_miss and any(cid in se_cids for cid in missing_required)
            else:
                is_ad_miss = (ad_r10 < 1.0) and (gt_count > 0)
                is_ag_recovery = is_ad_miss and (ag_r10 > ad_r10)
                is_se_recovery = is_ad_miss and (se_r10 > ad_r10)

            is_unnecessary = se_escalated and (not is_ad_miss or se_r10 <= ad_r10)
            is_missed = (not se_escalated) and is_ag_recovery


            records.append(
                SelectiveComparativeRecord(
                    query_id=q_id,
                    query_text=q_text,
                    category=category,
                    ground_truth_count=gt_count,
                    adaptive_recall_at_10=ad_r10,
                    adaptive_ndcg_at_10=ad_ndcg,
                    adaptive_mrr=ad_mrr,
                    adaptive_latency_ms=ad_lat,
                    agentic_recall_at_10=ag_r10,
                    agentic_ndcg_at_10=ag_ndcg,
                    agentic_mrr=ag_mrr,
                    agentic_latency_ms=ag_lat,
                    agentic_llm_calls=ag_llm,
                    agentic_hops=ag_hops,
                    selective_escalated=se_escalated,
                    selective_reasons=se_reasons,
                    selective_recall_at_10=se_r10,
                    selective_ndcg_at_10=se_ndcg,
                    selective_mrr=se_mrr,
                    selective_latency_ms=se_lat,
                    selective_llm_calls=se_llm,
                    selective_hops=se_hops,
                    selective_final_source=se_source,
                    is_adaptive_miss=is_ad_miss,
                    is_agentic_recovery=is_ag_recovery,
                    is_selective_recovery=is_se_recovery,
                    is_unnecessary_escalation=is_unnecessary,
                    is_missed_escalation=is_missed,
                )
            )

        total_q = len(records)
        adaptive_metrics = SelectiveSystemMetrics(
            recall_at_5=float(np.mean(ad_r5_list)) if ad_r5_list else 0.0,
            recall_at_10=float(np.mean(ad_r10_list)) if ad_r10_list else 0.0,
            recall_at_20=float(np.mean(ad_r20_list)) if ad_r20_list else 0.0,
            mrr=float(np.mean(ad_mrr_list)) if ad_mrr_list else 0.0,
            ndcg_at_10=float(np.mean(ad_ndcg_list)) if ad_ndcg_list else 0.0,
            mean_latency_ms=float(np.mean(ad_lat_list)) if ad_lat_list else 0.0,
            p50_latency_ms=float(np.percentile(ad_lat_list, 50)) if ad_lat_list else 0.0,
            p95_latency_ms=float(np.percentile(ad_lat_list, 95)) if ad_lat_list else 0.0,
            total_llm_calls=0,
            mean_llm_calls_per_query=0.0,
            mean_hops_per_query=1.0,
        )

        agentic_metrics = SelectiveSystemMetrics(
            recall_at_5=float(np.mean(ag_r5_list)) if ag_r5_list else 0.0,
            recall_at_10=float(np.mean(ag_r10_list)) if ag_r10_list else 0.0,
            recall_at_20=float(np.mean(ag_r20_list)) if ag_r20_list else 0.0,
            mrr=float(np.mean(ag_mrr_list)) if ag_mrr_list else 0.0,
            ndcg_at_10=float(np.mean(ag_ndcg_list)) if ag_ndcg_list else 0.0,
            mean_latency_ms=float(np.mean(ag_lat_list)) if ag_lat_list else 0.0,
            p50_latency_ms=float(np.percentile(ag_lat_list, 50)) if ag_lat_list else 0.0,
            p95_latency_ms=float(np.percentile(ag_lat_list, 95)) if ag_lat_list else 0.0,
            total_llm_calls=sum(ag_llm_list),
            mean_llm_calls_per_query=float(np.mean(ag_llm_list)) if ag_llm_list else 0.0,
            mean_hops_per_query=float(np.mean(ag_hop_list)) if ag_hop_list else 1.0,
        )

        selective_metrics = SelectiveSystemMetrics(
            recall_at_5=float(np.mean(se_r5_list)) if se_r5_list else 0.0,
            recall_at_10=float(np.mean(se_r10_list)) if se_r10_list else 0.0,
            recall_at_20=float(np.mean(se_r20_list)) if se_r20_list else 0.0,
            mrr=float(np.mean(se_mrr_list)) if se_mrr_list else 0.0,
            ndcg_at_10=float(np.mean(se_ndcg_list)) if se_ndcg_list else 0.0,
            mean_latency_ms=float(np.mean(se_lat_list)) if se_lat_list else 0.0,
            p50_latency_ms=float(np.percentile(se_lat_list, 50)) if se_lat_list else 0.0,
            p95_latency_ms=float(np.percentile(se_lat_list, 95)) if se_lat_list else 0.0,
            total_llm_calls=sum(se_llm_list),
            mean_llm_calls_per_query=float(np.mean(se_llm_list)) if se_llm_list else 0.0,
            mean_hops_per_query=float(np.mean(se_hop_list)) if se_hop_list else 1.0,
        )

        # Routing Statistics
        escalated_count = sum(1 for r in records if r.selective_escalated)
        escalation_rate = (escalated_count / total_q) if total_q > 0 else 0.0

        adaptive_misses = sum(1 for r in records if r.is_adaptive_miss)
        agentic_recoveries = sum(1 for r in records if r.is_agentic_recovery)
        selective_recoveries = sum(1 for r in records if r.is_selective_recovery)
        unnecessary_invocations = sum(1 for r in records if r.is_unnecessary_escalation)
        missed_escalations = sum(1 for r in records if r.is_missed_escalation)

        # Escalation Precision: (Recovered or Improved) / Escalated
        escalation_precision = (
            (selective_recoveries / escalated_count) if escalated_count > 0 else 1.0
        )
        # Escalation Recall: (Escalated Misses) / (Total Misses)
        escalated_misses = sum(1 for r in records if r.is_adaptive_miss and r.selective_escalated)
        escalation_recall = (
            (escalated_misses / adaptive_misses) if adaptive_misses > 0 else 1.0
        )

        unnecessary_rate = (
            (unnecessary_invocations / (total_q - adaptive_misses))
            if (total_q - adaptive_misses) > 0
            else 0.0
        )
        missed_rate = (
            (missed_escalations / adaptive_misses) if adaptive_misses > 0 else 0.0
        )

        # Retained vs Rejected
        retained = sum(1 for r in records if r.is_agentic_recovery and r.is_selective_recovery)
        rejected = sum(1 for r in records if r.is_agentic_recovery and not r.is_selective_recovery)

        # Overhead Reductions
        llm_reduction = (
            ((agentic_metrics.total_llm_calls - selective_metrics.total_llm_calls) / agentic_metrics.total_llm_calls * 100.0)
            if agentic_metrics.total_llm_calls > 0
            else 0.0
        )
        lat_reduction = (
            ((agentic_metrics.mean_latency_ms - selective_metrics.mean_latency_ms) / agentic_metrics.mean_latency_ms * 100.0)
            if agentic_metrics.mean_latency_ms > 0
            else 0.0
        )

        routing_summary = SelectiveRoutingSummary(
            total_queries=total_q,
            escalated_count=escalated_count,
            escalation_rate=escalation_rate,
            escalation_precision=escalation_precision,
            escalation_recall=escalation_recall,
            unnecessary_escalation_rate=unnecessary_rate,
            missed_escalation_rate=missed_rate,
            adaptive_misses=adaptive_misses,
            agentic_recoveries=agentic_recoveries,
            selective_recoveries=selective_recoveries,
            agentic_recovery_retained=retained,
            agentic_recovery_rejected=rejected,
            unnecessary_invocations=unnecessary_invocations,
            llm_overhead_reduction_pct=llm_reduction,
            latency_overhead_reduction_pct=lat_reduction,
        )

        # Category Breakdowns
        category_metrics: dict[str, dict[str, Any]] = {}
        categories = sorted(list({r.category for r in records}))
        for cat in categories:
            cat_recs = [r for r in records if r.category == cat]
            cat_count = len(cat_recs)
            cat_esc = sum(1 for r in cat_recs if r.selective_escalated)
            cat_ad_r10 = float(np.mean([r.adaptive_recall_at_10 for r in cat_recs]))
            cat_ag_r10 = float(np.mean([r.agentic_recall_at_10 for r in cat_recs]))
            cat_se_r10 = float(np.mean([r.selective_recall_at_10 for r in cat_recs]))
            cat_se_llm = sum(r.selective_llm_calls for r in cat_recs)
            cat_ag_llm = sum(r.agentic_llm_calls for r in cat_recs)

            category_metrics[cat] = {
                "count": cat_count,
                "escalated": cat_esc,
                "escalation_rate": (cat_esc / cat_count) if cat_count > 0 else 0.0,
                "adaptive_recall_at_10": cat_ad_r10,
                "agentic_recall_at_10": cat_ag_r10,
                "selective_recall_at_10": cat_se_r10,
                "selective_total_llm": cat_se_llm,
                "agentic_total_llm": cat_ag_llm,
            }

        return SelectiveBenchmarkReport(
            dataset_name=dataset_name,
            total_queries=total_q,
            adaptive_metrics=adaptive_metrics,
            agentic_metrics=agentic_metrics,
            selective_metrics=selective_metrics,
            routing_summary=routing_summary,
            category_metrics=category_metrics,
            query_records=records,
        )
