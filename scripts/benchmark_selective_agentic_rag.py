"""Selective Agentic RAG Escalation Benchmark CLI.

Evaluates:
  A. Adaptive baseline (cheap deterministic)
  B. Always-Agentic (bounded LangGraph state machine)
  C. Selective Agentic (deterministic escalation controller)

Across both frozen benchmarks:
  1. Hard-Case Retrieval-Trap Benchmark (50 queries)
  2. Multi-Hop & Adversarial Benchmark (50 queries)

Usage:
    python scripts/benchmark_selective_agentic_rag.py
"""

import asyncio
import os
import sys
from typing import Any

# Ensure workspace src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from competitive_intelligence.agentic_rag.evaluation import (
    get_multihop_evaluation_dataset,
    get_retrieval_trap_dataset,
)
from competitive_intelligence.agentic_rag.selective import (
    EscalationPolicyConfig,
    ResultSelectionStrategy,
    SelectiveAgenticEvaluator,
    SelectiveBenchmarkReport,
    SelectiveRetrieverConfig,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.lexical.bm25_index import BM25Index
from competitive_intelligence.retrieval_engine.lexical.bm25_retriever import BM25Retriever
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
)
from competitive_intelligence.retrieval_engine.routing.adaptive_retriever import (
    AdaptiveRetriever,
)
from competitive_intelligence.retrieval_engine.routing.models import RouterConfiguration
from competitive_intelligence.retrieval_engine.routing.router import (
    DeterministicRetrievalRouter,
)


class BM25BaseRetrieverAdapter(BaseRetriever):
    """Adapter wrapping BM25Retriever to conform to BaseRetriever interface."""

    def __init__(self, bm25: BM25Retriever) -> None:
        self.bm25 = bm25

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return await self.bm25.aretrieve(query=query.query_text, top_k=query.top_k)


def print_comparative_table(report: SelectiveBenchmarkReport) -> None:
    """Print 3-way comparative metrics table."""
    print("=" * 115)
    print(f"  3-WAY BENCHMARK REPORT: {report.dataset_name.upper()} ({report.total_queries} Queries)")
    print("=" * 115)
    print(
        f"{'Metric':<28} | {'(A) Adaptive':<16} | {'(B) Always-Agentic':<18} | {'(C) Selective':<16} | {'Selective vs Always':<18}"
    )
    print("-" * 115)

    ad = report.adaptive_metrics
    ag = report.agentic_metrics
    se = report.selective_metrics

    rows = [
        ("Recall@5", f"{ad.recall_at_5 * 100:.1f}%", f"{ag.recall_at_5 * 100:.1f}%", f"{se.recall_at_5 * 100:.1f}%", f"{(se.recall_at_5 - ag.recall_at_5) * 100:+.1f}%"),
        ("Recall@10", f"{ad.recall_at_10 * 100:.1f}%", f"{ag.recall_at_10 * 100:.1f}%", f"{se.recall_at_10 * 100:.1f}%", f"{(se.recall_at_10 - ag.recall_at_10) * 100:+.1f}%"),
        ("Recall@20", f"{ad.recall_at_20 * 100:.1f}%", f"{ag.recall_at_20 * 100:.1f}%", f"{se.recall_at_20 * 100:.1f}%", f"{(se.recall_at_20 - ag.recall_at_20) * 100:+.1f}%"),
        ("MRR", f"{ad.mrr:.4f}", f"{ag.mrr:.4f}", f"{se.mrr:.4f}", f"{se.mrr - ag.mrr:+.4f}"),
        ("NDCG@10", f"{ad.ndcg_at_10:.4f}", f"{ag.ndcg_at_10:.4f}", f"{se.ndcg_at_10:.4f}", f"{se.ndcg_at_10 - ag.ndcg_at_10:+.4f}"),
        ("Mean Latency", f"{ad.mean_latency_ms:.1f} ms", f"{ag.mean_latency_ms:.1f} ms", f"{se.mean_latency_ms:.1f} ms", f"{(se.mean_latency_ms - ag.mean_latency_ms) / ag.mean_latency_ms * 100:+.1f}%"),
        ("Median Latency (p50)", f"{ad.p50_latency_ms:.1f} ms", f"{ag.p50_latency_ms:.1f} ms", f"{se.p50_latency_ms:.1f} ms", f"{(se.p50_latency_ms - ag.p50_latency_ms) / ag.p50_latency_ms * 100:+.1f}%"),
        ("p95 Latency", f"{ad.p95_latency_ms:.1f} ms", f"{ag.p95_latency_ms:.1f} ms", f"{se.p95_latency_ms:.1f} ms", f"{(se.p95_latency_ms - ag.p95_latency_ms) / ag.p95_latency_ms * 100:+.1f}%"),
        ("Total LLM Calls", f"{ad.total_llm_calls}", f"{ag.total_llm_calls}", f"{se.total_llm_calls}", f"-{report.routing_summary.llm_overhead_reduction_pct:.1f}%"),
        ("Mean LLM Calls / Query", f"{ad.mean_llm_calls_per_query:.2f}", f"{ag.mean_llm_calls_per_query:.2f}", f"{se.mean_llm_calls_per_query:.2f}", f"{(se.mean_llm_calls_per_query - ag.mean_llm_calls_per_query):+.2f}"),
        ("Mean Retrieval Hops", f"{ad.mean_hops_per_query:.2f}", f"{ag.mean_hops_per_query:.2f}", f"{se.mean_hops_per_query:.2f}", f"{(se.mean_hops_per_query - ag.mean_hops_per_query):+.2f}"),
    ]

    for title, v_ad, v_ag, v_se, v_diff in rows:
        print(f"{title:<28} | {v_ad:<16} | {v_ag:<18} | {v_se:<16} | {v_diff:<18}")

    print("-" * 115)


def print_routing_summary(report: SelectiveBenchmarkReport) -> None:
    """Print detailed routing precision, recall, and overhead reduction statistics."""
    rs = report.routing_summary
    print("\n" + "=" * 115)
    print("  SELECTIVE ROUTING ANALYSIS & RECOVERY METRICS")
    print("=" * 115)
    print(f"  - Total Queries Evaluated                : {rs.total_queries}")
    print(f"  - Escalated Queries                      : {rs.escalated_count} ({rs.escalation_rate * 100:.1f}%)")
    print(f"  - Fast-Path (Adaptive-Only)              : {rs.total_queries - rs.escalated_count} ({(1 - rs.escalation_rate) * 100:.1f}%)")
    print("-" * 115)
    print(f"  - Adaptive Misses (Ground-truth deficit) : {rs.adaptive_misses}")
    print(f"  - Agentic Recoveries (on Adaptive misses): {rs.agentic_recoveries}")
    print(f"  - Selective Recoveries (Successful)      : {rs.selective_recoveries}")
    print(f"  - Agentic Recovery Retained              : {rs.agentic_recovery_retained} / {rs.agentic_recoveries} ({rs.agentic_recovery_retained / rs.agentic_recoveries * 100:.1f}%)" if rs.agentic_recoveries else f"  - Agentic Recovery Retained              : 0")
    print(f"  - Agentic Recovery Rejected (Rank dilut.): {rs.agentic_recovery_rejected}")
    print(f"  - Unnecessary Escalations                : {rs.unnecessary_invocations} ({rs.unnecessary_escalation_rate * 100:.1f}%)")
    print(f"  - Missed Escalations (False Negatives)   : {rs.total_queries - rs.escalated_count - (rs.total_queries - rs.escalated_count - (rs.adaptive_misses - rs.selective_recoveries))} ({rs.missed_escalation_rate * 100:.1f}%)")
    print("-" * 115)
    print(f"  - Escalation Precision                   : {rs.escalation_precision * 100:.1f}% (Escalations that yielded evidence recovery)")
    print(f"  - Escalation Recall                      : {rs.escalation_recall * 100:.1f}% (Adaptive misses caught by escalation)")
    print(f"  - LLM Overhead Reduction                : {rs.llm_overhead_reduction_pct:.1f}% (Fewer LLM invocations vs Always-Agentic)")
    print(f"  - Latency Overhead Reduction            : {rs.latency_overhead_reduction_pct:.1f}% (Mean latency improvement vs Always-Agentic)")

    print("-" * 115)


def print_category_table(report: SelectiveBenchmarkReport) -> None:
    """Print category breakdown table."""
    print("\n" + "=" * 115)
    print("  CATEGORY-LEVEL ESCALATION & RECALL BREAKDOWN")
    print("=" * 115)
    print(
        f"{'Category':<32} | {'Count':<5} | {'Escalated':<9} | {'Esc Rate':<8} | {'Ad R@10':<8} | {'Ag R@10':<8} | {'Se R@10':<8} | {'LLM Calls (Se/Ag)':<18}"
    )
    print("-" * 115)

    for cat_name, stats in report.category_metrics.items():
        print(
            f"{cat_name:<32} | {stats['count']:<5} | {stats['escalated']:<9} | {stats['escalation_rate'] * 100:>6.1f}% | "
            f"{stats['adaptive_recall_at_10'] * 100:>6.1f}% | {stats['agentic_recall_at_10'] * 100:>6.1f}% | {stats['selective_recall_at_10'] * 100:>6.1f}% | "
            f"{stats['selective_total_llm']:>7} / {stats['agentic_total_llm']:<8}"
        )
    print("-" * 115)


async def evaluate_trap_dataset() -> SelectiveBenchmarkReport:
    """Run selective evaluation on Hard-Case Retrieval-Trap dataset."""
    dataset = get_retrieval_trap_dataset()
    index = BM25Index()
    index.build_index(dataset.corpus)

    bm25_core = BM25Retriever(index=index)
    bm25_retriever = BM25BaseRetrieverAdapter(bm25=bm25_core)

    router_config = RouterConfiguration(
        known_entities=[
            "Snowflake", "Databricks", "OpenAI", "Anthropic", "Google", "Microsoft",
            "Qdrant", "Pinecone", "Meta", "Mistral", "Apple", "xAI", "Stripe", "Adyen", "Cloudflare",
        ],
    )
    router = DeterministicRetrievalRouter(config=router_config)

    adaptive_retriever = AdaptiveRetriever(
        router=router,
        bm25_retriever=bm25_retriever,
        semantic_retriever=bm25_retriever,
        hybrid_retriever=bm25_retriever,
        reranked_retriever=bm25_retriever,
    )

    policy_config = EscalationPolicyConfig(
        min_score_threshold=0.10,
        min_evidence_count=1,
        min_entity_coverage=0.99,
        min_facet_coverage=0.99,
    )

    selective_config = SelectiveRetrieverConfig(
        policy_config=policy_config,
        selection_strategy=ResultSelectionStrategy.PRESERVE_BETTER,
        final_top_k=20,
    )

    evaluator = SelectiveAgenticEvaluator(
        adaptive_retriever=adaptive_retriever,
        policy_config=policy_config,
        selective_config=selective_config,
    )

    report = await evaluator.evaluate_dataset(
        dataset=dataset,
        dataset_name="Retrieval-Trap Benchmark (50 Hard-Case Queries)",
    )
    return report


async def evaluate_multihop_dataset() -> SelectiveBenchmarkReport:
    """Run selective evaluation on Multi-Hop dataset."""
    dataset = get_multihop_evaluation_dataset()
    index = BM25Index()
    index.build_index(dataset.corpus)

    bm25_core = BM25Retriever(index=index)
    bm25_retriever = BM25BaseRetrieverAdapter(bm25=bm25_core)

    router_config = RouterConfiguration(
        known_entities=[
            "Snowflake", "Databricks", "OpenAI", "Anthropic", "Google", "Microsoft",
            "Qdrant", "Pinecone", "Meta", "Mistral", "Apple", "xAI", "Stripe", "Adyen", "Cloudflare",
        ],
    )
    router = DeterministicRetrievalRouter(config=router_config)

    adaptive_retriever = AdaptiveRetriever(
        router=router,
        bm25_retriever=bm25_retriever,
        semantic_retriever=bm25_retriever,
        hybrid_retriever=bm25_retriever,
        reranked_retriever=bm25_retriever,
    )

    policy_config = EscalationPolicyConfig(
        min_score_threshold=0.10,
        min_evidence_count=1,
        min_entity_coverage=0.99,
        min_facet_coverage=0.99,
    )

    selective_config = SelectiveRetrieverConfig(
        policy_config=policy_config,
        selection_strategy=ResultSelectionStrategy.PRESERVE_BETTER,
        final_top_k=20,
    )

    evaluator = SelectiveAgenticEvaluator(
        adaptive_retriever=adaptive_retriever,
        policy_config=policy_config,
        selective_config=selective_config,
    )

    report = await evaluator.evaluate_dataset(
        dataset=dataset,
        dataset_name="Multi-Hop & Adversarial Benchmark (50 Queries)",
    )
    return report


async def main() -> None:
    """Run both benchmarks and present comprehensive selective escalation reports."""
    print("=" * 115)
    print("  SELECTIVE AGENTIC RAG ESCALATION BENCHMARK EVALUATION")
    print("=" * 115)

    print("\n[Evaluating Benchmark 1: Retrieval-Trap Dataset (50 Hard-Case Queries)]...")
    trap_report = await evaluate_trap_dataset()
    print_comparative_table(trap_report)
    print_routing_summary(trap_report)
    print_category_table(trap_report)

    print("\n[Evaluating Benchmark 2: Multi-Hop Dataset (50 Queries)]...")
    multihop_report = await evaluate_multihop_dataset()
    print_comparative_table(multihop_report)
    print_routing_summary(multihop_report)
    print_category_table(multihop_report)


if __name__ == "__main__":
    asyncio.run(main())
