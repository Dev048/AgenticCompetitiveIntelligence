"""Domain models for query routing evaluation and cost/efficiency benchmarking."""

from typing import Any
from pydantic import BaseModel, Field

from competitive_intelligence.retrieval_engine.models import RetrievalStrategy


class RoutingEvaluationQuery(BaseModel):
    """Ground-truth labeled query for evaluating query router decisions."""

    query_id: str
    query_text: str
    expected_strategy: RetrievalStrategy
    expected_use_reranker: bool
    expected_candidate_pool_size: int
    expected_final_top_k: int
    expected_rule_name: str
    category: str
    notes: str | None = None


class RoutingEvaluationDataset(BaseModel):
    """Dataset containing labeled queries for evaluating routing decisions."""

    dataset_id: str
    queries: list[RoutingEvaluationQuery]


class RoutingCostEstimateConfig(BaseModel):
    """Configurable latency/compute cost estimates for different retrieval tiers."""

    bm25_cost_ms: float = 0.5
    semantic_cost_ms: float = 400.0
    hybrid_no_rerank_cost_ms: float = 430.0
    hybrid_rerank_20_cost_ms: float = 8000.0
    hybrid_rerank_30_cost_ms: float = 11800.0
    baseline_all_hybrid_rerank_cost_ms: float = 11800.0  # Cost if every query used Fixed 30-pool Hybrid+Reranker


class RoutingEvaluationReport(BaseModel):
    """Comprehensive evaluation metrics and efficiency report for query routing."""

    total_queries: int
    overall_accuracy: float
    strategy_accuracy: float
    reranker_accuracy: float
    candidate_pool_accuracy: float
    final_top_k_accuracy: float
    rule_accuracy: float

    per_rule_metrics: dict[str, dict[str, Any]]
    per_strategy_metrics: dict[str, dict[str, Any]]
    per_category_metrics: dict[str, dict[str, Any]]
    confusion_matrix: dict[str, dict[str, int]]  # expected_strategy -> actual_strategy -> count

    # Cost and compute analysis
    average_routed_estimated_cost: float
    baseline_fixed_estimated_cost: float
    estimated_compute_savings_pct: float

    # Per-query evaluation records
    query_evaluations: list[dict[str, Any]] = Field(default_factory=list)
