"""Evaluation suite for query router decisions and cost/compute efficiency analysis."""

from typing import Any
from competitive_intelligence.retrieval_engine.evaluation.routing_models import (
    RoutingCostEstimateConfig,
    RoutingEvaluationDataset,
    RoutingEvaluationReport,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetrievalRouter
from competitive_intelligence.retrieval_engine.models import RetrievalStrategy


class RoutingEvaluator:
    """Evaluates query routing accuracy, rule precision, and compute savings against labeled benchmarks."""

    def __init__(self, cost_config: RoutingCostEstimateConfig | None = None) -> None:
        """Initialize the evaluator with optional cost estimation configuration."""
        self.cost_config = cost_config if cost_config is not None else RoutingCostEstimateConfig()

    def evaluate(
        self,
        router: BaseRetrievalRouter,
        dataset: RoutingEvaluationDataset,
        cost_config_override: RoutingCostEstimateConfig | None = None,
    ) -> RoutingEvaluationReport:
        """Evaluate a router against a RoutingEvaluationDataset.

        Args:
            router: BaseRetrievalRouter instance under test.
            dataset: Labeled ground-truth routing dataset.
            cost_config_override: Optional override for latency/compute cost settings.

        Returns:
            RoutingEvaluationReport containing accuracies, confusion matrix, and cost savings.
        """
        active_cost_config = (
            cost_config_override if cost_config_override is not None else self.cost_config
        )

        total_queries = len(dataset.queries)
        if total_queries == 0:
            raise ValueError("Evaluation dataset contains 0 queries")

        correct_overall = 0
        correct_strategy = 0
        correct_reranker = 0
        correct_candidate_pool = 0
        correct_final_top_k = 0
        correct_rule = 0

        # Initialize confusion matrix for all strategy combinations
        strategies = [s.value for s in RetrievalStrategy]
        confusion_matrix: dict[str, dict[str, int]] = {
            s1: {s2: 0 for s2 in strategies} for s1 in strategies
        }

        query_evaluations: list[dict[str, Any]] = []
        per_rule_data: dict[str, dict[str, int]] = {}
        per_strategy_data: dict[str, dict[str, int]] = {}
        per_category_data: dict[str, dict[str, int]] = {}

        total_routed_estimated_cost = 0.0

        for eq in dataset.queries:
            decision = router.route(eq.query_text)

            is_strategy_match = decision.strategy == eq.expected_strategy
            is_reranker_match = decision.use_reranker == eq.expected_use_reranker
            is_pool_match = decision.candidate_pool_size == eq.expected_candidate_pool_size
            is_top_k_match = decision.final_top_k == eq.expected_final_top_k
            is_rule_match = decision.rule_name == eq.expected_rule_name

            is_overall_match = (
                is_strategy_match
                and is_reranker_match
                and is_pool_match
                and is_top_k_match
                and is_rule_match
            )

            if is_overall_match:
                correct_overall += 1
            if is_strategy_match:
                correct_strategy += 1
            if is_reranker_match:
                correct_reranker += 1
            if is_pool_match:
                correct_candidate_pool += 1
            if is_top_k_match:
                correct_final_top_k += 1
            if is_rule_match:
                correct_rule += 1

            # Update confusion matrix
            confusion_matrix[eq.expected_strategy.value][decision.strategy.value] += 1

            # Compute estimated query latency/compute cost
            query_cost = self._estimate_query_cost(
                strategy=decision.strategy,
                use_reranker=decision.use_reranker,
                candidate_pool_size=decision.candidate_pool_size,
                cost_config=active_cost_config,
            )
            total_routed_estimated_cost += query_cost

            # Track per-rule stats
            if eq.expected_rule_name not in per_rule_data:
                per_rule_data[eq.expected_rule_name] = {"total": 0, "correct": 0}
            per_rule_data[eq.expected_rule_name]["total"] += 1
            if is_rule_match:
                per_rule_data[eq.expected_rule_name]["correct"] += 1

            # Track per-strategy stats
            exp_strat = eq.expected_strategy.value
            if exp_strat not in per_strategy_data:
                per_strategy_data[exp_strat] = {"total": 0, "correct": 0}
            per_strategy_data[exp_strat]["total"] += 1
            if is_strategy_match:
                per_strategy_data[exp_strat]["correct"] += 1

            # Track per-category stats
            cat = eq.category
            if cat not in per_category_data:
                per_category_data[cat] = {"total": 0, "correct": 0}
            per_category_data[cat]["total"] += 1
            if is_overall_match:
                per_category_data[cat]["correct"] += 1

            query_evaluations.append({
                "query_id": eq.query_id,
                "query_text": eq.query_text,
                "category": eq.category,
                "expected_rule": eq.expected_rule_name,
                "actual_rule": decision.rule_name,
                "expected_strategy": eq.expected_strategy.value,
                "actual_strategy": decision.strategy.value,
                "expected_use_reranker": eq.expected_use_reranker,
                "actual_use_reranker": decision.use_reranker,
                "expected_pool_size": eq.expected_candidate_pool_size,
                "actual_pool_size": decision.candidate_pool_size,
                "expected_final_top_k": eq.expected_final_top_k,
                "actual_final_top_k": decision.final_top_k,
                "is_overall_match": is_overall_match,
                "estimated_cost_ms": query_cost,
                "rationale": decision.rationale,
            })

        # Calculate per-rule accuracy
        per_rule_metrics: dict[str, dict[str, Any]] = {}
        for r_name, data in per_rule_data.items():
            per_rule_metrics[r_name] = {
                "total": data["total"],
                "correct": data["correct"],
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0.0,
            }

        # Calculate per-strategy accuracy
        per_strategy_metrics: dict[str, dict[str, Any]] = {}
        for s_name, data in per_strategy_data.items():
            per_strategy_metrics[s_name] = {
                "total": data["total"],
                "correct": data["correct"],
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0.0,
            }

        # Calculate per-category accuracy
        per_category_metrics: dict[str, dict[str, Any]] = {}
        for c_name, data in per_category_data.items():
            per_category_metrics[c_name] = {
                "total": data["total"],
                "correct": data["correct"],
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0.0,
            }

        # Cost and savings calculation
        avg_routed_cost = total_routed_estimated_cost / total_queries
        baseline_fixed_cost = active_cost_config.baseline_all_hybrid_rerank_cost_ms
        compute_savings_pct = (
            (1.0 - (avg_routed_cost / baseline_fixed_cost)) * 100.0
            if baseline_fixed_cost > 0
            else 0.0
        )

        return RoutingEvaluationReport(
            total_queries=total_queries,
            overall_accuracy=correct_overall / total_queries,
            strategy_accuracy=correct_strategy / total_queries,
            reranker_accuracy=correct_reranker / total_queries,
            candidate_pool_accuracy=correct_candidate_pool / total_queries,
            final_top_k_accuracy=correct_final_top_k / total_queries,
            rule_accuracy=correct_rule / total_queries,
            per_rule_metrics=per_rule_metrics,
            per_strategy_metrics=per_strategy_metrics,
            per_category_metrics=per_category_metrics,
            confusion_matrix=confusion_matrix,
            average_routed_estimated_cost=avg_routed_cost,
            baseline_fixed_estimated_cost=baseline_fixed_cost,
            estimated_compute_savings_pct=compute_savings_pct,
            query_evaluations=query_evaluations,
        )

    def _estimate_query_cost(
        self,
        strategy: RetrievalStrategy,
        use_reranker: bool,
        candidate_pool_size: int,
        cost_config: RoutingCostEstimateConfig,
    ) -> float:
        """Estimate the compute / latency cost for a given routing decision."""
        if strategy == RetrievalStrategy.SPARSE_BM25:
            return cost_config.bm25_cost_ms

        if strategy == RetrievalStrategy.DENSE_VECTOR:
            return cost_config.semantic_cost_ms

        if strategy in (RetrievalStrategy.HYBRID_RRF, RetrievalStrategy.HYBRID_WEIGHTED):
            if not use_reranker:
                return cost_config.hybrid_no_rerank_cost_ms
            if candidate_pool_size >= 30:
                return cost_config.hybrid_rerank_30_cost_ms
            return cost_config.hybrid_rerank_20_cost_ms

        return cost_config.hybrid_no_rerank_cost_ms
