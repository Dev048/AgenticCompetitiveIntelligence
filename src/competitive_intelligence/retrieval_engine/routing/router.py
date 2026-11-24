"""Deterministic query router for Adaptive Retrieval.

Evaluates extracted query features against prioritized rule heuristics to determine
the optimal retrieval strategy, candidate pool size, and reranker usage.
"""

from competitive_intelligence.retrieval_engine.interfaces import BaseRetrievalRouter
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalStrategy,
)
from competitive_intelligence.retrieval_engine.routing.feature_extractor import (
    QueryFeatureExtractor,
)
from competitive_intelligence.retrieval_engine.routing.models import (
    QueryFeatures,
    RouterConfiguration,
    RoutingDecision,
)


class DeterministicRetrievalRouter(BaseRetrievalRouter):
    """Deterministic, rule-based query router for adaptive retrieval pipelines.

    Evaluates linguistic and structural signals (exact versions, identifiers, currencies,
    known entities, comparison terms, conceptual keywords) to select strategy and budget.
    """

    def __init__(
        self,
        config: RouterConfiguration | None = None,
        feature_extractor: QueryFeatureExtractor | None = None,
    ) -> None:
        """Initialize the router with optional configuration and feature extractor.

        Args:
            config: Configuration settings and threshold parameters.
            feature_extractor: Extractor instance for parsing query features.
        """
        self.config = config if config is not None else RouterConfiguration()
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else QueryFeatureExtractor(known_entities=self.config.known_entities)
        )

    def route(self, query: str | RetrievalQuery) -> RoutingDecision:
        """Analyze query and return a deterministic RoutingDecision based on prioritized rules.

        Args:
            query: Raw query string or RetrievalQuery model.

        Returns:
            RoutingDecision containing strategy, budget, reranking flag, and auditable rationale.
        """
        query_text = query.query_text if isinstance(query, RetrievalQuery) else query
        features = self.feature_extractor.extract(
            query=query_text,
            known_entities_override=self.config.known_entities,
        )

        return self._evaluate_rules(query_text=query_text, features=features)

    def _evaluate_rules(
        self,
        query_text: str,
        features: QueryFeatures,
    ) -> RoutingDecision:
        """Evaluate prioritized decision tree on extracted query features."""
        # ----------------------------------------------------------------------
        # Priority 1: Multi-Entity Comparison Rule
        # ----------------------------------------------------------------------
        if features.has_comparison or len(features.known_entities) >= 2:
            return RoutingDecision(
                query_text=query_text,
                strategy=RetrievalStrategy.HYBRID_RRF,
                use_reranker=True,
                candidate_pool_size=self.config.comparison_candidate_pool_size,
                final_top_k=self.config.default_top_k,
                rule_name="MultiEntityComparisonRule",
                rationale=(
                    "Multi-entity or comparative query detected; requires wide candidate "
                    f"fusion ({self.config.comparison_candidate_pool_size}) and cross-encoder "
                    "reranking to resolve entity boundaries."
                ),
                features=features,
            )

        # ----------------------------------------------------------------------
        # Priority 2: Complex Identifier or Version Rule
        # (Identifier/version present alongside semantic, conceptual, temporal, or entity modifiers)
        # ----------------------------------------------------------------------
        has_id_or_version = features.has_version or features.has_identifier or features.has_quotes
        has_complex_modifiers = (
            features.has_conceptual_terms
            or features.has_temporal
            or features.has_numeric
            or len(features.known_entities) >= 1
            or features.token_count > self.config.short_lookup_max_tokens
        )

        if has_id_or_version and has_complex_modifiers:
            return RoutingDecision(
                query_text=query_text,
                strategy=RetrievalStrategy.HYBRID_RRF,
                use_reranker=True,
                candidate_pool_size=self.config.numeric_candidate_pool_size,
                final_top_k=self.config.default_top_k,
                rule_name="ComplexIdentifierOrVersionRule",
                rationale=(
                    "Query contains exact identifier/version alongside semantic, conceptual, "
                    "temporal, or entity constraints; requires hybrid retrieval and reranking."
                ),
                features=features,
            )

        # ----------------------------------------------------------------------
        # Priority 3: Short Exact Lookup Rule
        # (Pure lookup without conceptual/temporal/comparison modifiers)
        # ----------------------------------------------------------------------
        if (
            has_id_or_version
            and not features.has_conceptual_terms
            and not features.has_comparison
            and not features.has_temporal
            and features.token_count <= self.config.short_lookup_max_tokens
        ):
            return RoutingDecision(
                query_text=query_text,
                strategy=RetrievalStrategy.SPARSE_BM25,
                use_reranker=False,
                candidate_pool_size=self.config.short_lookup_candidate_pool_size,
                final_top_k=self.config.exact_lookup_top_k,
                rule_name="ShortExactLookupRule",
                rationale=(
                    "Short exact lookup query with strong identifier and no semantic/temporal "
                    "modifiers; sparse BM25 is sufficient without reranker overhead."
                ),
                features=features,
            )

        # ----------------------------------------------------------------------
        # Priority 4: Numeric / Pricing / Metric Rule
        # ----------------------------------------------------------------------
        if features.has_numeric and (
            len(features.known_entities) >= 1
            or features.has_conceptual_terms
            or features.token_count <= 8
        ):
            return RoutingDecision(
                query_text=query_text,
                strategy=RetrievalStrategy.HYBRID_RRF,
                use_reranker=True,
                candidate_pool_size=self.config.numeric_candidate_pool_size,
                final_top_k=self.config.numeric_top_k,
                rule_name="NumericMetricRule",
                rationale=(
                    "Numeric, pricing, or financial metric query; hybrid candidate retrieval "
                    "with cross-encoder reranking is required to prevent hard-negative value confusion."
                ),
                features=features,
            )

        # ----------------------------------------------------------------------
        # Priority 5: Pure Conceptual / Exploratory Rule
        # (Conceptual inquiry without identifiers, numbers, quotes, or comparisons)
        # ----------------------------------------------------------------------
        if (
            features.has_conceptual_terms
            and not features.has_version
            and not features.has_identifier
            and not features.has_numeric
            and not features.has_comparison
            and not features.has_quotes
        ):
            return RoutingDecision(
                query_text=query_text,
                strategy=RetrievalStrategy.DENSE_VECTOR,
                use_reranker=False,
                candidate_pool_size=self.config.conceptual_candidate_pool_size,
                final_top_k=self.config.default_top_k,
                rule_name="PureConceptualExploratoryRule",
                rationale=(
                    "Broad conceptual or thematic query without specific identifiers or metrics; "
                    "dense vector search captures semantic similarity directly."
                ),
                features=features,
            )

        # ----------------------------------------------------------------------
        # Priority 6: Default Ambiguity Fallback Rule
        # ----------------------------------------------------------------------
        return RoutingDecision(
            query_text=query_text,
            strategy=RetrievalStrategy.HYBRID_RRF,
            use_reranker=True,
            candidate_pool_size=self.config.fallback_candidate_pool_size,
            final_top_k=self.config.default_top_k,
            rule_name="DefaultFallbackRule",
            rationale=(
                "Ambiguous or mixed query signals; defaulting to balanced hybrid retrieval "
                "with cross-encoder reranking to ensure high recall."
            ),
            features=features,
        )
