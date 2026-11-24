"""Unit tests for DeterministicRetrievalRouter across prioritized rules and conflicting signals."""

import pytest

from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalStrategy,
)
from competitive_intelligence.retrieval_engine.routing.models import (
    RouterConfiguration,
)
from competitive_intelligence.retrieval_engine.routing.router import (
    DeterministicRetrievalRouter,
)


@pytest.fixture
def router() -> DeterministicRetrievalRouter:
    config = RouterConfiguration(
        known_entities=["Snowflake", "Databricks", "Microsoft", "OpenAI", "Acme"],
        short_lookup_candidate_pool_size=10,
        numeric_candidate_pool_size=20,
        comparison_candidate_pool_size=30,
        conceptual_candidate_pool_size=20,
        fallback_candidate_pool_size=20,
        exact_lookup_top_k=5,
        numeric_top_k=5,
        default_top_k=10,
        short_lookup_max_tokens=5,
    )
    return DeterministicRetrievalRouter(config=config)


# ==============================================================================
# 1. Short Exact Lookups (BM25 Only)
# ==============================================================================


def test_short_version_lookup(router: DeterministicRetrievalRouter) -> None:
    """Test short version lookup routes to BM25 without reranking."""
    decision = router.route("v2.4.1")

    assert decision.strategy == RetrievalStrategy.SPARSE_BM25
    assert not decision.use_reranker
    assert decision.candidate_pool_size == 10
    assert decision.final_top_k == 5
    assert decision.rule_name == "ShortExactLookupRule"
    assert "Short exact lookup" in decision.rationale


def test_short_interrogative_version_lookup(router: DeterministicRetrievalRouter) -> None:
    """Test short question lookup for a version routes to BM25 (e.g. 'What is v2.4.1?')."""
    decision = router.route("What is v2.4.1?")

    assert decision.strategy == RetrievalStrategy.SPARSE_BM25
    assert not decision.use_reranker
    assert decision.candidate_pool_size == 10
    assert decision.final_top_k == 5
    assert decision.rule_name == "ShortExactLookupRule"


def test_short_error_code_identifier(router: DeterministicRetrievalRouter) -> None:
    """Test exact error code/SKU routes to BM25."""
    decision = router.route("ERR_AUTH_TIMEOUT_500")

    assert decision.strategy == RetrievalStrategy.SPARSE_BM25
    assert not decision.use_reranker
    assert decision.candidate_pool_size == 10
    assert decision.final_top_k == 5
    assert decision.rule_name == "ShortExactLookupRule"


def test_short_quoted_exact_lookup(router: DeterministicRetrievalRouter) -> None:
    """Test short quoted phrase lookup routes to BM25."""
    decision = router.route('"session timeout"')

    assert decision.strategy == RetrievalStrategy.SPARSE_BM25
    assert not decision.use_reranker
    assert decision.rule_name == "ShortExactLookupRule"


# ==============================================================================
# 2. Conflicting Signals: Identifier + Conceptual / Temporal / Question
# ==============================================================================


def test_identifier_with_conceptual_terms_routes_to_hybrid_rerank(
    router: DeterministicRetrievalRouter,
) -> None:
    """Test 'What were the breaking changes in Microsoft v3.0?' routes to Hybrid + Reranking."""
    decision = router.route("What were the breaking changes in Microsoft v3.0?")

    assert decision.strategy == RetrievalStrategy.HYBRID_RRF
    assert decision.use_reranker is True
    assert decision.candidate_pool_size == 20
    assert decision.final_top_k == 10
    assert decision.rule_name == "ComplexIdentifierOrVersionRule"


def test_identifier_with_temporal_term(router: DeterministicRetrievalRouter) -> None:
    """Test identifier with temporal constraint (e.g. 'Q3 roadmap for v2.5 release') routes to ComplexIdentifier."""
    decision = router.route("Q3 roadmap for v2.5 release")

    assert decision.strategy == RetrievalStrategy.HYBRID_RRF
    assert decision.use_reranker is True
    assert decision.candidate_pool_size == 20
    assert decision.rule_name == "ComplexIdentifierOrVersionRule"


def test_question_with_migration_and_version(router: DeterministicRetrievalRouter) -> None:
    """Test question with migration and version ('How to migrate to v2.0') routes to ComplexIdentifier."""
    decision = router.route("How to migrate to v2.0")

    assert decision.strategy == RetrievalStrategy.HYBRID_RRF
    assert decision.use_reranker is True
    assert decision.rule_name == "ComplexIdentifierOrVersionRule"


# ==============================================================================
# 3. Multi-Entity & Comparisons (Priority 1)
# ==============================================================================


def test_multi_known_entities_comparison(router: DeterministicRetrievalRouter) -> None:
    """Test comparison between two known entities triggers MultiEntityComparisonRule."""
    decision = router.route("Snowflake vs Databricks pricing and performance")

    assert decision.strategy == RetrievalStrategy.HYBRID_RRF
    assert decision.use_reranker is True
    assert decision.candidate_pool_size == 30
    assert decision.final_top_k == 10
    assert decision.rule_name == "MultiEntityComparisonRule"


def test_differences_between_versions(router: DeterministicRetrievalRouter) -> None:
    """Test comparison keywords with versions ('Differences between v1.0 and v2.0') triggers MultiEntityComparisonRule."""
    decision = router.route("Differences between v1.0 and v2.0")

    assert decision.strategy == RetrievalStrategy.HYBRID_RRF
    assert decision.use_reranker is True
    assert decision.candidate_pool_size == 30
    assert decision.final_top_k == 10
    assert decision.rule_name == "MultiEntityComparisonRule"


def test_heuristic_capitalized_words_do_not_falsely_trigger_multi_entity(
    router: DeterministicRetrievalRouter,
) -> None:
    """Test that capitalized words alone without comparison keywords do not trigger MultiEntityComparisonRule."""
    decision = router.route("Explain TurboEngine FastMode execution")

    assert decision.rule_name != "MultiEntityComparisonRule"


# ==============================================================================
# 4. Numeric & Pricing Metrics
# ==============================================================================


def test_numeric_pricing_with_known_entity(router: DeterministicRetrievalRouter) -> None:
    """Test 'Acme $99 pricing tier' routes to NumericMetricRule."""
    decision = router.route("Acme $99 pricing tier")

    assert decision.strategy == RetrievalStrategy.HYBRID_RRF
    assert decision.use_reranker is True
    assert decision.candidate_pool_size == 20
    assert decision.final_top_k == 5
    assert decision.rule_name == "NumericMetricRule"


def test_percentage_growth_metric(router: DeterministicRetrievalRouter) -> None:
    """Test percentage and financial metric query routes to NumericMetricRule."""
    decision = router.route("Databricks 15% revenue growth")

    assert decision.strategy == RetrievalStrategy.HYBRID_RRF
    assert decision.use_reranker is True
    assert decision.candidate_pool_size == 20
    assert decision.final_top_k == 5
    assert decision.rule_name == "NumericMetricRule"


# ==============================================================================
# 5. Pure Conceptual & Exploratory (Semantic Vector)
# ==============================================================================


def test_pure_conceptual_question(router: DeterministicRetrievalRouter) -> None:
    """Test broad conceptual question routes to DENSE_VECTOR without reranker overhead."""
    decision = router.route("How does the architecture handle multi-region scalability?")

    assert decision.strategy == RetrievalStrategy.DENSE_VECTOR
    assert not decision.use_reranker
    assert decision.candidate_pool_size == 20
    assert decision.final_top_k == 10
    assert decision.rule_name == "PureConceptualExploratoryRule"


def test_thematic_strategy_inquiry(router: DeterministicRetrievalRouter) -> None:
    """Test thematic market strategy question routes to DENSE_VECTOR."""
    decision = router.route("Explain company market expansion strategy and philosophy")

    assert decision.strategy == RetrievalStrategy.DENSE_VECTOR
    assert not decision.use_reranker
    assert decision.rule_name == "PureConceptualExploratoryRule"


# ==============================================================================
# 6. Ambiguity & Fallback Handling
# ==============================================================================


def test_ambiguous_query_fallback(router: DeterministicRetrievalRouter) -> None:
    """Test ambiguous general query safely falls back to Hybrid + Reranker."""
    decision = router.route("General platform capabilities")

    # 'capabilities' is a conceptual keyword, so it routes to PureConceptualExploratoryRule
    # Let's test a truly ambiguous query without conceptual, numeric, or version keywords:
    decision_ambiguous = router.route("notes and thoughts on product")

    assert decision_ambiguous.strategy == RetrievalStrategy.HYBRID_RRF
    assert decision_ambiguous.use_reranker is True
    assert decision_ambiguous.candidate_pool_size == 20
    assert decision_ambiguous.final_top_k == 10
    assert decision_ambiguous.rule_name == "DefaultFallbackRule"


# ==============================================================================
# 7. RetrievalQuery Support & Custom Configuration
# ==============================================================================


def test_retrieval_query_model_input(router: DeterministicRetrievalRouter) -> None:
    """Test passing RetrievalQuery instance to router."""
    req = RetrievalQuery(
        query_text="v2.4.1",
        metadata_filter=MetadataFilter(workspace_id="ws_intel"),
    )
    decision = router.route(req)

    assert decision.query_text == "v2.4.1"
    assert decision.strategy == RetrievalStrategy.SPARSE_BM25
    assert decision.rule_name == "ShortExactLookupRule"


def test_custom_router_configuration() -> None:
    """Test router respects custom budget and threshold configuration."""
    custom_cfg = RouterConfiguration(
        comparison_candidate_pool_size=50,
        short_lookup_candidate_pool_size=15,
        default_top_k=25,
        known_entities=["AlphaCorp", "BetaCorp"],
    )
    custom_router = DeterministicRetrievalRouter(config=custom_cfg)

    decision = custom_router.route("AlphaCorp vs BetaCorp")
    assert decision.candidate_pool_size == 50
    assert decision.final_top_k == 25
    assert decision.rule_name == "MultiEntityComparisonRule"
