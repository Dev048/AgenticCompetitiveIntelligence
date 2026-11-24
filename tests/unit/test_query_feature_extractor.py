"""Unit tests for QueryFeatureExtractor in retrieval routing."""

import pytest

from competitive_intelligence.retrieval_engine.routing.feature_extractor import (
    QueryFeatureExtractor,
)


@pytest.fixture
def extractor() -> QueryFeatureExtractor:
    return QueryFeatureExtractor(known_entities=["Snowflake", "Databricks", "Microsoft", "OpenAI", "Acme"])


def test_empty_and_whitespace_query(extractor: QueryFeatureExtractor) -> None:
    """Test feature extraction on empty or whitespace strings."""
    res_empty = extractor.extract("")
    assert res_empty.token_count == 0
    assert res_empty.char_count == 0
    assert not res_empty.has_version
    assert not res_empty.has_identifier
    assert not res_empty.has_numeric
    assert not res_empty.has_comparison
    assert not res_empty.has_conceptual_terms

    res_ws = extractor.extract("   \t\n  ")
    assert res_ws.token_count == 0
    assert res_ws.char_count == 7


def test_version_and_identifier_extraction(extractor: QueryFeatureExtractor) -> None:
    """Test extraction of semantic versions, error codes, SKUs, and model names."""
    q = "What is the status of v2.4.1 and SKU-9821 alongside GPT-4o and ERR_AUTH_TIMEOUT?"
    res = extractor.extract(q)

    assert "v2.4.1" in res.version_patterns
    assert res.has_version
    assert "SKU-9821" in res.identifier_patterns
    assert "GPT-4o" in res.identifier_patterns
    assert "ERR_AUTH_TIMEOUT" in res.identifier_patterns
    assert res.has_identifier
    assert res.is_interrogative


def test_quoted_phrase_extraction(extractor: QueryFeatureExtractor) -> None:
    """Test extraction of exact quoted phrases."""
    q = 'Search for "zero downtime migration" in documentation'
    res = extractor.extract(q)

    assert res.has_quotes
    assert res.quoted_phrases == ["zero downtime migration"]


def test_numeric_currency_and_metric_extraction(extractor: QueryFeatureExtractor) -> None:
    """Test extraction of dollars, euros, percentages, and financial metrics."""
    q = "Acme reduced pricing to $99 per seat, growing ARR by 15% to $10M"
    res = extractor.extract(q)

    assert res.has_numeric
    assert "$99" in res.numeric_patterns
    assert "15%" in res.numeric_patterns
    assert "$10M" in res.numeric_patterns


def test_known_entities_vs_heuristic_entities(extractor: QueryFeatureExtractor) -> None:
    """Test that known entities are accurately extracted and heuristic capitalized words are isolated."""
    q = "Compare Snowflake performance against Databricks using TurboEngine benchmark"
    res = extractor.extract(q)

    # Known entities (authoritative)
    assert "Snowflake" in res.known_entities
    assert "Databricks" in res.known_entities
    assert len(res.known_entities) == 2

    # Heuristic entities (capitalized tokens not in known entities or stopwords)
    assert "TurboEngine" in res.heuristic_entities
    assert "Compare" not in res.heuristic_entities  # first token skipped


def test_comparison_and_temporal_terms(extractor: QueryFeatureExtractor) -> None:
    """Test extraction of comparative keywords and temporal milestones."""
    q = "Differences between Databricks vs Snowflake in Q3 2024 annual roadmap"
    res = extractor.extract(q)

    assert res.has_comparison
    assert "vs" in res.comparison_terms or "differences between" in res.comparison_terms
    assert res.has_temporal
    assert any("q3" in t.lower() or "2024" in t.lower() or "roadmap" in t.lower() for t in res.temporal_terms)


def test_conceptual_analytical_terms(extractor: QueryFeatureExtractor) -> None:
    """Test extraction of architectural and conceptual keywords."""
    q = "Explain the multi-region scalability and security architecture"
    res = extractor.extract(q)

    assert res.has_conceptual_terms
    assert "scalability" in res.conceptual_terms or "architecture" in res.conceptual_terms or "security" in res.conceptual_terms


def test_interrogative_detection(extractor: QueryFeatureExtractor) -> None:
    """Test detection of question phrasing."""
    assert extractor.extract("How does encryption work?").is_interrogative
    assert extractor.extract("Why did the price change?").is_interrogative
    assert extractor.extract("What are the breaking changes?").is_interrogative
    assert not extractor.extract("Encryption architecture overview").is_interrogative
