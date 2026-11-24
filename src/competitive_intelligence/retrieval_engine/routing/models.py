"""Domain models and configuration for Retrieval Engine query routing subsystem.

Captures extracted query features, routing decisions, and router configuration parameters.
"""

from pydantic import BaseModel, Field, field_validator

from competitive_intelligence.retrieval_engine.models import RetrievalStrategy


class QueryFeatures(BaseModel):
    """Linguistic, structural, and semantic features extracted deterministically from a query."""

    raw_query: str
    token_count: int
    char_count: int

    # Identifier & Version signals
    version_patterns: list[str] = Field(default_factory=list)
    identifier_patterns: list[str] = Field(default_factory=list)
    has_quotes: bool = False
    quoted_phrases: list[str] = Field(default_factory=list)

    # Numeric & Financial signals
    numeric_patterns: list[str] = Field(default_factory=list)

    # Entity signals
    known_entities: list[str] = Field(default_factory=list)
    heuristic_entities: list[str] = Field(default_factory=list)

    # Semantic & Analytical signals
    comparison_terms: list[str] = Field(default_factory=list)
    temporal_terms: list[str] = Field(default_factory=list)
    conceptual_terms: list[str] = Field(default_factory=list)
    is_interrogative: bool = False

    @property
    def has_version(self) -> bool:
        """Whether any explicit version pattern was detected."""
        return len(self.version_patterns) > 0

    @property
    def has_identifier(self) -> bool:
        """Whether any exact identifier, error code, or SKU pattern was detected."""
        return len(self.identifier_patterns) > 0

    @property
    def has_numeric(self) -> bool:
        """Whether currency, percentage, or specific numeric metric was detected."""
        return len(self.numeric_patterns) > 0

    @property
    def has_comparison(self) -> bool:
        """Whether comparative keywords or operators were detected."""
        return len(self.comparison_terms) > 0

    @property
    def has_temporal(self) -> bool:
        """Whether temporal terms (quarter, year, recent, roadmap) were detected."""
        return len(self.temporal_terms) > 0

    @property
    def has_conceptual_terms(self) -> bool:
        """Whether conceptual, architectural, or analytical terms were detected."""
        return len(self.conceptual_terms) > 0


class RoutingDecision(BaseModel):
    """Deterministic routing decision specifying strategy, budget, and rationale for a query."""

    query_text: str
    strategy: RetrievalStrategy
    use_reranker: bool = False
    candidate_pool_size: int = 20
    final_top_k: int = 10
    rule_name: str
    rationale: str
    features: QueryFeatures

    @field_validator("candidate_pool_size", "final_top_k")
    @classmethod
    def validate_positive_integers(cls, value: int) -> int:
        """Ensure budget parameters are strictly positive integers."""
        if value <= 0:
            raise ValueError("Candidate pool size and final_top_k must be strictly positive integers")
        return value


class RouterConfiguration(BaseModel):
    """Configuration parameters and threshold settings for DeterministicRetrievalRouter."""

    # Candidate pool sizes for different routing tiers
    short_lookup_candidate_pool_size: int = 10
    numeric_candidate_pool_size: int = 20
    comparison_candidate_pool_size: int = 30
    conceptual_candidate_pool_size: int = 20
    fallback_candidate_pool_size: int = 20

    # Top-K defaults
    default_top_k: int = 10
    exact_lookup_top_k: int = 5
    numeric_top_k: int = 5

    # Configured known entities (authoritative entity signal)
    known_entities: list[str] = Field(default_factory=list)

    # Heuristic thresholds
    short_lookup_max_tokens: int = 5

    @field_validator(
        "short_lookup_candidate_pool_size",
        "numeric_candidate_pool_size",
        "comparison_candidate_pool_size",
        "conceptual_candidate_pool_size",
        "fallback_candidate_pool_size",
        "default_top_k",
        "exact_lookup_top_k",
        "numeric_top_k",
        "short_lookup_max_tokens",
    )
    @classmethod
    def validate_positive_integers(cls, value: int) -> int:
        """Ensure all configuration thresholds and budgets are strictly positive."""
        if value <= 0:
            raise ValueError("Router configuration values must be strictly positive integers")
        return value
