"""Data models and configuration schemas for Selective Agentic RAG Escalation."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from competitive_intelligence.retrieval_engine.models import RetrievalResult


class EscalationSignalType(str, Enum):
    """Deterministic signals indicating potential evidence deficiency or query complexity."""

    MULTI_ENTITY_UNBALANCED = "multi_entity_unbalanced"
    MULTI_FACET_DISJOINT = "multi_facet_disjoint"
    FACET_EVIDENCE_GAP = "facet_evidence_gap"
    ENTITY_FACET_UNBALANCED = "entity_facet_unbalanced"
    TEMPORAL_COMPARISON = "temporal_comparison"
    EVIDENCE_GAP_DETECTED = "evidence_gap_detected"
    LOW_EVIDENCE_COUNT = "low_evidence_count"
    LOW_TOP_SCORE = "low_top_score"
    HIGH_RANK_CONCENTRATION = "high_rank_concentration"
    POTENTIAL_CONTRADICTION = "potential_contradiction"



class EscalationAssessment(BaseModel):
    """Detailed deterministic assessment of whether a query warrants Agentic RAG escalation."""

    should_escalate: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    signals_detected: list[EscalationSignalType] = Field(default_factory=list)
    initial_evidence_count: int = 0
    top_score: float = 0.0
    entity_coverage_ratio: float = 1.0
    facet_coverage_ratio: float = 1.0
    assessment_metadata: dict[str, Any] = Field(default_factory=dict)


class EscalationPolicyConfig(BaseModel):
    """Explicit threshold configuration for the deterministic escalation policy.
    
    All thresholds are explicit and documented to eliminate scattered magic numbers.
    """

    min_score_threshold: float = Field(
        default=0.10,
        description="Heuristic threshold: minimum top-1 retrieval score below which evidence is flagged as weak.",
    )
    min_evidence_count: int = Field(
        default=1,
        description="Heuristic threshold: minimum number of initial candidates required to avoid low evidence count signal.",
    )
    min_entity_coverage: float = Field(
        default=0.99,
        description="Heuristic threshold: ratio of queried entities that must appear in initial top-k evidence.",
    )
    min_facet_coverage: float = Field(
        default=0.99,
        description="Heuristic threshold: ratio of multi-faceted query aspects present in initial top-k evidence.",
    )
    known_entities: list[str] = Field(
        default_factory=lambda: [
            "snowflake", "databricks", "openai", "anthropic", "google", "microsoft",
            "qdrant", "pinecone", "meta", "mistral", "apple", "xai", "stripe", "adyen",
            "cloudflare", "fastly", "akamai", "weaviate", "milvus", "chroma", "cohere",
        ],
        description="Domain list of known competitors and platforms used for multi-entity entity detection.",
    )
    comparison_indicators: list[str] = Field(
        default_factory=lambda: [
            "compare ", " vs ", " versus ", " between ", "difference between", "comparison between",
        ],
        description="Keywords indicating comparative queries that require balanced multi-entity retrieval.",
    )
    temporal_indicators: list[str] = Field(
        default_factory=lambda: [
            "change from", "change between", "evolve from", "historical", "before and after",
            "2023 and 2024", "2023 to 2024", "previous vs current",
        ],
        description="Phrases indicating temporal evolution queries.",
    )
    facet_splitters: list[str] = Field(
        default_factory=lambda: [
            " across ", " spanning ", " developments in ", " capabilities regarding ", " offerings across ",
        ],
        description="Phrases used to split compound multi-faceted queries into distinct semantic facets.",
    )
    gap_critical_terms: list[str] = Field(
        default_factory=lambda: [
            "credit", "remedy", "refund", "tariff", "egress", "penalty", "surge",
            "overage", "escrow", "earnout", "byok", "hsm", "rto", "rpo",
            "sovereign", "govcloud", "air-gapped", "tpm", "rpm",
        ],
        description="Critical contract, technical, and compliance terms often omitted by broad initial queries.",
    )
    assessment_top_k: int = Field(
        default=5,
        description="Heuristic depth: top candidate window examined for entity coverage, facet representation, and rank concentration.",
    )
    max_rank_concentration: float = Field(
        default=0.80,
        description="Heuristic threshold: maximum allowable fraction of top candidates dominated by a single entity in multi-entity queries.",
    )
    min_facet_word_overlap: float = Field(
        default=0.60,
        description="Heuristic threshold: minimum fraction of words in a facet phrase that must match candidate text for coverage.",
    )



class ResultSelectionStrategy(str, Enum):
    """Strategy for selecting or fusing evidence when escalation occurs."""

    PRESERVE_BETTER = "preserve_better"
    HYBRID_MERGE = "hybrid_merge"
    AGENTIC_PREFERRED = "agentic_preferred"
    ADAPTIVE_ONLY = "adaptive_only"


class SelectiveRetrieverConfig(BaseModel):
    """Configuration for SelectiveAgenticRetriever controller."""

    policy_config: EscalationPolicyConfig = Field(default_factory=EscalationPolicyConfig)
    selection_strategy: ResultSelectionStrategy = ResultSelectionStrategy.PRESERVE_BETTER
    final_top_k: int = 20
    enable_selective_escalation: bool = True


class SelectiveExecutionTrace(BaseModel):
    """Audit trace for a selective retrieval execution."""

    query_text: str
    escalated: bool
    assessment: EscalationAssessment
    initial_retrieval_count: int
    agentic_retrieval_count: int = 0
    final_result_count: int = 0
    final_source: str = "adaptive"
    latency_ms: float = 0.0
    llm_calls: int = 0
    retrieval_hops: int = 1
