"""Data models for Multi-Hop and Adversarial Agentic RAG evaluation."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.models import RetrievalDecision
from competitive_intelligence.knowledge_pipeline.models import DocumentChunk
from competitive_intelligence.retrieval_engine.models import MetadataFilter


class BenchmarkCategory(str, Enum):
    """Evaluation query categories for multi-hop and agentic RAG testing."""

    MULTI_ENTITY = "multi_entity"
    MULTI_FACETED = "multi_faceted"
    TERMINOLOGY_MISMATCH = "terminology_mismatch"
    EVIDENCE_GAP = "evidence_gap"
    CROSS_DOCUMENT = "cross_document"
    TEMPORAL_CHANGE = "temporal_change"
    CONTRADICTORY = "contradictory"
    ADVERSARIAL_NEGATIVE = "adversarial_negative"
    DIRECT_LOOKUP = "direct_lookup"


class DivergenceClassification(str, Enum):
    """Deterministic classification of how AgenticRetriever compared to AdaptiveRetriever on a query."""

    AGENTIC_SUCCESS = "agentic_success"
    AGENTIC_OVERHEAD = "agentic_overhead"
    ADAPTIVE_BETTER = "adaptive_better"
    BOTH_FAILED = "both_failed"
    NEUTRAL_NEGATIVE = "neutral_negative"


class MultiHopEvaluationQuery(BaseModel):
    """A benchmark query with category metadata and explicit relevance ground truth."""

    query_id: str
    query_text: str
    category: BenchmarkCategory
    relevant_chunk_ids: dict[str, int] = Field(
        default_factory=dict,
        description="Map of chunk_id to relevance grade (0=irrelevant, 1=marginal, 2=relevant, 3=essential).",
    )
    metadata_filter: MetadataFilter | None = None
    expected_decision: RetrievalDecision | None = None
    expected_subqueries_min: int = 1
    expected_subqueries_max: int = 4
    is_negative_case: bool = False
    distractor_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs with high keyword overlap that should NOT be retrieved as relevant.",
    )
    notes: str | None = None


class MultiHopEvaluationDataset(BaseModel):
    """A self-contained, frozen dataset for multi-hop retrieval benchmarking."""

    dataset_id: str
    corpus: list[DocumentChunk]
    queries: list[MultiHopEvaluationQuery]

    def get_queries_by_category(self, category: BenchmarkCategory) -> list[MultiHopEvaluationQuery]:
        """Filter queries by category."""
        return [q for q in self.queries if q.category == category]

    def validate_ground_truth(self) -> list[str]:
        """Verify that all referenced ground truth chunk IDs exist in the corpus."""
        valid_chunk_ids = {c.chunk_id for c in self.corpus}
        errors: list[str] = []
        for q in self.queries:
            for cid in q.relevant_chunk_ids:
                if cid not in valid_chunk_ids:
                    errors.append(f"Query '{q.query_id}' references unknown chunk_id: '{cid}'")
            for cid in q.distractor_chunk_ids:
                if cid not in valid_chunk_ids:
                    errors.append(f"Query '{q.query_id}' references unknown distractor chunk_id: '{cid}'")
        return errors


class QueryExecutionRecord(BaseModel):
    """Comprehensive trace of how both retrievers executed on a single query."""

    query_id: str
    query_text: str
    category: BenchmarkCategory
    relevant_chunk_ids: dict[str, int]
    distractor_chunk_ids: list[str] = Field(default_factory=list)

    # AdaptiveRetriever results
    adaptive_chunk_ids: list[str] = Field(default_factory=list)
    adaptive_recall_at_5: float = 0.0
    adaptive_recall_at_10: float = 0.0
    adaptive_recall_at_20: float = 0.0
    adaptive_mrr: float = 0.0
    adaptive_ndcg_at_10: float = 0.0
    adaptive_latency_ms: float = 0.0

    # AgenticRetriever results
    agentic_chunk_ids: list[str] = Field(default_factory=list)
    agentic_recall_at_5: float = 0.0
    agentic_recall_at_10: float = 0.0
    agentic_recall_at_20: float = 0.0
    agentic_mrr: float = 0.0
    agentic_ndcg_at_10: float = 0.0
    agentic_latency_ms: float = 0.0

    # Agentic control metadata
    agentic_decision: str = "direct_retrieval"
    agentic_subqueries: list[str] = Field(default_factory=list)
    agentic_hops: int = 1
    agentic_verifications_count: int = 0
    agentic_corrective_query: str | None = None
    agentic_is_fallback: bool = False
    agentic_llm_calls: int = 0
    agentic_retrieval_calls: int = 1

    # Analysis
    classification: DivergenceClassification = DivergenceClassification.AGENTIC_OVERHEAD
    classification_rationale: str = ""


class CategoryMetrics(BaseModel):
    """Aggregated retrieval and operational metrics for a specific benchmark category."""

    category: BenchmarkCategory
    query_count: int = 0

    # Retrieval Quality - Adaptive
    adaptive_recall_at_5: float = 0.0
    adaptive_recall_at_10: float = 0.0
    adaptive_recall_at_20: float = 0.0
    adaptive_mrr: float = 0.0
    adaptive_ndcg_at_10: float = 0.0
    adaptive_mean_latency_ms: float = 0.0
    adaptive_median_latency_ms: float = 0.0
    adaptive_p95_latency_ms: float = 0.0

    # Retrieval Quality - Agentic
    agentic_recall_at_5: float = 0.0
    agentic_recall_at_10: float = 0.0
    agentic_recall_at_20: float = 0.0
    agentic_mrr: float = 0.0
    agentic_ndcg_at_10: float = 0.0
    agentic_mean_latency_ms: float = 0.0
    agentic_median_latency_ms: float = 0.0
    agentic_p95_latency_ms: float = 0.0

    # Agentic Control Metrics
    decomposition_rate: float = 0.0
    direct_retrieval_rate: float = 0.0
    unnecessary_decomposition_rate: float = 0.0
    average_subqueries: float = 0.0
    average_hops: float = 0.0
    corrective_retrieval_rate: float = 0.0
    llm_calls_per_query: float = 0.0
    retrieval_calls_per_query: float = 0.0
    fallback_rate: float = 0.0

    # Divergence Counts
    divergence_counts: dict[str, int] = Field(default_factory=dict)


class MultiHopComparativeReport(BaseModel):
    """Complete benchmark comparative report between AdaptiveRetriever and AgenticRetriever."""

    dataset_id: str
    total_corpus_size: int
    total_queries: int
    overall_adaptive_metrics: dict[str, float] = Field(default_factory=dict)
    overall_agentic_metrics: dict[str, float] = Field(default_factory=dict)
    category_metrics: dict[str, CategoryMetrics] = Field(default_factory=dict)
    divergence_distribution: dict[str, int] = Field(default_factory=dict)
    query_records: list[QueryExecutionRecord] = Field(default_factory=list)
