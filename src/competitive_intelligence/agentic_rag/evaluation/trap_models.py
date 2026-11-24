"""Data models for the Hard-Case Retrieval-Trap Agentic RAG benchmark."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.evaluation.multihop_models import (
    DivergenceClassification,
)
from competitive_intelligence.agentic_rag.models import RetrievalDecision
from competitive_intelligence.knowledge_pipeline.models import DocumentChunk
from competitive_intelligence.retrieval_engine.models import MetadataFilter


class TrapBenchmarkCategory(str, Enum):
    """Evaluation query categories for retrieval trap and hard-case agentic RAG testing."""

    RETRIEVAL_TRAP = "retrieval_trap"
    BROAD_QUERY_DISTRACTOR = "broad_query_distractor"
    TERMINOLOGY_MISMATCH = "terminology_mismatch"
    MULTI_ENTITY_GAP = "multi_entity_gap"
    CROSS_DOCUMENT_GAP = "cross_document_gap"
    TEMPORAL_TRAP = "temporal_trap"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    MULTI_FACETED = "multi_faceted"
    DIRECT_EASY = "direct_easy"
    NEGATIVE_ADVERSARIAL = "negative_adversarial"


class TrapEvaluationQuery(BaseModel):
    """A hard-case benchmark query with explicit ground truth, required chunks, and distractor definitions."""

    query_id: str
    query_text: str
    category: TrapBenchmarkCategory
    relevant_chunk_ids: dict[str, int] = Field(
        default_factory=dict,
        description="Map of chunk_id to relevance grade (0=irrelevant, 1=marginal, 2=relevant, 3=essential).",
    )
    required_chunk_ids: list[str] = Field(
        default_factory=list,
        description="List of essential chunk_ids that must be retrieved for complete evidence coverage.",
    )
    distractor_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs with high lexical or surface overlap that act as retrieval distractors.",
    )
    metadata_filter: MetadataFilter | None = None
    expected_decision: RetrievalDecision | None = None
    is_negative_case: bool = False
    notes: str | None = None


class TrapEvaluationDataset(BaseModel):
    """A self-contained, frozen dataset for retrieval-trap benchmarking."""

    dataset_id: str
    corpus: list[DocumentChunk]
    queries: list[TrapEvaluationQuery]

    def get_queries_by_category(self, category: TrapBenchmarkCategory) -> list[TrapEvaluationQuery]:
        """Filter queries by category."""
        return [q for q in self.queries if q.category == category]

    def validate_ground_truth(self) -> list[str]:
        """Verify that all referenced chunk IDs exist in the corpus and required chunks are relevant."""
        valid_chunk_ids = {c.chunk_id for c in self.corpus}
        errors: list[str] = []
        for q in self.queries:
            for cid in q.relevant_chunk_ids:
                if cid not in valid_chunk_ids:
                    errors.append(f"Query '{q.query_id}' references unknown relevant chunk_id: '{cid}'")
            for cid in q.required_chunk_ids:
                if cid not in valid_chunk_ids:
                    errors.append(f"Query '{q.query_id}' references unknown required chunk_id: '{cid}'")
                elif cid not in q.relevant_chunk_ids or q.relevant_chunk_ids[cid] < 2:
                    errors.append(f"Query '{q.query_id}' required chunk_id '{cid}' is not marked relevant (>=2) in relevant_chunk_ids")
            for cid in q.distractor_chunk_ids:
                if cid not in valid_chunk_ids:
                    errors.append(f"Query '{q.query_id}' references unknown distractor chunk_id: '{cid}'")
        return errors


class TrapQueryExecutionRecord(BaseModel):
    """Comprehensive trace of how both retrievers executed on a retrieval-trap benchmark query."""

    query_id: str
    query_text: str
    category: TrapBenchmarkCategory
    relevant_chunk_ids: dict[str, int]
    required_chunk_ids: list[str] = Field(default_factory=list)
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

    # Required evidence miss and recovery tracking (evaluated at top_k=10)
    adaptive_missing_required_ids: list[str] = Field(default_factory=list)
    agentic_recovered_required_ids: list[str] = Field(default_factory=list)
    adaptive_missed_required: bool = False
    agentic_recovered_missed: bool = False
    agentic_unique_relevant_chunks: list[str] = Field(default_factory=list)

    # Agentic control metadata
    agentic_decision: str = "direct_retrieval"
    agentic_subqueries: list[str] = Field(default_factory=list)
    agentic_hops: int = 1
    agentic_verifications_count: int = 0
    agentic_corrective_query: str | None = None
    agentic_is_fallback: bool = False
    agentic_llm_calls: int = 0
    agentic_retrieval_calls: int = 1

    # Corrective hop utility
    corrective_produced_new_evidence: bool | None = None

    # Analysis
    classification: DivergenceClassification = DivergenceClassification.AGENTIC_OVERHEAD
    classification_rationale: str = ""


class TrapCategoryMetrics(BaseModel):
    """Aggregated retrieval, trap recovery, and operational metrics for a category."""

    category: TrapBenchmarkCategory
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

    # Trap & Recovery Metrics
    adaptive_miss_count: int = 0
    adaptive_miss_rate: float = 0.0
    agentic_recovery_count: int = 0
    agentic_recovery_rate: float = 0.0
    unique_evidence_recovered_count: int = 0

    # Control Metrics
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


class TrapBenchmarkReport(BaseModel):
    """Complete comparative evaluation report for the Hard-Case Retrieval-Trap Benchmark."""

    dataset_id: str
    total_corpus_size: int
    total_queries: int
    overall_adaptive_metrics: dict[str, float] = Field(default_factory=dict)
    overall_agentic_metrics: dict[str, float] = Field(default_factory=dict)
    trap_specific_metrics: dict[str, Any] = Field(default_factory=dict)
    category_metrics: dict[str, TrapCategoryMetrics] = Field(default_factory=dict)
    divergence_distribution: dict[str, int] = Field(default_factory=dict)
    query_records: list[TrapQueryExecutionRecord] = Field(default_factory=list)
