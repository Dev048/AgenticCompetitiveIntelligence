"""Domain models for Retrieval Engine subsystem.

NOTE: These models are provisional placeholders for initial architectural scaffolding,
not finalized domain contracts. They will be refined during Subsystem 3 detailed design.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from competitive_intelligence.knowledge_pipeline.models import DocumentChunk


class RetrievalStrategy(str, Enum):
    """Retrieval strategies selectable by the query router."""

    DENSE_VECTOR = "dense_vector"
    SPARSE_BM25 = "sparse_bm25"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_WEIGHTED = "hybrid_weighted"


class MetadataFilter(BaseModel):
    """Structured filters applied during vector and lexical retrieval."""

    workspace_id: str
    company_ids: list[str] | None = None
    source_categories: list[str] | None = None
    source_tiers: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    custom_filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    """Query payload sent to the retrieval engine."""

    query_text: str
    metadata_filter: MetadataFilter
    top_k: int = Field(default=20, ge=1, le=100)
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RRF
    min_score_threshold: float = 0.0


class ScoredDocumentChunk(BaseModel):
    """DocumentChunk enriched with score and retrieval ranking provenance."""

    chunk: DocumentChunk
    score: float
    rank: int = 0
    retrieval_mode: str = "dense"


class RetrievalResult(BaseModel):
    """Retrieval-method agnostic scored result item returned by dense, lexical, or hybrid retrievers."""

    chunk_id: str
    score: float
    text: str
    document_id: str
    version_id: str
    source_id: str | None = None
    company_id: str | None = None
    workspace_id: str | None = None
    source_category: str | None = None
    content_type: str | None = None
    canonical_url: str | None = None
    chunk_index: int = 0
    parent_chunk_id: str | None = None
    chunk_level: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Reranking metadata
    reranker_score: float | None = None
    reranker_model: str | None = None
    original_rank: int | None = None

    @classmethod
    def from_chunk(cls, chunk: DocumentChunk, score: float) -> "RetrievalResult":
        """Factory constructor creating a RetrievalResult from a DocumentChunk and score."""
        return cls(
            chunk_id=chunk.chunk_id,
            score=score,
            text=chunk.content,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            source_id=chunk.metadata.source_id,
            company_id=chunk.metadata.company_id,
            workspace_id=chunk.metadata.workspace_id,
            source_category=chunk.metadata.source_category,
            content_type=chunk.metadata.custom_tags.get("content_type"),
            canonical_url=chunk.metadata.source_url,
            chunk_index=chunk.chunk_index,
            parent_chunk_id=chunk.parent_chunk_id,
            chunk_level=chunk.chunk_level,
            metadata=dict(chunk.metadata.custom_tags),
        )

    @classmethod
    def from_vector_search_result(cls, res: Any) -> "RetrievalResult":
        """Factory constructor creating a RetrievalResult from a VectorSearchResult."""
        return cls(
            chunk_id=res.chunk_id,
            score=res.score,
            text=res.text,
            document_id=res.document_id,
            version_id=res.version_id,
            source_id=res.source_id,
            company_id=res.company_id,
            workspace_id=res.workspace_id,
            source_category=res.source_category,
            content_type=res.content_type,
            canonical_url=res.canonical_url,
            chunk_index=res.chunk_index,
            parent_chunk_id=res.parent_chunk_id,
            chunk_level=res.chunk_level,
            metadata=dict(res.metadata),
        )


class RetrievalEvaluationMetrics(BaseModel):
    """Evaluation benchmark metrics for retrieval quality."""

    recall_at_k: dict[int, float] = Field(default_factory=dict)
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    sample_size: int = 0
    average_latency_ms: float = 0.0


class HybridSearchConfiguration(BaseModel):
    """Configuration parameters for Reciprocal Rank Fusion hybrid retrieval."""

    fusion_candidate_k: int = 50
    final_top_k: int | None = 10
    candidate_k: int | None = None
    semantic_candidate_k: int | None = None
    lexical_candidate_k: int | None = None
    rrf_k: int = 60

    @field_validator("fusion_candidate_k", "rrf_k")
    @classmethod
    def validate_positive_integers(cls, value: int) -> int:
        """Ensure fusion_candidate_k and rrf_k are strictly positive."""
        if value <= 0:
            raise ValueError("Value must be a positive integer")
        return value
        
    @field_validator("final_top_k")
    @classmethod
    def validate_optional_positive_integers_for_final(cls, value: int | None) -> int | None:
        """Ensure final_top_k is positive if specified."""
        if value is not None and value <= 0:
            raise ValueError("Value must be a positive integer if specified")
        return value

    @field_validator("candidate_k", "semantic_candidate_k", "lexical_candidate_k")
    @classmethod
    def validate_optional_positive_integers(cls, value: int | None) -> int | None:
        """Ensure candidate pool sizes are positive if specified."""
        if value is not None and value <= 0:
            raise ValueError("Candidate pool size must be a positive integer if specified")
        return value

    @property
    def effective_semantic_k(self) -> int:
        """Effective candidate pool size requested from semantic retriever."""
        if self.semantic_candidate_k is not None:
            return self.semantic_candidate_k
        if self.candidate_k is not None:
            return self.candidate_k
        return 5 * self.fusion_candidate_k

    @property
    def effective_lexical_k(self) -> int:
        """Effective candidate pool size requested from BM25 lexical retriever."""
        if self.lexical_candidate_k is not None:
            return self.lexical_candidate_k
        if self.candidate_k is not None:
            return self.candidate_k
        return 5 * self.fusion_candidate_k
