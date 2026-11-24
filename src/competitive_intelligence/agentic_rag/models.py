"""Domain models and state definitions for the Agentic RAG subsystem.

Subsystem 4: Bounded Agentic RAG Retrieval Controller.
Defines structured schemas for query analysis, subquery planning, evidence verification,
contradiction claims, and graph state transitions without final answer generation.
"""

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    ScoredDocumentChunk,
)


class RetrievalDecision(str, Enum):
    """Decision on whether and how to retrieve external evidence."""

    DIRECT_RETRIEVAL = "direct_retrieval"
    DECOMPOSE_AND_RETRIEVE = "decompose_and_retrieve"


class SufficiencyDecision(str, Enum):
    """Evaluation result regarding retrieved evidence quality and completeness."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT_NEED_HOP = "insufficient_need_hop"
    CONTRADICTORY_NEED_VERIFICATION = "contradictory_need_verification"


class SubQueryPlan(BaseModel):
    """A decomposed sub-query focused on a specific competitor, topic, or source category."""

    sub_query: str
    target_company_ids: list[str] = Field(default_factory=list)
    target_source_categories: list[str] = Field(default_factory=list)
    strategy_hint: RetrievalStrategy | None = None
    rationale: str = Field(default="", description="Explanation of what aspect this sub-query retrieves.")


class QueryAnalysis(BaseModel):
    """Result of the query understanding and decomposition step."""

    original_query: str
    decision: RetrievalDecision
    reasoning: str
    sub_queries: list[SubQueryPlan] = Field(default_factory=list)


class ContradictionClaim(BaseModel):
    """Structured representation of conflicting observations across retrieved evidence chunks.
    
    Identifies conflicting observations without declaring one source factually false.
    All claims must reference actual retrieved chunk IDs.
    """

    claim_description: str
    conflicting_chunk_ids: list[str] = Field(
        default_factory=list,
        description="List of chunk IDs that contain conflicting statements or data.",
    )
    rationale: str = Field(
        default="",
        description="Explanation of why these chunks present conflicting observations.",
    )


class VerificationAssessment(BaseModel):
    """Evaluation of retrieved candidate evidence against the information need."""

    decision: SufficiencyDecision
    is_sufficient: bool
    reasoning: str
    detected_contradictions: list[ContradictionClaim] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    recommended_next_query: str | None = None


class AgenticRAGState(BaseModel):
    """State graph representation passed across LangGraph nodes and execution stages."""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str
    company_id: str | None = None
    original_query: str
    metadata_filter: MetadataFilter | None = None
    query_analysis: QueryAnalysis | None = None
    accumulated_evidence: list[RetrievalResult] = Field(default_factory=list)
    hops_executed: int = 0
    max_hops: int = 3
    verifications: list[VerificationAssessment] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    corrective_query: str | None = None
    termination_reason: str | None = None
    is_fallback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
