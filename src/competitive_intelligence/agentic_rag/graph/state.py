"""State representation for the Agentic RAG LangGraph workflow."""

from typing import Any, TypedDict

from competitive_intelligence.agentic_rag.models import (
    QueryAnalysis,
    SubQueryPlan,
    VerificationAssessment,
)
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalResult,
)


class AgenticGraphState(TypedDict, total=False):
    """TypedDict state passed across LangGraph nodes during retrieval orchestration."""

    task_id: str
    workspace_id: str
    company_id: str | None
    original_query: str
    metadata_filter: MetadataFilter
    top_k: int
    query_analysis: QueryAnalysis | None
    sub_queries: list[SubQueryPlan]
    current_sub_queries: list[SubQueryPlan]
    accumulated_results: list[RetrievalResult]
    hop_count: int
    max_hops: int
    verification_assessments: list[VerificationAssessment]
    latest_verification: VerificationAssessment | None
    missing_aspects: list[str]
    corrective_query: str | None
    termination_reason: str | None
    is_fallback: bool
    fallback_reason: str | None
    metadata: dict[str, Any]
