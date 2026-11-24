"""Unit tests for Agentic RAG domain models, schemas, and invariants."""

import pytest
from pydantic import ValidationError

from competitive_intelligence.agentic_rag.models import (
    AgenticRAGState,
    ContradictionClaim,
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalResult,
    RetrievalStrategy,
)


def test_retrieval_decision_enum_has_no_direct_answer() -> None:
    """Verify RetrievalDecision strictly contains retrieval options and excludes direct answering."""
    assert RetrievalDecision.DIRECT_RETRIEVAL.value == "direct_retrieval"
    assert RetrievalDecision.DECOMPOSE_AND_RETRIEVE.value == "decompose_and_retrieve"
    assert not hasattr(RetrievalDecision, "DIRECT_ANSWER")


def test_sub_query_plan_defaults_and_validation() -> None:
    """Verify SubQueryPlan defaults and fields."""
    plan = SubQueryPlan(
        sub_query="Databricks pricing tiers 2026",
        target_company_ids=["databricks"],
        strategy_hint=RetrievalStrategy.SPARSE_BM25,
        rationale="Targeted lookup on pricing tables",
    )
    assert plan.sub_query == "Databricks pricing tiers 2026"
    assert plan.target_company_ids == ["databricks"]
    assert plan.strategy_hint == RetrievalStrategy.SPARSE_BM25
    assert plan.rationale == "Targeted lookup on pricing tables"


def test_query_analysis_model() -> None:
    """Verify QueryAnalysis structured schema."""
    analysis = QueryAnalysis(
        original_query="Compare Snowflake and Databricks pricing",
        decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
        reasoning="Multi-entity query requiring separate company searches",
        sub_queries=[
            SubQueryPlan(sub_query="Snowflake pricing plans", target_company_ids=["snowflake"]),
            SubQueryPlan(sub_query="Databricks pricing plans", target_company_ids=["databricks"]),
        ],
    )
    assert analysis.decision == RetrievalDecision.DECOMPOSE_AND_RETRIEVE
    assert len(analysis.sub_queries) == 2
    assert analysis.sub_queries[0].sub_query == "Snowflake pricing plans"


def test_contradiction_claim_model() -> None:
    """Verify ContradictionClaim structured model and chunk ID referencing."""
    claim = ContradictionClaim(
        claim_description="Conflicting reporting on Acme Corp Q3 revenue",
        conflicting_chunk_ids=["chunk_news_1", "chunk_sec_2"],
        rationale="News report claims $50M revenue while SEC filing indicates $42M.",
    )
    assert len(claim.conflicting_chunk_ids) == 2
    assert "chunk_news_1" in claim.conflicting_chunk_ids
    assert "chunk_sec_2" in claim.conflicting_chunk_ids


def test_verification_assessment_model() -> None:
    """Verify VerificationAssessment fields and sufficiency decisions."""
    assessment = VerificationAssessment(
        decision=SufficiencyDecision.CONTRADICTORY_NEED_VERIFICATION,
        is_sufficient=False,
        reasoning="Differing pricing data found across two retrieved sources.",
        detected_contradictions=[
            ContradictionClaim(
                claim_description="Price mismatch on Enterprise tier",
                conflicting_chunk_ids=["chunk_a", "chunk_b"],
                rationale="Chunk A says $99/mo, Chunk B says $79/mo",
            )
        ],
        missing_aspects=["Need official pricing page confirmation."],
        cited_chunk_ids=["chunk_a", "chunk_b"],
        recommended_next_query="Acme Corp official enterprise tier pricing 2026",
    )
    assert assessment.decision == SufficiencyDecision.CONTRADICTORY_NEED_VERIFICATION
    assert assessment.is_sufficient is False
    assert len(assessment.detected_contradictions) == 1
    assert assessment.recommended_next_query == "Acme Corp official enterprise tier pricing 2026"


def test_agentic_rag_state_has_no_synthesized_answer() -> None:
    """Verify AgenticRAGState strictly functions as a retrieval controller state without answer synthesis."""
    res = RetrievalResult(
        chunk_id="c1",
        score=0.95,
        text="Sample retrieved evidence text",
        document_id="doc1",
        version_id="v1",
        workspace_id="ws_1",
    )
    state = AgenticRAGState(
        workspace_id="ws_1",
        original_query="What are Snowflake's newest features?",
        metadata_filter=MetadataFilter(workspace_id="ws_1"),
        accumulated_evidence=[res],
        hops_executed=1,
        max_hops=3,
    )
    assert state.workspace_id == "ws_1"
    assert len(state.accumulated_evidence) == 1
    assert not hasattr(state, "synthesized_answer")
