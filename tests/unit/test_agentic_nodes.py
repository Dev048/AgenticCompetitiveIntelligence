"""Unit tests for individual LangGraph processing nodes."""

import pytest

from competitive_intelligence.agentic_rag.graph.state import AgenticGraphState
from competitive_intelligence.agentic_rag.llm.errors import LLMUnavailableError
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.models import (
    ContradictionClaim,
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.agentic_rag.nodes import (
    CorrectiveRetrievalNode,
    ExecuteSubqueriesNode,
    QueryAnalysisNode,
    VerifyEvidenceNode,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)


class SpyRetriever(BaseRetriever):
    """Spy retriever tracking received queries and returning configurable results."""

    def __init__(self, canned_results: dict[str, list[RetrievalResult]] | None = None) -> None:
        self.canned_results = canned_results or {}
        self.recorded_queries: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.recorded_queries.append(query)
        return self.canned_results.get(query.query_text, [])


def make_result(chunk_id: str, score: float = 0.9, text: str = "Evidence text") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        document_id="doc_1",
        version_id="ver_1",
        workspace_id="ws_test",
        company_id="comp_test",
    )


# ==============================================================================
# 1. Query Analysis Node Tests
# ==============================================================================


@pytest.mark.anyio
async def test_query_analysis_node_direct_retrieval() -> None:
    """Test QueryAnalysisNode produces direct retrieval plan."""
    llm = MockLLMClient()
    node = QueryAnalysisNode(llm_client=llm)

    state: AgenticGraphState = {
        "original_query": "What is Snowflake's revenue in 2025?",
        "workspace_id": "ws_test",
    }
    update = await node(state)

    assert update["hop_count"] == 1
    analysis: QueryAnalysis = update["query_analysis"]
    assert analysis.decision == RetrievalDecision.DIRECT_RETRIEVAL
    assert len(update["sub_queries"]) == 1


@pytest.mark.anyio
async def test_query_analysis_node_decomposition() -> None:
    """Test QueryAnalysisNode produces multi-part subqueries."""
    analysis_canned = QueryAnalysis(
        original_query="Compare Snowflake and Databricks",
        decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
        reasoning="Multi-entity query",
        sub_queries=[
            SubQueryPlan(sub_query="Snowflake features", target_company_ids=["snowflake"]),
            SubQueryPlan(sub_query="Databricks features", target_company_ids=["databricks"]),
        ],
    )
    llm = MockLLMClient(canned_responses={QueryAnalysis: analysis_canned})
    node = QueryAnalysisNode(llm_client=llm)

    state: AgenticGraphState = {
        "original_query": "Compare Snowflake and Databricks",
        "workspace_id": "ws_test",
    }
    update = await node(state)

    assert len(update["sub_queries"]) == 2
    assert update["sub_queries"][0].sub_query == "Snowflake features"
    assert update["sub_queries"][1].sub_query == "Databricks features"


@pytest.mark.anyio
async def test_query_analysis_node_propagates_llm_failure_for_atomic_fallback() -> None:
    """Test QueryAnalysisNode propagates LLM failures so that atomic fallback is triggered by outer retriever."""
    llm = MockLLMClient(simulate_failure=LLMUnavailableError("API Error"))
    node = QueryAnalysisNode(llm_client=llm)

    state: AgenticGraphState = {
        "original_query": "Failing query",
        "workspace_id": "ws_test",
        "company_id": "comp_1",
    }
    with pytest.raises(LLMUnavailableError, match="API Error"):
        await node(state)


# ==============================================================================
# 2. Execute Subqueries Node Tests
# ==============================================================================


@pytest.mark.anyio
async def test_execute_subqueries_node_concurrent_and_deduplicated() -> None:
    """Test ExecuteSubqueriesNode runs queries concurrently, preserves filters, and deduplicates."""
    r1 = make_result("chunk_1", score=0.8)
    r2 = make_result("chunk_2", score=0.85)
    r1_higher = make_result("chunk_1", score=0.95)  # Same chunk, higher score

    spy_retriever = SpyRetriever(canned_results={
        "query alpha": [r1, r2],
        "query beta": [r1_higher],
    })
    node = ExecuteSubqueriesNode(retriever=spy_retriever)

    state: AgenticGraphState = {
        "workspace_id": "ws_tenant_1",
        "metadata_filter": MetadataFilter(workspace_id="ws_tenant_1"),
        "current_sub_queries": [
            SubQueryPlan(sub_query="query alpha", rationale="Reason alpha"),
            SubQueryPlan(sub_query="query beta", rationale="Reason beta"),
        ],
        "accumulated_results": [],
        "hop_count": 1,
        "top_k": 10,
    }

    update = await node(state)
    results: list[RetrievalResult] = update["accumulated_results"]

    # Verify both queries were executed
    assert len(spy_retriever.recorded_queries) == 2
    # Verify workspace isolation preserved
    assert all(q.metadata_filter.workspace_id == "ws_tenant_1" for q in spy_retriever.recorded_queries)

    # Verify deduplication: chunk_1 and chunk_2 (total 2 unique chunks)
    assert len(results) == 2
    chunk_map = {r.chunk_id: r for r in results}
    assert "chunk_1" in chunk_map
    assert "chunk_2" in chunk_map
    # Verify higher score is retained for chunk_1
    assert chunk_map["chunk_1"].score == 0.95
    # Verify provenance attached
    assert chunk_map["chunk_1"].metadata["agentic_hop"] == 1


# ==============================================================================
# 3. Verify Evidence Node Tests
# ==============================================================================


@pytest.mark.anyio
async def test_verify_evidence_node_empty_evidence() -> None:
    """Test VerifyEvidenceNode handles 0 chunks without invoking LLM."""
    llm = MockLLMClient()
    node = VerifyEvidenceNode(llm_client=llm)

    state: AgenticGraphState = {
        "original_query": "What are the specs?",
        "accumulated_results": [],
        "hop_count": 1,
    }
    update = await node(state)

    assessment: VerificationAssessment = update["latest_verification"]
    assert assessment.is_sufficient is False
    assert assessment.decision == SufficiencyDecision.INSUFFICIENT_NEED_HOP
    assert len(llm.call_history) == 0  # Did not waste LLM call


@pytest.mark.anyio
async def test_verify_evidence_node_short_circuits_on_termination_reason() -> None:
    """Test VerifyEvidenceNode immediately short-circuits without LLM call when termination_reason is set."""
    llm = MockLLMClient()
    node = VerifyEvidenceNode(llm_client=llm)

    real_chunk = make_result("chunk_1", score=0.9)
    state: AgenticGraphState = {
        "original_query": "What are the specs?",
        "accumulated_results": [real_chunk],
        "hop_count": 1,
        "termination_reason": "Budget limit reached on hop: retrieval hops exceeded",
    }
    update = await node(state)

    # Must return empty dict and NOT invoke LLM
    assert update == {}
    assert len(llm.call_history) == 0


@pytest.mark.anyio
async def test_verify_evidence_node_short_circuits_on_is_fallback() -> None:
    """Test VerifyEvidenceNode immediately short-circuits without LLM call when is_fallback is True."""
    llm = MockLLMClient()
    node = VerifyEvidenceNode(llm_client=llm)

    real_chunk = make_result("chunk_1", score=0.9)
    state: AgenticGraphState = {
        "original_query": "What are the specs?",
        "accumulated_results": [real_chunk],
        "hop_count": 1,
        "is_fallback": True,
    }
    update = await node(state)

    assert update == {}
    assert len(llm.call_history) == 0


@pytest.mark.anyio
async def test_verify_evidence_node_contradiction_discarded_for_single_valid_and_hallucinated() -> None:
    """Test that a contradiction with ['valid_chunk_1', 'hallucinated_chunk'] is discarded."""
    canned_assessment = VerificationAssessment(
        decision=SufficiencyDecision.CONTRADICTORY_NEED_VERIFICATION,
        is_sufficient=False,
        reasoning="Contradiction observed across chunks",
        detected_contradictions=[
            ContradictionClaim(
                claim_description="Pricing mismatch",
                conflicting_chunk_ids=["valid_chunk_1", "hallucinated_chunk"],
                rationale="One valid and one hallucinated chunk",
            ),
        ],
        cited_chunk_ids=["valid_chunk_1", "hallucinated_chunk"],
    )

    llm = MockLLMClient(canned_responses={VerificationAssessment: canned_assessment})
    node = VerifyEvidenceNode(llm_client=llm)

    valid_chunk_1 = make_result("valid_chunk_1", score=0.9)
    state: AgenticGraphState = {
        "original_query": "Check pricing",
        "accumulated_results": [valid_chunk_1],
        "hop_count": 1,
    }
    update = await node(state)

    assessment: VerificationAssessment = update["latest_verification"]
    # Contradiction cannot be grounded in a single chunk -> discarded
    assert len(assessment.detected_contradictions) == 0
    # Cited valid ID is retained while hallucinated ID is removed
    assert assessment.cited_chunk_ids == ["valid_chunk_1"]


@pytest.mark.anyio
async def test_verify_evidence_node_contradiction_retained_for_two_valid_chunks() -> None:
    """Test that a contradiction with ['valid_chunk_1', 'valid_chunk_2'] is retained."""
    canned_assessment = VerificationAssessment(
        decision=SufficiencyDecision.CONTRADICTORY_NEED_VERIFICATION,
        is_sufficient=False,
        reasoning="Contradiction observed across chunks",
        detected_contradictions=[
            ContradictionClaim(
                claim_description="Feature support mismatch",
                conflicting_chunk_ids=["valid_chunk_1", "valid_chunk_2"],
                rationale="Source A claims supported, Source B claims unsupported",
            ),
        ],
        cited_chunk_ids=["valid_chunk_1", "valid_chunk_2"],
    )

    llm = MockLLMClient(canned_responses={VerificationAssessment: canned_assessment})
    node = VerifyEvidenceNode(llm_client=llm)

    valid_chunk_1 = make_result("valid_chunk_1", score=0.9)
    valid_chunk_2 = make_result("valid_chunk_2", score=0.85)
    state: AgenticGraphState = {
        "original_query": "Check feature support",
        "accumulated_results": [valid_chunk_1, valid_chunk_2],
        "hop_count": 1,
    }
    update = await node(state)

    assessment: VerificationAssessment = update["latest_verification"]
    # Retained because at least two distinct valid chunk IDs exist
    assert len(assessment.detected_contradictions) == 1
    assert assessment.detected_contradictions[0].conflicting_chunk_ids == ["valid_chunk_1", "valid_chunk_2"]
    assert assessment.cited_chunk_ids == ["valid_chunk_1", "valid_chunk_2"]


@pytest.mark.anyio
async def test_verify_evidence_node_contradiction_discarded_for_duplicate_same_chunk() -> None:
    """Test that a contradiction containing duplicate references to the same single chunk is discarded."""
    canned_assessment = VerificationAssessment(
        decision=SufficiencyDecision.CONTRADICTORY_NEED_VERIFICATION,
        is_sufficient=False,
        reasoning="Contradiction observed within single chunk duplicate",
        detected_contradictions=[
            ContradictionClaim(
                claim_description="Single chunk self-contradiction",
                conflicting_chunk_ids=["valid_chunk_1", "valid_chunk_1"],
                rationale="Duplicate chunk ID",
            ),
        ],
        cited_chunk_ids=["valid_chunk_1"],
    )

    llm = MockLLMClient(canned_responses={VerificationAssessment: canned_assessment})
    node = VerifyEvidenceNode(llm_client=llm)

    valid_chunk_1 = make_result("valid_chunk_1", score=0.9)
    state: AgenticGraphState = {
        "original_query": "Check single chunk contradiction",
        "accumulated_results": [valid_chunk_1],
        "hop_count": 1,
    }
    update = await node(state)

    assessment: VerificationAssessment = update["latest_verification"]
    # Deduplication leaves 1 chunk -> discarded
    assert len(assessment.detected_contradictions) == 0


# ==============================================================================
# 4. Corrective Retrieval Node Tests
# ==============================================================================


@pytest.mark.anyio
async def test_corrective_retrieval_node_plans_next_hop() -> None:
    """Test CorrectiveRetrievalNode plans recommended next query and increments hop_count."""
    node = CorrectiveRetrievalNode()

    verification = VerificationAssessment(
        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
        is_sufficient=False,
        reasoning="Missing enterprise pricing",
        missing_aspects=["Enterprise tier pricing"],
        recommended_next_query="Snowflake Enterprise pricing details",
    )

    state: AgenticGraphState = {
        "original_query": "Snowflake pricing",
        "hop_count": 1,
        "max_hops": 3,
        "company_id": "snowflake",
        "latest_verification": verification,
        "missing_aspects": ["Enterprise tier pricing"],
    }
    update = await node(state)

    assert update["hop_count"] == 2
    assert update["corrective_query"] == "Snowflake Enterprise pricing details"
    assert len(update["current_sub_queries"]) == 1
    assert update["current_sub_queries"][0].sub_query == "Snowflake Enterprise pricing details"
    assert update["current_sub_queries"][0].target_company_ids == ["snowflake"]


@pytest.mark.anyio
async def test_corrective_retrieval_node_max_hops_termination() -> None:
    """Test CorrectiveRetrievalNode terminates when hop_count reaches max_hops."""
    node = CorrectiveRetrievalNode()

    state: AgenticGraphState = {
        "original_query": "Snowflake pricing",
        "hop_count": 3,
        "max_hops": 3,
    }
    update = await node(state)

    assert update.get("termination_reason") == "max_hops_reached"
    assert "current_sub_queries" not in update
