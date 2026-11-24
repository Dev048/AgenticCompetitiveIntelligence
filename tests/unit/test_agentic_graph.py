"""Unit tests for LangGraph state machine assembly, workflow execution, and termination conditions."""

from typing import Any
import pytest

from competitive_intelligence.agentic_rag.graph.workflow import (
    AgenticRAGWorkflow,
    build_agentic_rag_graph,
    should_continue,
)
from competitive_intelligence.agentic_rag.graph.state import AgenticGraphState
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.models import (
    AgenticRAGState,
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)


class MockEngineRetriever(BaseRetriever):
    """Mock retriever returning predetermined results keyed by subquery strings."""

    def __init__(self, data: dict[str, list[RetrievalResult]]) -> None:
        self.data = data
        self.call_history: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.call_history.append(query)
        return self.data.get(query.query_text, [])


def make_result(chunk_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"Text for chunk {chunk_id}",
        document_id="doc_1",
        version_id="ver_1",
        workspace_id="ws_test",
        company_id="comp_test",
    )


def test_should_continue_logic() -> None:
    """Test conditional edge routing logic for should_continue."""
    # 1. Fallback state terminates immediately
    assert should_continue({"is_fallback": True}) == "__end__"

    # 2. Termination reason terminates
    assert should_continue({"termination_reason": "budget_exhausted"}) == "__end__"

    # 3. Sufficient verification terminates
    sufficient = VerificationAssessment(
        decision=SufficiencyDecision.SUFFICIENT,
        is_sufficient=True,
        reasoning="All good",
    )
    assert should_continue({"latest_verification": sufficient, "hop_count": 1, "max_hops": 3}) == "__end__"

    # 4. Insufficient with hops remaining continues to corrective_retrieval
    insufficient = VerificationAssessment(
        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
        is_sufficient=False,
        reasoning="Need more info",
    )
    assert should_continue({"latest_verification": insufficient, "hop_count": 1, "max_hops": 3}) == "corrective_retrieval"

    # 5. Insufficient but max_hops reached terminates
    assert should_continue({"latest_verification": insufficient, "hop_count": 3, "max_hops": 3}) == "__end__"


@pytest.mark.anyio
async def test_workflow_single_hop_flow() -> None:
    """Test standard single-hop workflow execution to completion."""
    res = make_result("chunk_alpha", score=0.92)
    engine = MockEngineRetriever({"What is Acme revenue?": [res]})

    llm = MockLLMClient()  # Default: direct retrieval & sufficient verification
    workflow = AgenticRAGWorkflow(retriever=engine, llm_client=llm)

    query = RetrievalQuery(
        query_text="What is Acme revenue?",
        metadata_filter=MetadataFilter(workspace_id="ws_1", company_ids=["acme"]),
        top_k=5,
    )

    state: AgenticRAGState = await workflow.run(query=query, max_hops=3)

    assert state.hops_executed == 1
    assert len(state.accumulated_evidence) == 1
    assert state.accumulated_evidence[0].chunk_id == "chunk_alpha"
    assert len(state.verifications) == 1
    assert state.verifications[0].is_sufficient is True


@pytest.mark.anyio
async def test_workflow_multi_hop_corrective_flow() -> None:
    """Test multi-hop corrective workflow where hop 1 is insufficient and hop 2 succeeds."""
    res1 = make_result("chunk_hop1", score=0.7)
    res2 = make_result("chunk_hop2", score=0.95)

    engine = MockEngineRetriever({
        "initial query": [res1],
        "initial query missing details": [res2],
    })

    call_count = 0

    def custom_llm_handler(prompt: str, schema: type[Any]) -> Any:
        nonlocal call_count
        call_count += 1

        if schema is QueryAnalysis:
            return QueryAnalysis(
                original_query="initial query",
                decision=RetrievalDecision.DIRECT_RETRIEVAL,
                reasoning="Direct search",
                sub_queries=[SubQueryPlan(sub_query="initial query")],
            )

        if schema is VerificationAssessment:
            # On first verification (call 2), return insufficient
            if "chunk_hop1" in prompt and "chunk_hop2" not in prompt:
                return VerificationAssessment(
                    decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                    is_sufficient=False,
                    reasoning="Missing key details",
                    missing_aspects=["missing details"],
                    recommended_next_query="initial query missing details",
                )
            # On second verification, return sufficient
            return VerificationAssessment(
                decision=SufficiencyDecision.SUFFICIENT,
                is_sufficient=True,
                reasoning="Now sufficient with hop 2 data",
                cited_chunk_ids=["chunk_hop2"],
            )

        return schema()

    llm = MockLLMClient(custom_handler=custom_llm_handler)
    workflow = AgenticRAGWorkflow(retriever=engine, llm_client=llm)

    query = RetrievalQuery(
        query_text="initial query",
        metadata_filter=MetadataFilter(workspace_id="ws_1"),
        top_k=5,
    )

    state: AgenticRAGState = await workflow.run(query=query, max_hops=3)

    assert state.hops_executed == 2
    assert len(state.accumulated_evidence) == 2
    assert len(state.verifications) == 2
    assert state.verifications[0].is_sufficient is False
    assert state.verifications[1].is_sufficient is True
    assert len(engine.call_history) == 2


@pytest.mark.anyio
async def test_workflow_max_hops_termination() -> None:
    """Test workflow terminates cleanly when max_hops is exhausted without infinite looping."""
    res = make_result("chunk_sparse", score=0.4)
    engine = MockEngineRetriever({"never_sufficient": [res]})

    # Always return insufficient verification
    always_insufficient = VerificationAssessment(
        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
        is_sufficient=False,
        reasoning="Never enough data",
        missing_aspects=["everything"],
    )
    llm = MockLLMClient(canned_responses={VerificationAssessment: always_insufficient})
    workflow = AgenticRAGWorkflow(retriever=engine, llm_client=llm)

    query = RetrievalQuery(
        query_text="never_sufficient",
        metadata_filter=MetadataFilter(workspace_id="ws_1"),
        top_k=5,
    )

    state: AgenticRAGState = await workflow.run(query=query, max_hops=2)

    # Must terminate at max_hops (2)
    assert state.hops_executed == 2
    assert state.verifications[-1].is_sufficient is False
    assert len(state.accumulated_evidence) >= 1
