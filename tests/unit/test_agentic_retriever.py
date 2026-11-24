"""Unit tests for AgenticRetriever adapter, BaseRetriever compatibility, and fallback behavior."""

from typing import Any
import pytest

from competitive_intelligence.agentic_rag.llm.errors import (
    LLMResponseMalformedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.models import (
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.errors import BudgetExceededError
from competitive_intelligence.orchestration.interfaces import StageExecutionContext
from competitive_intelligence.orchestration.models import WorkflowInputs, WorkflowRun
from competitive_intelligence.orchestration.policies import ExecutionBudget
from competitive_intelligence.orchestration.stages.retrieval_stage import RetrievalStage
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)


class MockAdaptiveRetriever(BaseRetriever):
    """Mock AdaptiveRetriever that tracks received queries and returns canned responses."""

    def __init__(self, canned_data: dict[str, list[RetrievalResult]] | None = None) -> None:
        self.canned_data = canned_data or {}
        self.recorded_queries: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.recorded_queries.append(query)
        if query.query_text in self.canned_data:
            return [r.model_copy(deep=True) for r in self.canned_data[query.query_text]]

        # Default fallback item
        return [
            RetrievalResult(
                chunk_id=f"fallback_chunk_{query.query_text[:10]}",
                score=0.88,
                text=f"Fallback evidence for: {query.query_text}",
                document_id="doc_fallback",
                version_id="ver_fallback",
                workspace_id=query.metadata_filter.workspace_id,
                company_id=query.metadata_filter.company_ids[0] if query.metadata_filter.company_ids else None,
                metadata={"routing_rule": "DeterministicFallbackRule"},
            )
        ]


def create_result(chunk_id: str, score: float = 0.95, text: str = "Evidence text") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        document_id="doc_1",
        version_id="ver_1",
        workspace_id="ws_prod",
        company_id="comp_1",
        canonical_url="https://example.com/doc1",
        metadata={"routing_strategy": "hybrid_rrf"},
    )


@pytest.fixture
def base_query() -> RetrievalQuery:
    return RetrievalQuery(
        query_text="What are Acme Corp latest pricing changes?",
        metadata_filter=MetadataFilter(
            workspace_id="ws_tenant_secure",
            company_ids=["acme_corp"],
        ),
        top_k=5,
    )


# ==============================================================================
# 1. BaseRetriever Contract and RetrievalStage Compatibility
# ==============================================================================


def test_agentic_retriever_is_base_retriever() -> None:
    """Verify AgenticRetriever strictly implements the BaseRetriever contract."""
    mock_engine = MockAdaptiveRetriever()
    agentic = AgenticRetriever(executor_retriever=mock_engine)

    assert isinstance(agentic, BaseRetriever)
    assert hasattr(agentic, "retrieve")
    assert hasattr(agentic, "aretrieve")


@pytest.mark.anyio
async def test_retrieval_stage_accepts_agentic_retriever_seamlessly(base_query: RetrievalQuery) -> None:
    """Verify RetrievalStage executes AgenticRetriever without requiring downstream changes."""
    item = create_result("chunk_agentic_1", score=0.99)
    mock_engine = MockAdaptiveRetriever({"What are Acme Corp latest pricing changes?": [item]})

    agentic_retriever = AgenticRetriever(executor_retriever=mock_engine)
    retrieval_stage = RetrievalStage(retriever=agentic_retriever)

    # Construct standard StageExecutionContext
    context = StageExecutionContext(
        run=WorkflowRun(
            workflow_id="wf_1",
            workspace_id="ws_tenant_secure",
            company_id="acme_corp",
        ),
        inputs=WorkflowInputs(
            workspace_id=base_query.metadata_filter.workspace_id,
            company_id="acme_corp",
            query_text=base_query.query_text,
        ),
        budget=ExecutionBudget(),
    )

    record = await retrieval_stage.execute(context)

    assert record.status.value == "completed"
    assert context.retrieval_results is not None
    assert len(context.retrieval_results) == 1
    assert context.retrieval_results[0].chunk_id == "chunk_agentic_1"


# ==============================================================================
# 2. Workspace and Tenancy Isolation
# ==============================================================================


@pytest.mark.anyio
async def test_workspace_isolation_enforced_across_all_hops(base_query: RetrievalQuery) -> None:
    """Verify that every subquery strictly preserves original workspace_id."""
    mock_engine = MockAdaptiveRetriever()
    agentic = AgenticRetriever(executor_retriever=mock_engine)

    results = await agentic.retrieve(base_query)

    assert len(results) > 0
    # Check that all recorded queries to underlying engine inherited workspace_id
    for rec_query in mock_engine.recorded_queries:
        assert rec_query.metadata_filter.workspace_id == "ws_tenant_secure"
        assert rec_query.metadata_filter.company_ids == ["acme_corp"]


# ==============================================================================
# 3. Provenance Preservation and Enrichment
# ==============================================================================


@pytest.mark.anyio
async def test_provenance_preservation_and_agentic_enrichment(base_query: RetrievalQuery) -> None:
    """Verify core retrieval provenance is preserved and agentic provenance is attached."""
    orig_res = create_result("chunk_prov_100", score=0.93)
    orig_res.canonical_url = "https://acme.com/pricing"
    mock_engine = MockAdaptiveRetriever({"What are Acme Corp latest pricing changes?": [orig_res]})

    agentic = AgenticRetriever(executor_retriever=mock_engine)
    results = await agentic.retrieve(base_query)

    assert len(results) == 1
    res = results[0]
    # Core fields preserved
    assert res.chunk_id == "chunk_prov_100"
    assert res.score == 0.93
    assert res.canonical_url == "https://acme.com/pricing"
    assert res.workspace_id == "ws_prod"

    # Agentic provenance attached
    assert "agentic_hop" in res.metadata
    assert "agentic_verification_status" in res.metadata
    assert res.metadata["agentic_verification_status"] == "sufficient"


# ==============================================================================
# 4. Atomic Fallback Behavior
# ==============================================================================


@pytest.mark.anyio
async def test_atomic_fallback_on_llm_unavailable(base_query: RetrievalQuery) -> None:
    """Test clean atomic fallback when LLM service is unavailable."""
    mock_engine = MockAdaptiveRetriever()
    failing_llm = MockLLMClient(simulate_failure=LLMUnavailableError("API Offline"))

    agentic = AgenticRetriever(
        executor_retriever=mock_engine,
        llm_client=failing_llm,
    )

    results = await agentic.retrieve(base_query)

    # Fallback executed
    assert len(results) > 0
    assert results[0].metadata.get("agentic_fallback") is True
    assert "LLMUnavailableError" in str(results[0].metadata.get("agentic_fallback_reason"))


@pytest.mark.anyio
async def test_atomic_fallback_on_timeout(base_query: RetrievalQuery) -> None:
    """Test clean atomic fallback when agentic execution exceeds timeout."""
    mock_engine = MockAdaptiveRetriever()
    failing_llm = MockLLMClient(simulate_failure=LLMTimeoutError("Request timed out"))

    agentic = AgenticRetriever(
        executor_retriever=mock_engine,
        llm_client=failing_llm,
        timeout_seconds=0.001,
    )

    results = await agentic.retrieve(base_query)

    assert len(results) > 0
    assert results[0].metadata.get("agentic_fallback") is True


@pytest.mark.anyio
async def test_atomic_fallback_on_budget_exhaustion(base_query: RetrievalQuery) -> None:
    """Test clean atomic fallback when token budget is exhausted."""
    mock_engine = MockAdaptiveRetriever()
    # Tiny token budget to force BudgetExceededError
    agentic = AgenticRetriever(
        executor_retriever=mock_engine,
        llm_client=MockLLMClient(prompt_tokens=500),
        task_token_budget=100,
    )

    results = await agentic.retrieve(base_query)

    assert len(results) > 0
    assert results[0].metadata.get("agentic_fallback") is True
    assert "BudgetExceededError" in str(results[0].metadata.get("agentic_fallback_reason"))


# ==============================================================================
# 5. Circular Retriever Safety and Budget Enforcement
# ==============================================================================


def test_agentic_retriever_circular_safety_forbidden() -> None:
    """Verify AgenticRetriever cannot accept another AgenticRetriever as its executor."""
    mock_engine = MockAdaptiveRetriever()
    agentic_inner = AgenticRetriever(executor_retriever=mock_engine)

    with pytest.raises(ValueError, match="circular dependency"):
        AgenticRetriever(executor_retriever=agentic_inner)


@pytest.mark.anyio
async def test_subquery_budget_exhaustion_skips_verification_llm_call(base_query: RetrievalQuery) -> None:
    """Regression test proving budget exhaustion during subquery execution does not result in another LLM call."""
    res1 = create_result("chunk_budget_test_1", score=0.92)
    mock_engine = MockAdaptiveRetriever({base_query.query_text: [res1]})

    # LLM tracks all calls made
    llm = MockLLMClient()

    # AgenticRetriever configured with max_hops=0 so that subquery execution immediately exhausts hop budget
    agentic = AgenticRetriever(
        executor_retriever=mock_engine,
        llm_client=llm,
        max_hops=0,
    )

    results = await agentic.retrieve(base_query)

    # Verify query analysis happened (1 call)
    analysis_calls = [c for c in llm.call_history if c["response_schema"] == QueryAnalysis]
    assert len(analysis_calls) == 1

    # Verify verification LLM call was NEVER made (0 calls) due to immediate short-circuiting
    verification_calls = [c for c in llm.call_history if c["response_schema"] == VerificationAssessment]
    assert len(verification_calls) == 0

    # Results returned cleanly
    assert len(results) > 0
