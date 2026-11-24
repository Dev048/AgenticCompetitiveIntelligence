"""Unit tests for the LLM abstraction boundary, budget tracking, and error handling."""

import pytest
from pydantic import BaseModel

from competitive_intelligence.agentic_rag.llm.errors import (
    LLMError,
    LLMResponseMalformedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.llm.openai_client import OpenAILLMClient
from competitive_intelligence.agentic_rag.models import (
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.errors import BudgetExceededError


class SampleSchema(BaseModel):
    name: str
    count: int


@pytest.mark.anyio
async def test_mock_llm_default_query_analysis() -> None:
    """Test MockLLMClient default deterministic QueryAnalysis generation."""
    client = MockLLMClient()
    result = await client.generate_structured("Test query", QueryAnalysis)

    assert isinstance(result, QueryAnalysis)
    assert result.decision == RetrievalDecision.DIRECT_RETRIEVAL
    assert len(result.sub_queries) == 1
    assert result.sub_queries[0].sub_query == "Test query"


@pytest.mark.anyio
async def test_mock_llm_canned_response() -> None:
    """Test MockLLMClient returning configured canned response."""
    canned = QueryAnalysis(
        original_query="q",
        decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
        reasoning="Canned reasoning",
        sub_queries=[
            SubQueryPlan(sub_query="sub1"),
            SubQueryPlan(sub_query="sub2"),
        ],
    )
    client = MockLLMClient(canned_responses={QueryAnalysis: canned})
    result = await client.generate_structured("q", QueryAnalysis)

    assert result == canned
    assert len(result.sub_queries) == 2


@pytest.mark.anyio
async def test_mock_llm_custom_handler() -> None:
    """Test MockLLMClient using custom handler function."""
    def handler(prompt: str, schema: type[BaseModel]) -> BaseModel:
        return SampleSchema(name=f"handled:{prompt}", count=42)

    client = MockLLMClient(custom_handler=handler)
    result = await client.generate_structured("my_prompt", SampleSchema)

    assert isinstance(result, SampleSchema)
    assert result.name == "handled:my_prompt"
    assert result.count == 42


@pytest.mark.anyio
async def test_mock_llm_simulated_failure() -> None:
    """Test MockLLMClient raising simulated failure."""
    client = MockLLMClient(simulate_failure=LLMUnavailableError("Service down"))

    with pytest.raises(LLMUnavailableError, match="Service down"):
        await client.generate_structured("q", QueryAnalysis)


@pytest.mark.anyio
async def test_mock_llm_budget_tracking() -> None:
    """Test MockLLMClient records tokens and respects budget limits."""
    tracker = LLMBudgetTracker(task_id="t1", max_llm_calls=2, max_tokens=200)
    client = MockLLMClient(prompt_tokens=40, completion_tokens=20)

    # Call 1: 60 tokens total
    await client.generate_structured("q1", QueryAnalysis, budget_tracker=tracker)
    assert tracker.usage.llm_calls == 1
    assert tracker.usage.total_tokens == 60

    # Call 2: 120 tokens total
    await client.generate_structured("q2", QueryAnalysis, budget_tracker=tracker)
    assert tracker.usage.llm_calls == 2
    assert tracker.usage.total_tokens == 120

    # Call 3: Exceeds max_llm_calls (2)
    with pytest.raises(BudgetExceededError, match="exceeded max LLM calls limit"):
        await client.generate_structured("q3", QueryAnalysis, budget_tracker=tracker)


@pytest.mark.anyio
async def test_openai_client_missing_key_raises_unavailable() -> None:
    """Test OpenAILLMClient raises LLMUnavailableError when no API key is provided."""
    client = OpenAILLMClient(api_key=None)
    # Force empty key
    client.api_key = None

    with pytest.raises(LLMUnavailableError, match="API key not configured"):
        await client.generate_structured("q", QueryAnalysis)

    with pytest.raises(LLMUnavailableError, match="API key not configured"):
        await client.generate_text("q")
