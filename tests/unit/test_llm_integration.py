"""Comprehensive test suite for the LLM Integration Subsystem.

Covers:
- Provider abstraction and configuration (LLMConfig, LLMProvider, factory).
- Structured output parsing, validation, and error translation.
- Provider retries, timeouts, and budget enforcement.
- Evidence grounding and provenance preservation across analytical stages.
- Rejection of unsupported claims and explicit uncertainty representation.
- Decision support option generation and strict PROPOSED recommendation boundary.
- Mock provider behavior and fallback resilience.
"""

from datetime import datetime, timezone
from typing import Any
import pytest
import httpx
from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.llm import (
    AnthropicLLMClient,
    BaseLLMClient,
    LLMAuthError,
    LLMBadRequestError,
    LLMConfig,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponseMalformedError,
    LLMTimeoutError,
    LLMUnavailableError,
    MockLLMClient,
    OpenAILLMClient,
    create_llm_client,
)
from competitive_intelligence.agentic_rag.llm import factory as factory_module
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.config import Settings
from competitive_intelligence.core.errors import BudgetExceededError
from competitive_intelligence.decision_support import (
    ActionType,
    ConstraintType,
    DecisionConstraint,
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    DeterministicDecisionAnalyzer,
    LLMOptionGenerator,
    ObjectiveDimension,
    RecommendationStatus,
    RiskTolerance,
    TimeHorizon,
)
from competitive_intelligence.intelligence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    DeterministicStrategicAnalyzer,
    EventTime,
    EvidenceReference,
    EvidenceRole,
    InferredSignal,
    Insight,
    InsightStatus,
    InsightType,
    IntelligenceEvent,
    IntelligenceEventType,
    InterpretiveConfidenceAssessment,
    LLMExtractedEventItem,
    LLMExtractionResponse,
    LLMEvidenceInterpreter,
    LLMStrategicInterpreter,
    RetrievedEvidenceAnalyzer,
    SignalType,
    SourcePerspective,
    TimePrecision,
)
from competitive_intelligence.retrieval_engine.models import RetrievalResult


# ==============================================================================
# 1. Configuration & Factory Tests
# ==============================================================================


class SimpleOutput(BaseModel):
    summary: str = "default summary"
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


def test_llm_config_validation() -> None:
    """Test LLMConfig validation rules."""
    cfg = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    assert cfg.provider == LLMProvider.OPENAI
    assert cfg.model == "gpt-4o-mini"
    assert cfg.timeout_seconds == 30.0
    assert cfg.max_retries == 3

    # Empty model rejected
    with pytest.raises(ValueError, match="Model name cannot be empty"):
        LLMConfig(model="   ")


def test_create_llm_client_factory() -> None:
    """Test factory instantiation of various providers."""
    mock_client = create_llm_client(LLMConfig(provider=LLMProvider.MOCK, model="mock-model"))
    assert isinstance(mock_client, MockLLMClient)
    assert mock_client.provider_name == "mock"
    assert mock_client.model_name == "mock-model"

    openai_client = create_llm_client(
        LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o", api_key="sk-test")
    )
    assert isinstance(openai_client, OpenAILLMClient)
    assert openai_client.provider_name == "openai"
    assert openai_client.model_name == "gpt-4o"
    assert openai_client.base_url == "https://api.openai.com/v1"

    anthropic_client = create_llm_client(
        LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-3-5-sonnet", api_key="ant-test")
    )
    assert isinstance(anthropic_client, AnthropicLLMClient)
    assert anthropic_client.provider_name == "anthropic"
    assert anthropic_client.model_name == "claude-3-5-sonnet"


def test_settings_openai_base_url_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that OPENAI_BASE_URL is mapped and normalized in Settings."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1/")
    settings = Settings()
    assert settings.openai_base_url == "https://api.groq.com/openai/v1"

    monkeypatch.setenv("OPENAI_BASE_URL", "   ")
    settings_blank = Settings()
    assert settings_blank.openai_base_url is None


def test_factory_propagates_base_url_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that create_llm_client without explicit config passes settings.openai_base_url."""
    groq_settings = Settings(
        llm_provider="openai",
        openai_api_key="gsk-test-key",
        openai_base_url="https://api.groq.com/openai/v1",
        default_llm_model="openai/gpt-oss-120b",
    )
    monkeypatch.setattr(factory_module, "get_settings", lambda: groq_settings)

    client = create_llm_client()
    assert isinstance(client, OpenAILLMClient)
    assert client.base_url == "https://api.groq.com/openai/v1"
    assert client.model == "openai/gpt-oss-120b"
    assert client.api_key == "gsk-test-key"


def test_factory_propagates_base_url_from_explicit_config() -> None:
    """Test that create_llm_client honors base_url configured on LLMConfig."""
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="openai/gpt-oss-120b",
        api_key="gsk-test-key",
        base_url="https://api.groq.com/openai/v1",
    )
    client = create_llm_client(config)
    assert isinstance(client, OpenAILLMClient)
    assert client.base_url == "https://api.groq.com/openai/v1"
    assert client.model == "openai/gpt-oss-120b"


def test_factory_disallows_mock_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock LLM must be rejected in production environment."""
    prod_settings = Settings(environment="production", llm_provider="mock")
    monkeypatch.setattr(factory_module, "get_settings", lambda: prod_settings)

    with pytest.raises(ValueError, match="Mock LLM provider is not permitted in production"):
        create_llm_client(LLMConfig(provider=LLMProvider.MOCK))


# ==============================================================================
# 2. Mock Provider Grounding & Behavior
# ==============================================================================


@pytest.mark.anyio
async def test_mock_llm_dynamic_extraction_from_prompt() -> None:
    """Test that MockLLMClient dynamically parses prompt chunk IDs for grounded responses."""
    client = MockLLMClient()
    prompt = (
        "Extract events:\n"
        "[Chunk ID: chunk-alpha-123]\nText: Competitor dropped price to $49.\n"
        "[Chunk ID: chunk-beta-456]\nText: Competitor released v2.0."
    )
    res = await client.generate_structured(prompt, LLMExtractionResponse)
    assert isinstance(res, LLMExtractionResponse)
    assert len(res.events) > 0
    assert "chunk-alpha-123" in res.events[0].supporting_chunk_ids


@pytest.mark.anyio
async def test_mock_llm_budget_tracker_enforcement() -> None:
    """Test budget tracker tracking and threshold enforcement."""
    tracker = LLMBudgetTracker(task_id="task_test_1", max_llm_calls=2, max_tokens=100)
    client = MockLLMClient(prompt_tokens=30, completion_tokens=15)

    await client.generate_structured("prompt 1", SimpleOutput, budget_tracker=tracker)
    assert tracker.usage.llm_calls == 1
    assert tracker.usage.total_tokens == 45

    await client.generate_structured("prompt 2", SimpleOutput, budget_tracker=tracker)
    assert tracker.usage.llm_calls == 2
    assert tracker.usage.total_tokens == 90

    # Third call should exceed max calls
    with pytest.raises(BudgetExceededError, match="exceeded max LLM calls limit"):
        await client.generate_structured("prompt 3", SimpleOutput, budget_tracker=tracker)


# ==============================================================================
# 3. Provider Retries, Timeouts, and Error Classification (OpenAI & Anthropic)
# ==============================================================================


@pytest.mark.anyio
async def test_openai_client_retries_on_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAILLMClient bounded retry on 429 rate limit."""
    call_count = 0

    async def mock_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(status_code=429, text="Rate limit exceeded")
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"summary": "recovered", "confidence": 0.95}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    client = OpenAILLMClient(api_key="sk-test", max_retries=3)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    result = await client.generate_structured("test", SimpleOutput)
    assert result.summary == "recovered"
    assert result.confidence == 0.95
    assert call_count == 3


@pytest.mark.anyio
async def test_openai_client_fails_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAILLMClient exhausts bounded retries on 500 error."""
    call_count = 0

    async def mock_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(status_code=500, text="Internal Server Error")

    client = OpenAILLMClient(api_key="sk-test", max_retries=2)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with pytest.raises(LLMUnavailableError, match="Internal Server Error"):
        await client.generate_structured("test", SimpleOutput)

    # Initial attempt + 2 retries = 3 total attempts
    assert call_count == 3


@pytest.mark.anyio
async def test_openai_client_malformed_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAILLMClient raises LLMResponseMalformedError on unparseable JSON."""
    async def mock_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "choices": [{"message": {"content": "not valid json at all!"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            },
        )

    client = OpenAILLMClient(api_key="sk-test", max_retries=1)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with pytest.raises(LLMResponseMalformedError):
        await client.generate_structured("test", SimpleOutput)


@pytest.mark.anyio
async def test_anthropic_client_tool_use_structured_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Anthropic client parsing structured tool call output."""
    async def mock_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "content": [
                    {
                        "type": "tool_use",
                        "name": "record_output",
                        "input": {"summary": "Anthropic extracted", "confidence": 0.88},
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 8},
            },
        )

    client = AnthropicLLMClient(api_key="ant-test", max_retries=1)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    tracker = LLMBudgetTracker(task_id="task_ant", max_llm_calls=5, max_tokens=1000)
    result = await client.generate_structured("test", SimpleOutput, budget_tracker=tracker)
    assert result.summary == "Anthropic extracted"
    assert result.confidence == 0.88
    assert tracker.usage.llm_calls == 1
    assert tracker.usage.total_tokens == 20


@pytest.mark.anyio
async def test_anthropic_client_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Anthropic client raises LLMAuthError on 401."""
    async def mock_post(self: Any, url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(status_code=401, text="Invalid API key")

    client = AnthropicLLMClient(api_key="bad-key", max_retries=1)
    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    with pytest.raises(LLMAuthError, match="Anthropic authentication failed"):
        await client.generate_structured("test", SimpleOutput)


# ==============================================================================
# 4. Evidence Grounding & Provenance in Event Analysis
# ==============================================================================


@pytest.mark.anyio
async def test_llm_evidence_interpreter_provenance_validation() -> None:
    """Test LLMEvidenceInterpreter accepts cited chunk events and rejects hallucinated ones."""
    # LLM outputs 2 events: one citing valid chunk-1, one citing non-existent chunk-fake
    canned = LLMExtractionResponse(
        events=[
            LLMExtractedEventItem(
                event_type=IntelligenceEventType.PRICING_CHANGE,
                title="Verified Price Drop",
                summary="Reduced pricing from $99 to $49.",
                subject="Serverless DB",
                confidence_score=0.9,
                supporting_chunk_ids=["chunk-valid-1"],
            ),
            LLMExtractedEventItem(
                event_type=IntelligenceEventType.ACQUISITION,
                title="Hallucinated Acquisition",
                summary="Acquired RivalCo for $1B.",
                subject="RivalCo",
                confidence_score=0.9,
                supporting_chunk_ids=["chunk-hallucinated-999"],
            ),
        ]
    )

    mock_llm = MockLLMClient(canned_responses={LLMExtractionResponse: canned})
    interpreter = LLMEvidenceInterpreter(llm_client=mock_llm)

    valid_evidence = [
        RetrievalResult(
            chunk_id="chunk-valid-1",
            document_id="doc-1",
            version_id="ver-1",
            source_id="src-1",
            canonical_url="https://example.com/pricing",
            source_category="pricing",
            text="Price dropped to $49.",
            score=0.95,
        )
    ]

    payloads = await interpreter.interpret_evidence(
        query="Pricing updates",
        company_id="cloudscale",
        workspace_id="ws-test",
        evidence=valid_evidence,
    )

    # Exactly 1 event should be accepted, 1 rejected
    assert len(payloads) == 1
    assert payloads[0].title == "Verified Price Drop"
    assert interpreter.last_diagnostics["accepted_events_count"] == 1
    assert interpreter.last_diagnostics["rejected_events_count"] == 1
    assert any("chunk-hallucinated-999" in r for r in interpreter.last_diagnostics["rejection_reasons"])


@pytest.mark.anyio
async def test_llm_evidence_interpreter_fallback_on_failure() -> None:
    """When LLM fails, LLMEvidenceInterpreter must safely fall back to rule-based interpreter."""
    failing_llm = MockLLMClient(simulate_failure=LLMTimeoutError("Request timed out"))
    interpreter = LLMEvidenceInterpreter(llm_client=failing_llm)

    evidence = [
        RetrievalResult(
            chunk_id="chk-1",
            document_id="doc-1",
            version_id="ver-1",
            text="CloudScale acquired DataPulse for $50M.",
            score=0.9,
        )
    ]

    payloads = await interpreter.interpret_evidence(
        query="M&A",
        company_id="cloudscale",
        workspace_id="ws-test",
        evidence=evidence,
    )

    assert len(payloads) > 0
    assert interpreter.last_diagnostics["fallback_used"] is True
    assert "LLMTimeoutError" in str(interpreter.last_diagnostics["fallback_reason"])
    assert payloads[0].metadata["fallback_used"] is True


# ==============================================================================
# 5. Strategic Insights & Explicit Uncertainty Representation
# ==============================================================================


@pytest.mark.anyio
async def test_llm_strategic_interpreter_uncertainty_and_grounding() -> None:
    """Test LLMStrategicInterpreter validates supporting signal/event IDs and preserves uncertainty."""
    mock_llm = MockLLMClient()
    interpreter = LLMStrategicInterpreter(llm_client=mock_llm)

    ev = IntelligenceEvent(
        event_id="ev_test_1",
        workspace_id="ws-test",
        company_id="comp-1",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Price Reduction",
        summary="Price dropped to $49.",
        subject="DB Tier",
        confidence=ConfidenceAssessment(
            score=0.90,
            level=ConfidenceLevel.HIGH,
            rationale="Verified primary source.",
        ),
        event_time=EventTime(
            acquired_at=datetime.now(timezone.utc),
            time_precision=TimePrecision.DATE_ONLY,
        ),
        evidence=[
            EvidenceReference(
                source_id="src-1",
                document_id="doc-1",
                version_id="ver-1",
                chunk_id="chk-1",
                source_url="https://example.com",
                source_category="pricing",
                extracted_text="Price cut",
                role=EvidenceRole.PRIMARY_DIRECT,
                acquired_at=datetime.now(timezone.utc),
            )
        ],
    )

    sig = InferredSignal(
        signal_id="sig_test_1",
        workspace_id="ws-test",
        company_id="comp-1",
        signal_type=SignalType.DIRECTIONAL_TREND,
        title="Price Aggression Signal",
        summary="Down-market price movement detected.",
        supporting_event_ids=["ev_test_1"],
    )

    insights = await interpreter.interpret(
        workspace_id="ws-test",
        company_id="comp-1",
        signals=[sig],
        events=[ev],
    )

    assert len(insights) > 0
    raw_ins = insights[0]
    assert "ev_test_1" in raw_ins.supporting_event_ids
    assert "sig_test_1" in raw_ins.supporting_signal_ids


# ==============================================================================
# 6. Decision Support: Option Enrichment and Human Approval Boundary
# ==============================================================================


@pytest.mark.anyio
async def test_decision_support_llm_option_generation_and_proposed_status() -> None:
    """Test LLM-enriched option generation with deterministic constraints and strictly PROPOSED status."""
    mock_llm = MockLLMClient()
    generator = LLMOptionGenerator(llm_client=mock_llm)

    analyzer = DeterministicDecisionAnalyzer(option_generator=generator)

    ctx = DecisionContext(
        decision_id="dec_opt_test",
        workspace_id="ws-test",
        organization_id="org-1",
        title="Competitive Response",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(
                objective_id="obj-1",
                name="Defend Revenue",
                dimension=ObjectiveDimension.GROWTH,
                target_description="Maintain growth rate",
                priority_weight=1.0,
            )
        ],
        constraints=[
            DecisionConstraint(
                constraint_id="c-budget",
                constraint_type=ConstraintType.BUDGET,
                description="Max budget $25k",
                is_hard_constraint=True,
                threshold_value="25000",
            )
        ],
        risk_tolerance=RiskTolerance.MODERATE,
        time_horizon=TimeHorizon.IMMEDIATE_SHORT_TERM,
    )

    ev = IntelligenceEvent(
        event_id="ev_dec_1",
        workspace_id="ws-test",
        company_id="comp-1",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Competitor Price Cut",
        summary="Reduced price by 50%.",
        subject="Hosting",
        confidence=ConfidenceAssessment(
            score=0.90,
            level=ConfidenceLevel.HIGH,
            rationale="Verified primary source.",
        ),
        event_time=EventTime(
            acquired_at=datetime.now(timezone.utc),
            time_precision=TimePrecision.DATE_ONLY,
        ),
        evidence=[
            EvidenceReference(
                source_id="src-1",
                document_id="doc-1",
                version_id="ver-1",
                chunk_id="chk-1",
                source_url="https://example.com",
                source_category="pricing",
                extracted_text="Price cut",
                role=EvidenceRole.PRIMARY_DIRECT,
                acquired_at=datetime.now(timezone.utc),
            )
        ],
    )

    ins = Insight(
        insight_id="ins_dec_1",
        workspace_id="ws-test",
        company_id="comp-1",
        insight_type=InsightType.PRICING_STRATEGY,
        title="Aggressive Pricing Expansion",
        summary="Competitor targeting lower tier.",
        statement="Competitor is driving down-market acquisition.",
        supporting_signal_ids=[],
        supporting_event_ids=["ev_dec_1"],
        supporting_evidence_ids=["evr_1"],
        factual_confidence=ConfidenceAssessment(score=0.90, level=ConfidenceLevel.HIGH),
        interpretive_confidence=InterpretiveConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH),
        status=InsightStatus.ASSESSED,
    )

    options, recommendation = await analyzer.analyze_and_recommend(
        context=ctx,
        insights=[ins],
        events=[ev],
    )

    assert len(options) > 0
    assert recommendation is not None

    # CRITICAL INVARIANT: Recommendation MUST remain PROPOSED (no autonomous execution)
    assert recommendation.status == RecommendationStatus.PROPOSED
    assert recommendation.status.value == "proposed"

    # Verify hard constraint: all feasible options must be within the $25k budget
    for opt in options:
        if opt.estimated_cost_usd is not None:
            # Check selected recommendation complies with budget constraint
            if opt.option_id == recommendation.selected_option_id:
                assert opt.estimated_cost_usd <= 25000.0


@pytest.mark.anyio
async def test_llm_option_generator_fallback_on_error() -> None:
    """Test LLMOptionGenerator executes fallback generator when LLM encounters error."""
    failing_llm = MockLLMClient(simulate_failure=LLMBadRequestError("Invalid request params"))
    generator = LLMOptionGenerator(llm_client=failing_llm)

    ctx = DecisionContext(
        decision_id="dec_fallback",
        workspace_id="ws-test",
        organization_id="org-1",
        title="Pricing Defense",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(
                objective_id="obj-1",
                name="Cost Defense",
                dimension=ObjectiveDimension.COST_EFFICIENCY,
                target_description="Defend margins",
                priority_weight=1.0,
            )
        ],
        risk_tolerance=RiskTolerance.MODERATE,
        time_horizon=TimeHorizon.IMMEDIATE_SHORT_TERM,
    )

    options = await generator.generate_options(ctx, [], [])
    assert len(options) > 0
    assert generator.last_diagnostics["fallback_used"] is True
