"""Mock LLM client for deterministic unit testing, benchmarking, and offline runs."""

from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from competitive_intelligence.agentic_rag.llm.errors import (
    LLMError,
    LLMResponseMalformedError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.agentic_rag.models import (
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.metrics import track_llm_call

T = TypeVar("T", bound=BaseModel)


class MockLLMClient(BaseLLMClient):
    """Deterministic Mock LLM client with programmable responses and budget tracking."""

    def __init__(
        self,
        canned_responses: dict[type[Any], Any] | None = None,
        custom_handler: Callable[[str, type[Any]], Any] | None = None,
        simulate_failure: Exception | None = None,
        model: str = "mock-model",
        prompt_tokens: int = 50,
        completion_tokens: int = 30,
    ) -> None:
        self.canned_responses = canned_responses or {}
        self.custom_handler = custom_handler
        self.simulate_failure = simulate_failure
        self._model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.call_history: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model


    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> T:
        """Return a mocked structured response conforming to response_schema."""
        self.call_history.append({
            "prompt": prompt,
            "response_schema": response_schema,
            "system_prompt": system_prompt,
            "temperature": temperature,
        })

        if budget_tracker:
            budget_tracker.record_llm_call(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            )

        if self.simulate_failure is not None:
            track_llm_call(
                provider=self.provider_name,
                model=self.model_name,
                status="failure",
                duration_seconds=0.001,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=0,
            )
            raise self.simulate_failure

        track_llm_call(
            provider=self.provider_name,
            model=self.model_name,
            status="success",
            duration_seconds=0.001,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )

        if self.custom_handler is not None:
            result = self.custom_handler(prompt, response_schema)
            if isinstance(result, response_schema):
                return result
            if isinstance(result, dict):
                return response_schema.model_validate(result)
            raise LLMResponseMalformedError(f"Custom handler returned invalid type: {type(result)}")

        if response_schema in self.canned_responses:
            val = self.canned_responses[response_schema]
            if isinstance(val, response_schema):
                return val
            if isinstance(val, dict):
                return response_schema.model_validate(val)
            raise LLMResponseMalformedError(f"Canned response invalid for {response_schema}: {val}")

        # Default deterministic mock responses for standard Agentic RAG schemas
        if response_schema is QueryAnalysis:
            query_text = prompt
            if "Original Query: " in prompt:
                query_text = prompt.split("Original Query: ", 1)[1].split("\n", 1)[0].strip()

            return response_schema(
                original_query=query_text,
                decision=RetrievalDecision.DIRECT_RETRIEVAL,
                reasoning="Mock default query analysis: direct retrieval",
                sub_queries=[
                    SubQueryPlan(
                        sub_query=query_text,
                        rationale="Direct retrieval of input query",
                    )
                ],
            )

        if response_schema is VerificationAssessment:
            return response_schema(
                decision=SufficiencyDecision.SUFFICIENT,
                is_sufficient=True,
                reasoning="Mock default evidence verification: sufficient evidence",
                detected_contradictions=[],
                missing_aspects=[],
                cited_chunk_ids=[],
            )

        if response_schema.__name__ == "LLMExtractionResponse":
            import re
            from competitive_intelligence.intelligence.analyzers.retrieved_evidence_analyzer import (
                LLMExtractedEventItem,
            )
            from competitive_intelligence.intelligence.models import (
                ConfidenceLevel,
                IntelligenceEventType,
                TimePrecision,
            )
            chunk_ids = re.findall(r"\[Chunk ID:\s*([^\]]+)\]", prompt)
            events = []
            if chunk_ids:
                comp_match = re.search(r"Target Company:\s*([^\n]+)", prompt)
                comp_name = comp_match.group(1).strip() if comp_match else "Target Competitor"

                lower_prompt = prompt.lower()
                if "acqui" in lower_prompt:
                    ev_type = IntelligenceEventType.ACQUISITION
                    ev_title = f"Acquisition: {comp_name}"
                    ev_summary = f"Observed acquisition activity involving {comp_name}."
                    ev_subject = "Acquisitions"
                elif "pric" in lower_prompt:
                    ev_type = IntelligenceEventType.PRICING_CHANGE
                    ev_title = f"Pricing Adjustment: {comp_name}"
                    ev_summary = f"Observed pricing adjustment for {comp_name}."
                    ev_subject = "Pricing"
                else:
                    ev_type = IntelligenceEventType.STRATEGIC_ANNOUNCEMENT
                    ev_title = f"Strategic Announcement: {comp_name}"
                    ev_summary = f"Strategic announcement observed for {comp_name}."
                    ev_subject = "Strategy"

                events.append(
                    LLMExtractedEventItem(
                        event_type=ev_type,
                        title=ev_title,
                        summary=ev_summary,
                        subject=ev_subject,
                        time_precision=TimePrecision.UNKNOWN,
                        confidence_score=0.88,
                        confidence_level=ConfidenceLevel.HIGH,
                        confidence_rationale="Grounded in retrieved evidence chunk.",
                        supporting_chunk_ids=chunk_ids[:2],
                    )
                )
            return response_schema(events=events)

        if response_schema.__name__ == "LLMStrategicInsightsResponse":
            import re
            from competitive_intelligence.intelligence.insights.llm_interpreter import (
                LLMCandidateInsight,
            )
            from competitive_intelligence.intelligence.models import (
                ConfidenceLevel,
                InsightType,
            )
            event_ids = re.findall(r"\[Event ID:\s*([^\]]+)\]", prompt)
            signal_ids = re.findall(r"\[Signal ID:\s*([^\]]+)\]", prompt)
            insights = []
            if event_ids:
                insights.append(
                    LLMCandidateInsight(
                        insight_type=InsightType.PRICING_STRATEGY,
                        title="Strategic Pricing Shift",
                        summary="Competitor exhibits strategic pricing adjustments across tiers.",
                        statement="Observed pricing actions reflect targeted positioning evolution.",
                        supporting_signal_ids=signal_ids[:1],
                        supporting_event_ids=event_ids[:2],
                        interpretive_confidence_score=0.85,
                        interpretive_confidence_level=ConfidenceLevel.HIGH,
                        interpretive_rationale="Synthesized from observed pricing change events.",
                        is_uncertain=False,
                    )
                )
            return response_schema(insights=insights)

        if response_schema.__name__ == "LLMOptionGenerationResponse":
            import re
            from competitive_intelligence.decision_support.llm_generator import (
                LLMCandidateOption,
                LLMCandidatePredictedConsequence,
                LLMCandidateRisk,
            )
            from competitive_intelligence.decision_support.models import (
                ActionType,
                ObjectiveDimension,
                RiskBasisType,
            )
            from competitive_intelligence.intelligence.models import (
                ConfidenceLevel,
                SeverityLevel,
            )
            event_ids = re.findall(r"\[Event ID:\s*([^\]]+)\]", prompt)
            insight_ids = re.findall(r"\[Insight ID:\s*([^\]]+)\]", prompt)
            options = [
                LLMCandidateOption(
                    title="Maintain Current Pricing & Monitor Conversion",
                    description="Keep existing pricing structure intact while actively tracking funnel conversion changes.",
                    action_type=ActionType.MONITOR_AND_WAIT,
                    estimated_cost_usd=0.0,
                    expected_effects=[
                        LLMCandidatePredictedConsequence(
                            dimension=ObjectiveDimension.COST_EFFICIENCY,
                            description="Preserves gross margins without incurring implementation or discounting costs.",
                            is_positive=True,
                        ),
                    ],
                    required_resources=["Telemetry monitoring"],
                    risks=[
                        LLMCandidateRisk(
                            description="Market share drag in entry segment.",
                            likelihood=ConfidenceLevel.MEDIUM,
                            severity=SeverityLevel.MEDIUM,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                            mitigation_strategy="Set weekly conversion alert thresholds.",
                        )
                    ],
                    tradeoffs=["Preserves unit margins but risks slower top-of-funnel acquisition."],
                    assumptions=["Competitor price reduction does not immediately trigger mass migration."],
                    supporting_insight_ids=insight_ids[:2],
                    supporting_event_ids=event_ids[:2],
                ),
                LLMCandidateOption(
                    title="Targeted Entry-Tier Price Match",
                    description="Introduce a matched entry-level pricing tier with bounded usage limits.",
                    action_type=ActionType.DIRECT_COUNTER_MOVE,
                    estimated_cost_usd=20000.0,
                    expected_effects=[
                        LLMCandidatePredictedConsequence(
                            dimension=ObjectiveDimension.GROWTH,
                            description="Neutralizes competitor pricing advantage and restores acquisition parity.",
                            is_positive=True,
                        ),
                    ],
                    required_resources=["Billing engineering team", "Updated pricing pages"],
                    risks=[
                        LLMCandidateRisk(
                            description="Down-market cannibalization of standard tier.",
                            likelihood=ConfidenceLevel.HIGH,
                            severity=SeverityLevel.HIGH,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                            mitigation_strategy="Gate advanced capabilities behind pro tier.",
                        )
                    ],
                    tradeoffs=["Accelerates market share defense at the expense of lower unit margins."],
                    assumptions=["Acquisition volume compensates for lower entry ARPU."],
                    supporting_insight_ids=insight_ids[:2],
                    supporting_event_ids=event_ids[:2],
                ),
            ]
            return response_schema(options=options)

        # Generic fallback instance if default constructible
        try:
            return response_schema()
        except Exception as e:
            raise LLMResponseMalformedError(
                f"No mock response configured for schema {response_schema.__name__}: {e}"
            ) from e


    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> str:
        """Return a mocked text completion."""
        self.call_history.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
        })

        if budget_tracker:
            budget_tracker.record_llm_call(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
            )

        if self.simulate_failure is not None:
            track_llm_call(
                provider=self.provider_name,
                model=self.model_name,
                status="failure",
                duration_seconds=0.001,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=0,
            )
            raise self.simulate_failure

        track_llm_call(
            provider=self.provider_name,
            model=self.model_name,
            status="success",
            duration_seconds=0.001,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )

        return "Mock LLM text completion"
