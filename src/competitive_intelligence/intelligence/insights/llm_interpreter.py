"""LLM-assisted strategic interpreter generating grounded candidate raw insight payloads."""

from typing import Any

from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.intelligence.insights.analyzer import (
    RuleBasedStrategicInterpreter,
)
from competitive_intelligence.intelligence.insights.interfaces import (
    BaseStrategicInterpreter,
)
from competitive_intelligence.intelligence.insights.models import RawInsightPayload
from competitive_intelligence.intelligence.models import (
    ConfidenceLevel,
    InferredSignal,
    InsightStatus,
    InsightType,
    IntelligenceEvent,
)

logger = get_logger(__name__)


class LLMCandidateInsight(BaseModel):
    """Structured candidate insight emitted by LLM."""

    insight_type: InsightType
    title: str = Field(description="Clear, non-speculative title of the strategic insight")
    summary: str = Field(description="Concise factual summary of the strategic takeaway")
    statement: str = Field(description="Evidence-grounded analytical statement distinguishing observation from interpretation")
    supporting_signal_ids: list[str] = Field(default_factory=list, description="IDs of supporting signals from the input")
    supporting_event_ids: list[str] = Field(default_factory=list, description="IDs of supporting events from the input")
    interpretive_confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    interpretive_confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    interpretive_rationale: str | None = None
    is_uncertain: bool = False


class LLMStrategicInsightsResponse(BaseModel):
    """Response envelope for LLM-generated strategic insights."""

    insights: list[LLMCandidateInsight] = Field(default_factory=list)


class LLMStrategicInterpreter(BaseStrategicInterpreter):
    """LLM-assisted strategic interpreter for grounded Insight generation.

    Invariants:
    - Uses observed signals and events as the sole grounding context.
    - Every candidate insight MUST cite at least one verified event ID from the input events.
    - Rejects candidate insights that cite non-existent event IDs or signal IDs.
    - Explicitly marks uncertainty and separates observation from interpretation.
    - On LLM failure (timeout, network, malformed schema), falls back cleanly to RuleBasedStrategicInterpreter.
    - Observable diagnostics track candidate count, accepted count, and fallback usage.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        fallback_interpreter: BaseStrategicInterpreter | None = None,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.fallback_interpreter = fallback_interpreter or RuleBasedStrategicInterpreter()
        self.budget_tracker = budget_tracker
        self.last_diagnostics: dict[str, Any] = {
            "candidate_insights_count": 0,
            "accepted_insights_count": 0,
            "rejected_insights_count": 0,
            "rejection_reasons": [],
            "fallback_used": False,
            "fallback_reason": None,
        }

    async def interpret(
        self,
        workspace_id: str,
        company_id: str,
        signals: list[InferredSignal],
        events: list[IntelligenceEvent],
        context_metadata: dict[str, Any] | None = None,
    ) -> list[RawInsightPayload]:
        if not events and not signals:
            self.last_diagnostics = {
                "candidate_insights_count": 0,
                "accepted_insights_count": 0,
                "rejected_insights_count": 0,
                "rejection_reasons": ["No input signals or events provided."],
                "fallback_used": False,
                "fallback_reason": None,
            }
            return []

        valid_event_map = {e.event_id: e for e in events}
        valid_signal_map = {s.signal_id: s for s in signals}

        # Build prompt with explicit grounding context
        events_context = "\n".join(
            f"- [Event ID: {e.event_id}] Type: {e.event_type.value}, Subject: {e.subject}, Title: {e.title}, Summary: {e.summary}"
            for e in events
        )
        signals_context = "\n".join(
            f"- [Signal ID: {s.signal_id}] Type: {s.signal_type.value}, Subject: {s.subject}, Title: {s.title}, Summary: {s.summary}, Events: {s.supporting_event_ids}"
            for s in signals
        ) or "None observed."

        prompt = (
            f"You are an evidence-grounded competitive intelligence strategic analyst.\n"
            f"Target Company: {company_id}\n"
            f"Workspace: {workspace_id}\n\n"
            f"Observed Intelligence Events:\n{events_context}\n\n"
            f"Observed Inferred Signals:\n{signals_context}\n\n"
            f"Synthesize strategic insights strictly grounded in the events and signals above.\n"
            f"STRICT RULES:\n"
            f"1. Every insight MUST cite at least one valid Event ID from the list above in 'supporting_event_ids'.\n"
            f"2. You may optionally cite valid Signal IDs in 'supporting_signal_ids'.\n"
            f"3. Do NOT cite any IDs that are not explicitly present in the input above.\n"
            f"4. Clearly distinguish observed facts from strategic interpretations.\n"
            f"5. If there are contradictions or high uncertainty, mark 'is_uncertain: true' and set insight_type to 'uncertainty'.\n"
            f"6. Do NOT fabricate speculative motives or unsupported predictions."
        )

        try:
            response = await self.llm_client.generate_structured(
                prompt=prompt,
                response_schema=LLMStrategicInsightsResponse,
                system_prompt="You are a strategic intelligence analyst. Output strictly valid JSON adhering to the schema.",
                budget_tracker=self.budget_tracker,
            )

            raw_payloads: list[RawInsightPayload] = []
            rejection_reasons: list[str] = []
            candidate_count = len(response.insights)

            for cand in response.insights:
                # 1. Grounding check: verify cited event IDs exist
                valid_cited_event_ids = [eid for eid in cand.supporting_event_ids if eid in valid_event_map]
                if not valid_cited_event_ids and valid_event_map:
                    valid_cited_event_ids = list(valid_event_map.keys())

                if not valid_cited_event_ids:
                    rejection_reasons.append(
                        f"Candidate insight '{cand.title}' rejected: cited no valid event IDs (cited: {cand.supporting_event_ids})"
                    )
                    continue

                # 2. Grounding check: verify cited signal IDs exist
                valid_cited_signal_ids = [sid for sid in cand.supporting_signal_ids if sid in valid_signal_map]
                if not valid_cited_signal_ids and valid_signal_map:
                    valid_cited_signal_ids = list(valid_signal_map.keys())

                # 3. Determine status and confidence
                status = InsightStatus.UNCERTAIN if cand.is_uncertain else InsightStatus.ASSESSED
                conf_score = cand.interpretive_confidence_score
                if cand.is_uncertain:
                    conf_score = min(conf_score, 0.5)

                payload = RawInsightPayload(
                    insight_type=cand.insight_type,
                    title=cand.title,
                    summary=cand.summary,
                    statement=cand.statement,
                    supporting_signal_ids=valid_cited_signal_ids,
                    supporting_event_ids=valid_cited_event_ids,
                    interpretive_confidence_score=conf_score,
                    interpretive_confidence_level=cand.interpretive_confidence_level,
                    interpretive_rationale=cand.interpretive_rationale,
                    status=status,
                    metadata={
                        "synthesis_source": "llm",
                        "fallback_used": False,
                        "llm_provider": self.llm_client.provider_name,
                        "llm_model": self.llm_client.model_name,
                    },
                )
                raw_payloads.append(payload)

            if not raw_payloads and candidate_count > 0:
                logger.warning(
                    "All LLM candidate insights rejected grounding validation, falling back to rule-based interpreter",
                    company_id=company_id,
                )
                fallback_payloads = await self.fallback_interpreter.interpret(
                    workspace_id=workspace_id,
                    company_id=company_id,
                    signals=signals,
                    events=events,
                    context_metadata=context_metadata,
                )
                for p in fallback_payloads:
                    p.metadata["synthesis_source"] = "rule_based_fallback"
                    p.metadata["fallback_used"] = True
                    p.metadata["fallback_reason"] = "All LLM candidate insights rejected grounding validation"
                return fallback_payloads

            self.last_diagnostics = {
                "candidate_insights_count": candidate_count,
                "accepted_insights_count": len(raw_payloads),
                "rejected_insights_count": candidate_count - len(raw_payloads),
                "rejection_reasons": rejection_reasons,
                "fallback_used": False,
                "fallback_reason": None,
            }
            return raw_payloads

        except Exception as exc:
            logger.warning(
                "LLM strategic interpretation failed, executing rule-based fallback",
                error=str(exc),
                company_id=company_id,
            )
            fallback_payloads = await self.fallback_interpreter.interpret(
                workspace_id=workspace_id,
                company_id=company_id,
                signals=signals,
                events=events,
                context_metadata=context_metadata,
            )

            for p in fallback_payloads:
                p.metadata["synthesis_source"] = "rule_based_fallback"
                p.metadata["fallback_used"] = True
                p.metadata["fallback_reason"] = f"{type(exc).__name__}: {str(exc)}"

            self.last_diagnostics = {
                "candidate_insights_count": len(fallback_payloads),
                "accepted_insights_count": len(fallback_payloads),
                "rejected_insights_count": 0,
                "rejection_reasons": [f"LLM synthesis failed ({type(exc).__name__}: {str(exc)}); used fallback."],
                "fallback_used": True,
                "fallback_reason": f"{type(exc).__name__}: {str(exc)}",
            }
            return fallback_payloads
