"""LLM-assisted candidate option generator implementing BaseOptionGenerator."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.decision_support.analyzer import (
    RuleBasedOptionGenerator,
)
from competitive_intelligence.decision_support.interfaces import BaseOptionGenerator
from competitive_intelligence.decision_support.models import (
    ActionType,
    DecisionContext,
    DecisionOption,
    DecisionRisk,
    ObjectiveDimension,
    PredictedConsequence,
    RiskBasisType,
)
from competitive_intelligence.intelligence.models import (
    ConfidenceLevel,
    Insight,
    IntelligenceEvent,
    SeverityLevel,
)

logger = get_logger(__name__)


class LLMCandidatePredictedConsequence(BaseModel):
    """Predicted effect of an action candidate."""

    dimension: ObjectiveDimension
    description: str
    is_positive: bool
    confidence_score: float | None = Field(default=0.8, ge=0.0, le=1.0)
    underlying_assumptions: list[str] = Field(default_factory=list)


class LLMCandidateRisk(BaseModel):
    """Predicted risk associated with an option candidate."""

    description: str
    likelihood: ConfidenceLevel = ConfidenceLevel.MEDIUM
    severity: SeverityLevel = SeverityLevel.MEDIUM
    basis_type: RiskBasisType = RiskBasisType.MODEL_PREDICTED
    mitigation_strategy: str | None = None


class LLMCandidateOption(BaseModel):
    """Structured decision option candidate emitted by LLM."""

    title: str
    description: str
    action_type: ActionType
    estimated_cost_usd: float | None = None
    expected_effects: list[LLMCandidatePredictedConsequence] = Field(default_factory=list)
    required_resources: list[str] = Field(default_factory=list)
    risks: list[LLMCandidateRisk] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    supporting_insight_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)


class LLMOptionGenerationResponse(BaseModel):
    """Response envelope for LLM-generated decision options."""

    options: list[LLMCandidateOption] = Field(default_factory=list)


class LLMOptionGenerator(BaseOptionGenerator):
    """LLM-assisted candidate option generator for Decision Support.

    Invariants:
    - Uses DecisionContext, business objectives, constraints, insights, and events as grounding context.
    - Every generated option MUST cite valid supporting insight and/or event IDs from the context.
    - Predicted consequences and risks are explicitly structured and labeled as predictions.
    - On LLM failure (timeout, network, malformed schema), falls back cleanly to RuleBasedOptionGenerator.
    - Generated options are passed to DeterministicDecisionAnalyzer for hard constraint filtering,
      objective alignment scoring, and recommendation validation.
    - Never automatically accepts or executes recommendations (status remains PROPOSED).
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        fallback_generator: BaseOptionGenerator | None = None,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.fallback_generator = fallback_generator or RuleBasedOptionGenerator()
        self.budget_tracker = budget_tracker
        self.last_diagnostics: dict[str, Any] = {
            "candidate_options_count": 0,
            "accepted_options_count": 0,
            "rejected_options_count": 0,
            "rejection_reasons": [],
            "fallback_used": False,
            "fallback_reason": None,
        }

    async def generate_options(
        self,
        context: DecisionContext,
        insights: list[Insight],
        events: list[IntelligenceEvent],
    ) -> list[DecisionOption]:
        if not context.business_objectives:
            self.last_diagnostics = {
                "candidate_options_count": 0,
                "accepted_options_count": 0,
                "rejected_options_count": 0,
                "rejection_reasons": ["No business objectives defined in decision context."],
                "fallback_used": False,
                "fallback_reason": None,
            }
            return []

        valid_insight_ids = {ins.insight_id for ins in insights}
        valid_event_map = {e.event_id: e for e in events}

        # Build prompt
        objectives_text = "\n".join(
            f"- Dimension: {obj.dimension.value}, Name: {obj.name}, Target: {obj.target_description}, Weight: {obj.priority_weight}"
            for obj in context.business_objectives
        )
        constraints_text = "\n".join(
            f"- Type: {c.constraint_type.value}, Desc: {c.description}, Hard: {c.is_hard_constraint}, Threshold: {c.threshold_value}"
            for c in context.constraints
        ) or "None defined."
        insights_text = "\n".join(
            f"- [Insight ID: {ins.insight_id}] Type: {ins.insight_type.value}, Title: {ins.title}, Statement: {ins.statement}"
            for ins in insights
        ) or "None available."
        events_text = "\n".join(
            f"- [Event ID: {e.event_id}] Type: {e.event_type.value}, Title: {e.title}, Summary: {e.summary}"
            for e in events
        ) or "None available."

        prompt = (
            f"You are a strategic decision analysis assistant.\n"
            f"Decision Title: {context.title}\n"
            f"Domain: {context.decision_domain.value}\n"
            f"Risk Tolerance: {context.risk_tolerance.value}\n"
            f"Time Horizon: {context.time_horizon.value}\n\n"
            f"Business Objectives:\n{objectives_text}\n\n"
            f"Operational Constraints:\n{constraints_text}\n\n"
            f"Grounded Market Insights:\n{insights_text}\n\n"
            f"Observed Intelligence Events:\n{events_text}\n\n"
            f"Generate candidate strategic decision options responding to these market conditions.\n"
            f"STRICT RULES:\n"
            f"1. Generate bounded, realistic options (e.g., monitor & wait, direct counter-move, differentiation, packaging adjustment).\n"
            f"2. Each option MUST cite relevant supporting insight IDs from the list above.\n"
            f"3. Each option MUST define expected effects across relevant objective dimensions, with positive/negative flag.\n"
            f"4. Each option MUST explicitly list risks with likelihood and severity.\n"
            f"5. Clearly define tradeoffs, resource requirements, and critical assumptions.\n"
            f"6. Do NOT select the final recommendation; only generate evaluated candidate options."
        )

        try:
            response = await self.llm_client.generate_structured(
                prompt=prompt,
                response_schema=LLMOptionGenerationResponse,
                system_prompt="You are a decision options analyst. Output strictly valid JSON adhering to the schema.",
                budget_tracker=self.budget_tracker,
            )

            options: list[DecisionOption] = []
            rejection_reasons: list[str] = []
            candidate_count = len(response.options)

            # Transitive fallback evidence IDs
            all_evidence_ids: list[str] = []
            for ev in events:
                all_evidence_ids.extend([ref.evidence_id for ref in ev.evidence])

            for cand in response.options:
                if not cand.title or not cand.title.strip():
                    rejection_reasons.append("Candidate option rejected: empty title.")
                    continue

                # Grounding validation
                cited_insight_ids = [iid for iid in cand.supporting_insight_ids if iid in valid_insight_ids]
                if not cited_insight_ids and valid_insight_ids:
                    # Link to available insights if none cited
                    cited_insight_ids = list(valid_insight_ids)

                cited_event_ids = [eid for eid in cand.supporting_event_ids if eid in valid_event_map]
                if not cited_event_ids and valid_event_map:
                    cited_event_ids = list(valid_event_map.keys())

                # Collect transitive evidence IDs
                evidence_ids: list[str] = []
                for eid in cited_event_ids:
                    if eid in valid_event_map:
                        evidence_ids.extend([ref.evidence_id for ref in valid_event_map[eid].evidence])
                if not evidence_ids and all_evidence_ids:
                    evidence_ids = list(all_evidence_ids)

                # Convert predicted consequences
                consequences = [
                    PredictedConsequence(
                        dimension=eff.dimension,
                        description=eff.description,
                        is_positive=eff.is_positive,
                        confidence_score=eff.confidence_score,
                        underlying_assumptions=eff.underlying_assumptions,
                    )
                    for eff in cand.expected_effects
                ]

                # Convert risks
                risks = [
                    DecisionRisk(
                        description=r.description,
                        likelihood=r.likelihood,
                        severity=r.severity,
                        basis_type=r.basis_type,
                        mitigation_strategy=r.mitigation_strategy,
                    )
                    for r in cand.risks
                ]

                opt = DecisionOption(
                    option_id=str(uuid4()),
                    decision_id=context.decision_id,
                    title=cand.title.strip(),
                    description=cand.description.strip(),
                    action_type=cand.action_type,
                    expected_effects=consequences,
                    required_resources=cand.required_resources,
                    estimated_cost_usd=cand.estimated_cost_usd,
                    risks=risks,
                    tradeoffs=cand.tradeoffs,
                    assumptions=cand.assumptions,
                    supporting_insight_ids=cited_insight_ids,
                    supporting_event_ids=cited_event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
                options.append(opt)

            if not options and candidate_count > 0:
                logger.warning("All LLM candidate options rejected, falling back to rule-based generator")
                return await self.fallback_generator.generate_options(context, insights, events)

            self.last_diagnostics = {
                "candidate_options_count": candidate_count,
                "accepted_options_count": len(options),
                "rejected_options_count": candidate_count - len(options),
                "rejection_reasons": rejection_reasons,
                "fallback_used": False,
                "fallback_reason": None,
            }
            return options

        except Exception as exc:
            logger.warning(
                "LLM option generation failed, executing rule-based fallback",
                error=str(exc),
                decision_id=context.decision_id,
            )
            fallback_options = await self.fallback_generator.generate_options(context, insights, events)
            self.last_diagnostics = {
                "candidate_options_count": len(fallback_options),
                "accepted_options_count": len(fallback_options),
                "rejected_options_count": 0,
                "rejection_reasons": [f"LLM option generation failed ({type(exc).__name__}: {str(exc)}); used fallback."],
                "fallback_used": True,
                "fallback_reason": f"{type(exc).__name__}: {str(exc)}",
            }
            return fallback_options
