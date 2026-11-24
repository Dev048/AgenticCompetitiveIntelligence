"""Decision Analyzer and reference RuleBasedOptionGenerator components."""

from datetime import datetime, timezone
from typing import Any

from competitive_intelligence.decision_support.interfaces import (
    BaseDecisionAnalyzer,
    BaseOptionGenerator,
    BaseRecommendationValidator,
)
from competitive_intelligence.decision_support.models import (
    ActionType,
    ConstraintType,
    DecisionContext,
    DecisionDomain,
    DecisionOption,
    DecisionPolicy,
    DecisionRisk,
    DecisionUrgency,
    EvaluatedAlternative,
    ObjectiveDimension,
    PredictedConsequence,
    Recommendation,
    RecommendationConfidenceAssessment,
    RecommendationStatus,
    RiskBasisType,
    RiskTolerance,
)
from competitive_intelligence.decision_support.validator import (
    DeterministicRecommendationValidator,
)
from competitive_intelligence.intelligence.models import (
    ConfidenceLevel,
    Insight,
    InsightType,
    IntelligenceEvent,
    SeverityLevel,
)


class RuleBasedOptionGenerator(BaseOptionGenerator):
    """Reference deterministic option generator implementing BaseOptionGenerator.

    Produces bounded, structured candidate decision options based on explicit DecisionContext.
    """

    async def generate_options(
        self,
        context: DecisionContext,
        insights: list[Insight],
        events: list[IntelligenceEvent],
    ) -> list[DecisionOption]:
        if not context.business_objectives:
            return []

        options: list[DecisionOption] = []
        target_domain = context.decision_domain

        # Transitive provenance IDs from context
        insight_ids = [ins.insight_id for ins in insights]
        event_ids = [e.event_id for e in events]
        for ins in insights:
            event_ids.extend(ins.supporting_event_ids)
        event_ids = list(dict.fromkeys(event_ids))

        evidence_ids: list[str] = []
        for e in events:
            evidence_ids.extend([ref.evidence_id for ref in e.evidence])
        for ins in insights:
            evidence_ids.extend(ins.supporting_evidence_ids)
        evidence_ids = list(dict.fromkeys(evidence_ids))

        if target_domain == DecisionDomain.PRICING:
            # Option 1: Maintain & Monitor (Low Cost / Conservative)
            options.append(
                DecisionOption(
                    decision_id=context.decision_id,
                    title="Maintain Current Pricing & Monitor Conversion",
                    description="Keep existing pricing structure intact while actively tracking funnel conversion changes.",
                    action_type=ActionType.MONITOR_AND_WAIT,
                    estimated_cost_usd=0.0,
                    expected_effects=[
                        PredictedConsequence(
                            dimension=ObjectiveDimension.COST_EFFICIENCY,
                            description="Preserves gross margins without incurring implementation or discounting costs.",
                            is_positive=True,
                            underlying_assumptions=["Assumes existing customers have moderate switching costs."],
                        ),
                        PredictedConsequence(
                            dimension=ObjectiveDimension.GROWTH,
                            description="May experience temporary drag on new user acquisition in cost-sensitive segments.",
                            is_positive=False,
                            underlying_assumptions=["Assumes price-sensitive prospects evaluate alternative tiers."],
                        ),
                    ],
                    required_resources=["Telemetry dashboard monitoring"],
                    risks=[
                        DecisionRisk(
                            description="Market share erosion in entry tier.",
                            likelihood=ConfidenceLevel.MEDIUM,
                            severity=SeverityLevel.MEDIUM,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                            mitigation_strategy="Set weekly conversion alert thresholds.",
                        )
                    ],
                    tradeoffs=["Preserves revenue margins but risks slower top-of-funnel acquisition."],
                    assumptions=["Competitor price reduction does not immediately trigger mass migration."],
                    supporting_insight_ids=insight_ids,
                    supporting_event_ids=event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )

            # Option 2: Targeted Counter-Adjustment (Higher Cost / Aggressive)
            options.append(
                DecisionOption(
                    decision_id=context.decision_id,
                    title="Targeted Entry-Tier Price Match",
                    description="Introduce a matched entry-level pricing tier with bounded usage limits.",
                    action_type=ActionType.DIRECT_COUNTER_MOVE,
                    estimated_cost_usd=20000.0,
                    expected_effects=[
                        PredictedConsequence(
                            dimension=ObjectiveDimension.GROWTH,
                            description="Neutralizes competitor pricing advantage and restores acquisition parity.",
                            is_positive=True,
                            underlying_assumptions=["Assumes price parity directly restores prospect consideration."],
                        ),
                        PredictedConsequence(
                            dimension=ObjectiveDimension.REVENUE,
                            description="Reduces average revenue per user (ARPU) across entry-tier accounts.",
                            is_positive=False,
                            underlying_assumptions=["Assumes some existing users downgrade to entry tier."],
                        ),
                    ],
                    required_resources=["Billing engineering team", "Updated pricing pages"],
                    risks=[
                        DecisionRisk(
                            description="Down-market cannibalization of standard tier.",
                            likelihood=ConfidenceLevel.HIGH,
                            severity=SeverityLevel.HIGH,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                            mitigation_strategy="Gate advanced capabilities behind pro tier.",
                        )
                    ],
                    tradeoffs=["Accelerates market share defense at the expense of lower unit margins."],
                    assumptions=["Acquisition volume compensates for lower entry ARPU."],
                    supporting_insight_ids=insight_ids,
                    supporting_event_ids=event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )

            # Option 3: Value Differentiation (Moderate Cost / Value Focus)
            options.append(
                DecisionOption(
                    decision_id=context.decision_id,
                    title="Value-Add Packaging Differentiation",
                    description="Retain pricing but bundle premium support and high-performance throughput add-ons.",
                    action_type=ActionType.DIFFERENTIATION,
                    estimated_cost_usd=5000.0,
                    expected_effects=[
                        PredictedConsequence(
                            dimension=ObjectiveDimension.STRATEGIC_POSITIONING,
                            description="Strengthens premium value proposition without initiating a price war.",
                            is_positive=True,
                            underlying_assumptions=["Enterprise buyers prioritize support quality and performance over minor price deltas."],
                        )
                    ],
                    required_resources=["Product marketing documentation", "Support packaging update"],
                    risks=[
                        DecisionRisk(
                            description="Prospects may overlook bundled value in favor of headline price.",
                            likelihood=ConfidenceLevel.LOW,
                            severity=SeverityLevel.MEDIUM,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                            mitigation_strategy="Highlight SLA metrics in comparison collaterals.",
                        )
                    ],
                    tradeoffs=["Protects brand positioning but requires active marketing communication."],
                    assumptions=["Target audience values reliability over lowest nominal cost."],
                    supporting_insight_ids=insight_ids,
                    supporting_event_ids=event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )

        elif target_domain == DecisionDomain.PRODUCT:
            # Option 1: Accelerated Complementary Feature (High Cost)
            options.append(
                DecisionOption(
                    decision_id=context.decision_id,
                    title="Accelerate Complementary Feature Release",
                    description="Fast-track developer tooling and API integrations to counter competitor capability expansion.",
                    action_type=ActionType.DIRECT_COUNTER_MOVE,
                    estimated_cost_usd=30000.0,
                    expected_effects=[
                        PredictedConsequence(
                            dimension=ObjectiveDimension.TIME_TO_MARKET,
                            description="Closes competitive capability gap within 30 days.",
                            is_positive=True,
                            underlying_assumptions=["Sprint reallocation can deliver feature without quality degradation."],
                        )
                    ],
                    required_resources=["2 dedicated engineering pods for 1 month"],
                    risks=[
                        DecisionRisk(
                            description="Delay to existing quarterly roadmap commitments.",
                            likelihood=ConfidenceLevel.HIGH,
                            severity=SeverityLevel.MEDIUM,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                            mitigation_strategy="Backfill secondary tasks with contractor bandwidth.",
                        )
                    ],
                    tradeoffs=["Closes capability gap rapidly at the cost of pausing secondary features."],
                    assumptions=["Competitor release is actively influencing buyer evaluations."],
                    supporting_insight_ids=insight_ids,
                    supporting_event_ids=event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )

            # Option 2: Roadmap Focus & Differentiation (Low Cost)
            options.append(
                DecisionOption(
                    decision_id=context.decision_id,
                    title="Maintain Core Platform Roadmap Focus",
                    description="Continue execution on core architecture differentiators rather than reacting to tactical features.",
                    action_type=ActionType.DIFFERENTIATION,
                    estimated_cost_usd=4000.0,
                    expected_effects=[
                        PredictedConsequence(
                            dimension=ObjectiveDimension.STRATEGIC_POSITIONING,
                            description="Deepens long-term platform differentiation and reliability.",
                            is_positive=True,
                            underlying_assumptions=["Core platform stability is our primary competitive moat."],
                        )
                    ],
                    required_resources=["Existing engineering allocation"],
                    risks=[
                        DecisionRisk(
                            description="Short-term loss of competitive feature checklist comparisons.",
                            likelihood=ConfidenceLevel.MEDIUM,
                            severity=SeverityLevel.LOW,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                            mitigation_strategy="Publish benchmark whitepaper explaining architecture advantage.",
                        )
                    ],
                    tradeoffs=["Maintains engineering focus but concedes short-term tactical feature parity."],
                    assumptions=["Competitor feature is peripheral to core enterprise workload."],
                    supporting_insight_ids=insight_ids,
                    supporting_event_ids=event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )

        elif target_domain == DecisionDomain.HIRING_TALENT:
            # Option 1: External Leadership Sourcing (High Cost)
            options.append(
                DecisionOption(
                    decision_id=context.decision_id,
                    title="External Strategic Talent Acquisition",
                    description="Retain executive search partner to recruit senior leadership in specialized domain.",
                    action_type=ActionType.RESOURCE_REALLOCATION,
                    estimated_cost_usd=50000.0,
                    expected_effects=[
                        PredictedConsequence(
                            dimension=ObjectiveDimension.GROWTH,
                            description="Accelerates strategic capability building through external domain expertise.",
                            is_positive=True,
                            underlying_assumptions=["External hire brings proven execution playbook."],
                        )
                    ],
                    required_resources=["Executive search retainer", "Executive onboarding budget"],
                    risks=[
                        DecisionRisk(
                            description="High compensation overhead and hiring timeline risk.",
                            likelihood=ConfidenceLevel.MEDIUM,
                            severity=SeverityLevel.HIGH,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                            mitigation_strategy="Structure performance-based milestone vesting.",
                        )
                    ],
                    tradeoffs=["Brings rapid domain maturity at significant upfront recruiting cost."],
                    assumptions=["Internal promotion candidates lack specialized domain background."],
                    supporting_insight_ids=insight_ids,
                    supporting_event_ids=event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )

            # Option 2: Internal Upskilling & Lateral Reallocation (Low Cost)
            options.append(
                DecisionOption(
                    decision_id=context.decision_id,
                    title="Internal Capability Development & Upskilling",
                    description="Promote high-performing internal leads and provide specialized training resources.",
                    action_type=ActionType.DIFFERENTIATION,
                    estimated_cost_usd=8000.0,
                    expected_effects=[
                        PredictedConsequence(
                            dimension=ObjectiveDimension.COST_EFFICIENCY,
                            description="Builds organizational domain capacity while maintaining low cost and cultural alignment.",
                            is_positive=True,
                            underlying_assumptions=["Internal leads have strong technical foundation."],
                        )
                    ],
                    required_resources=["Training workshops", "Mentorship bandwidth"],
                    risks=[
                        DecisionRisk(
                            description="Longer ramp-up time to reach peak domain execution.",
                            likelihood=ConfidenceLevel.MEDIUM,
                            severity=SeverityLevel.MEDIUM,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                            mitigation_strategy="Pair internal team with specialized advisory consultant.",
                        )
                    ],
                    tradeoffs=["Low financial risk and high cultural cohesion at the expense of ramp-up speed."],
                    assumptions=["Current organizational roadmap allows 3-month ramp-up window."],
                    supporting_insight_ids=insight_ids,
                    supporting_event_ids=event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )

        else:
            # General Strategy Default Options
            options.append(
                DecisionOption(
                    decision_id=context.decision_id,
                    title="Monitor Monitored Competitors & Defer Action",
                    description="Track competitor developments closely before committing capital or operational changes.",
                    action_type=ActionType.MONITOR_AND_WAIT,
                    estimated_cost_usd=0.0,
                    expected_effects=[
                        PredictedConsequence(
                            dimension=ObjectiveDimension.RISK_REDUCTION,
                            description="Avoids premature capital expenditure under incomplete market information.",
                            is_positive=True,
                            underlying_assumptions=["Market signal is early-stage and may shift."],
                        )
                    ],
                    required_resources=["Standard CI monitoring setup"],
                    risks=[],
                    tradeoffs=["Conserves resources but foregoes early first-mover positioning."],
                    assumptions=["Competitor activity does not present immediate operational threat."],
                    supporting_insight_ids=insight_ids,
                    supporting_event_ids=event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )
            options.append(
                DecisionOption(
                    decision_id=context.decision_id,
                    title="Proactive Strategic Initiative",
                    description="Launch targeted pilot initiative to validate market opportunity.",
                    action_type=ActionType.EXPERIMENTATION_TEST,
                    estimated_cost_usd=10000.0,
                    expected_effects=[
                        PredictedConsequence(
                            dimension=ObjectiveDimension.STRATEGIC_POSITIONING,
                            description="Tests customer response in controlled pilot before broad commitment.",
                            is_positive=True,
                            underlying_assumptions=["Pilot feedback reflects broader market demand."],
                        )
                    ],
                    required_resources=["Pilot program coordinator", "User research bandwidth"],
                    risks=[
                        DecisionRisk(
                            description="Pilot outcome may be inconclusive.",
                            likelihood=ConfidenceLevel.LOW,
                            severity=SeverityLevel.LOW,
                            basis_type=RiskBasisType.MODEL_PREDICTED,
                        )
                    ],
                    tradeoffs=["Provides empirical validation at modest investment cost."],
                    assumptions=["Pilot scope is sufficient to gather statistically meaningful signal."],
                    supporting_insight_ids=insight_ids,
                    supporting_event_ids=event_ids,
                    supporting_evidence_ids=evidence_ids,
                )
            )

        return options


class DeterministicDecisionAnalyzer(BaseDecisionAnalyzer):
    """Production component synthesizing candidate options and justified recommendations."""

    def __init__(
        self,
        option_generator: BaseOptionGenerator | None = None,
        recommendation_validator: BaseRecommendationValidator | None = None,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self.option_generator = option_generator or RuleBasedOptionGenerator()
        self.recommendation_validator = (
            recommendation_validator or DeterministicRecommendationValidator()
        )
        self.policy = policy or DecisionPolicy()

    async def analyze_and_recommend(
        self,
        context: DecisionContext,
        insights: list[Insight],
        events: list[IntelligenceEvent],
        existing_recommendations: list[Recommendation] | None = None,
    ) -> tuple[list[DecisionOption], Recommendation | None]:
        """Evaluate context and insights to generate candidate options and an optimal recommendation."""
        # Stage 1: Context Gating
        if not context.business_objectives:
            return [], None

        # Stage 2: Candidate Option Generation
        candidate_options = await self.option_generator.generate_options(
            context=context,
            insights=insights,
            events=events,
        )
        if not candidate_options:
            return [], None

        # Stage 3: Hard Constraint Filtering
        evaluated_alts: list[EvaluatedAlternative] = []
        feasible_options: list[DecisionOption] = []

        for opt in candidate_options:
            is_eliminated = False
            elimination_reason: str | None = None

            if self.policy.enforce_hard_constraints and context.constraints:
                for c in context.constraints:
                    if not c.is_hard_constraint:
                        continue

                    # A. Budget Hard Constraint
                    if c.constraint_type == ConstraintType.BUDGET and c.threshold_value is not None:
                        max_budget = float(c.threshold_value)
                        if opt.estimated_cost_usd is not None and opt.estimated_cost_usd > max_budget:
                            is_eliminated = True
                            elimination_reason = f"Estimated cost (${opt.estimated_cost_usd:,.0f}) exceeds hard budget limit (${max_budget:,.0f})."
                            break

                    # B. Risk Tolerance Hard Constraint
                    if c.constraint_type == ConstraintType.MAX_ACCEPTABLE_RISK:
                        if context.risk_tolerance == RiskTolerance.VERY_LOW_CONSERVATIVE:
                            has_high_risk = any(
                                r.severity == SeverityLevel.HIGH for r in opt.risks
                            )
                            if has_high_risk:
                                is_eliminated = True
                                elimination_reason = "Option involves HIGH severity risk, violating VERY_LOW_CONSERVATIVE risk constraint."
                                break

            if is_eliminated:
                evaluated_alts.append(
                    EvaluatedAlternative(
                        option_id=opt.option_id,
                        title=opt.title,
                        is_selected=False,
                        is_eliminated=True,
                        elimination_reason=elimination_reason,
                    )
                )
            else:
                feasible_options.append(opt)
                evaluated_alts.append(
                    EvaluatedAlternative(
                        option_id=opt.option_id,
                        title=opt.title,
                        is_selected=False,
                        is_eliminated=False,
                    )
                )

        # If zero options remain feasible -> return all options and None
        if not feasible_options:
            return candidate_options, None

        # Stage 4 & 5: Selection Analysis
        selected_option: DecisionOption | None = None
        selection_rationale: str = ""

        # Case A: Exactly 1 option remains feasible after hard constraints
        if len(feasible_options) == 1:
            selected_option = feasible_options[0]
            eliminated_count = len(candidate_options) - 1
            selection_rationale = (
                f"Selected '{selected_option.title}' as the only feasible candidate action "
                f"after {eliminated_count} alternative(s) were eliminated by hard constraints."
            )

        # Case B: Multiple feasible options exist -> Check for explicit objective weights
        else:
            has_explicit_weights = any(
                obj.priority_weight is not None for obj in context.business_objectives
            )

            if has_explicit_weights:
                # Rank by explicit weights
                best_score = -1.0
                best_opt: DecisionOption | None = None

                for opt in feasible_options:
                    score = 0.0
                    for obj in context.business_objectives:
                        weight = obj.priority_weight if obj.priority_weight is not None else 0.0
                        # Check if any positive predicted consequence aligns with objective dimension
                        aligns = any(
                            eff.dimension == obj.dimension and eff.is_positive
                            for eff in opt.expected_effects
                        )
                        if aligns:
                            score += weight

                    if score > best_score:
                        best_score = score
                        best_opt = opt

                if best_opt and best_score > 0.0:
                    selected_option = best_opt
                    selection_rationale = (
                        f"Selected '{selected_option.title}' based on highest alignment score ({best_score:.2f}) "
                        f"with user-supplied objective priority weights."
                    )
            else:
                # No explicit weights and multiple options -> Return options + None (no hidden optimizer!)
                return candidate_options, None

        if not selected_option:
            return candidate_options, None

        # Stage 6: Build Recommendation & Alternative Audit Trail
        # Mark selected alternative and populate trade-offs
        for alt in evaluated_alts:
            if alt.option_id == selected_option.option_id:
                alt.is_selected = True
            else:
                alt.is_selected = False
                if not alt.is_eliminated:
                    alt.tradeoff_vs_selected = (
                        f"Alternative '{alt.title}' offers different strategic trade-offs but was not prioritized under supplied objectives."
                    )

        # Collect transitive provenance
        supporting_insights = [ins.insight_id for ins in insights]
        supporting_events = [e.event_id for e in events]
        for ins in insights:
            supporting_events.extend(ins.supporting_event_ids)
        supporting_events.extend(selected_option.supporting_event_ids)
        supporting_events = list(dict.fromkeys(supporting_events))

        supporting_evidence: list[str] = list(selected_option.supporting_evidence_ids)
        for ins in insights:
            supporting_evidence.extend(ins.supporting_evidence_ids)
        supporting_evidence = list(dict.fromkeys(supporting_evidence))

        # Expected benefits from selected option
        expected_benefits = [
            eff.description for eff in selected_option.expected_effects if eff.is_positive
        ]
        if not expected_benefits:
            expected_benefits = [f"Executes preferred course of action: {selected_option.title}"]

        # Urgency derivation from context
        urgency = self._derive_urgency(context, insights)

        # Propagate strategic uncertainties from conflicting/uncertain insights
        unresolved_uncertainties = [
            f"Contradictory/uncertain evidence in insight '{ins.title}': {ins.summary}"
            for ins in insights
            if ins.insight_type == InsightType.UNCERTAINTY
        ]

        has_uncertainties = len(unresolved_uncertainties) > 0
        conf_score = 0.65 if has_uncertainties else 0.85
        conf_level = ConfidenceLevel.MEDIUM if has_uncertainties else ConfidenceLevel.HIGH
        conf_rationale = (
            "Recommendation evaluated under observed strategic uncertainty and conflicting market signals."
            if has_uncertainties
            else "Recommendation justified by explicit constraints and objective alignment."
        )

        recommendation = Recommendation(
            decision_id=context.decision_id,
            workspace_id=context.workspace_id,
            selected_option_id=selected_option.option_id,
            urgency=urgency,
            status=RecommendationStatus.PROPOSED,
            title=f"Recommendation: {selected_option.title}",
            summary=f"Proposed action for {context.title}: {selected_option.description}",
            rationale=selection_rationale,
            evaluated_alternatives=evaluated_alts,
            expected_benefits=expected_benefits,
            tradeoffs=selected_option.tradeoffs,
            critical_assumptions=selected_option.assumptions,
            unresolved_uncertainties=unresolved_uncertainties,
            confidence=RecommendationConfidenceAssessment(
                score=conf_score,
                level=conf_level,
                rationale=conf_rationale,
                option_analysis_confidence=0.80,
            ),
            supporting_insight_ids=supporting_insights,
            supporting_event_ids=supporting_events,
            supporting_evidence_ids=supporting_evidence,
            generated_at=datetime.now(timezone.utc),
            metadata=context.metadata,
        )

        # Stage 7: Validate Recommendation
        is_valid, validation_errors = self.recommendation_validator.validate(
            payload=recommendation,
            context=context,
            options=candidate_options,
            insights=insights,
            events=events,
        )
        if not is_valid:
            return candidate_options, None

        return candidate_options, recommendation

    def _derive_urgency(self, context: DecisionContext, insights: list[Insight]) -> DecisionUrgency:
        """Derive urgency from explicit context deadlines and insight characteristics."""
        if context.deadline is not None:
            now = datetime.now(timezone.utc)
            delta_days = (context.deadline - now).total_seconds() / 86400.0
            if delta_days <= 7:
                return DecisionUrgency.URGENT
            if delta_days <= 30:
                return DecisionUrgency.RECOMMENDED

        if any(ins.insight_type == InsightType.RISK_SIGNAL for ins in insights):
            return DecisionUrgency.RECOMMENDED

        return self.policy.default_urgency
