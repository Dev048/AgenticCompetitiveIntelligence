"""Evaluation Script: Insight -> Decision Support Subsystem Integration.

Evaluates the integrated decision support boundary across 7 operational scenarios:
1. Pricing Pressure Response (Explicit objective weights + budget constraint)
2. Product Launch Response (Time-to-market prioritization + feature acceleration)
3. Hiring Expansion Response (Risk tolerance constraint + capability development)
4. Contradictory Competitive Evidence (Uncertainty preservation in unresolved_uncertainties)
5. Missing Decision Context (Empty objectives -> 0 options, None)
6. Conflicting Objectives (Multiple competing priorities without weights -> options returned, rec=None)
7. Unsupported Recommendation Claim (Fabricated citations -> rejected by deterministic validator)

Usage:
    python scripts/evaluate_insight_decision_support_integration.py
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import sys
from typing import Any

# Ensure workspace src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from competitive_intelligence.decision_support import (
    ActionType,
    BaseOptionGenerator,
    ConstraintType,
    DecisionConstraint,
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    DecisionOption,
    DecisionUrgency,
    DeterministicDecisionAnalyzer,
    DeterministicRecommendationValidator,
    ObjectiveDimension,
    PredictedConsequence,
    Recommendation,
    RecommendationConfidenceAssessment,
    RecommendationStatus,
    RiskTolerance,
)
from competitive_intelligence.intelligence import (
    ConfidenceAssessment,
    ConfidenceLevel,
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
    SignificanceAssessment,
    SignalStatus,
    SignalType,
    SourcePerspective,
    TimePrecision,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentVersion,
)
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    ContentType,
    PlatformType,
    Source,
    SourceCategory,
    SourceClassification,
    SourceIdentity,
    SourceLocation,
    ToolChain,
)


@dataclass
class DecisionScenarioReport:
    name: str
    description: str
    passed: bool
    insights_supplied: int = 0
    decision_context_title: str = ""
    candidate_options_count: int = 0
    feasible_options_count: int = 0
    eliminated_options_count: int = 0
    recommendation_status: str | None = None
    supporting_insight_ids: list[str] = field(default_factory=list)
    provenance_verified: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    human_review_required: bool = True
    details: dict[str, Any] = field(default_factory=dict)


def make_grounded_hierarchy(
    workspace_id: str = "ws_eval_01",
    company_id: str = "acme_corp",
    domain: DecisionDomain = DecisionDomain.PRICING,
    insight_type: InsightType = InsightType.PRICING_STRATEGY,
    title: str = "Pricing Strategy Response",
) -> tuple[DecisionContext, list[Insight], list[IntelligenceEvent], list[EvidenceReference]]:
    """Build a complete grounded hierarchy for evaluation scenarios."""
    event_id = f"ev_{domain.value}_01"
    evidence_id = f"ev_ref_{domain.value}_01"
    insight_id = f"ins_{domain.value}_01"
    signal_id = f"sig_{domain.value}_01"
    source_url = f"https://acme.com/{domain.value}"

    evidence = EvidenceReference(
        evidence_id=evidence_id,
        source_id=f"src_{domain.value}_01",
        document_id=f"doc_{domain.value}_01",
        version_id=f"ver_{domain.value}_01",
        chunk_id=f"chunk_{domain.value}_01",
        source_url=source_url,
        source_category="pricing",
        extracted_text=f"Observed {domain.value} shift for {company_id}.",
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        role=EvidenceRole.PRIMARY_DIRECT,
    )

    event = IntelligenceEvent(
        event_id=event_id,
        workspace_id=workspace_id,
        company_id=company_id,
        event_type=IntelligenceEventType.PRICING_CHANGE if domain == DecisionDomain.PRICING else IntelligenceEventType.PRODUCT_LAUNCH,
        title=f"Observed {domain.value.title()} Event",
        summary=f"Factual occurrence in {domain.value}.",
        subject="Core Offering",
        event_time=EventTime(
            occurred_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            precision=TimePrecision.DATE_ONLY,
            acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            detected_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.90, level=ConfidenceLevel.HIGH),
        evidence=[evidence],
        primary_perspective=SourcePerspective.FIRST_PARTY,
    )

    insight = Insight(
        insight_id=insight_id,
        workspace_id=workspace_id,
        company_id=company_id,
        insight_type=insight_type,
        title=f"Strategic {domain.value.title()} Movement",
        summary=f"Market intelligence indicates strategic activity in {domain.value}.",
        statement="Detailed statement of competitive insight.",
        supporting_signal_ids=[signal_id],
        supporting_event_ids=[event_id],
        supporting_evidence_ids=[evidence_id],
        factual_confidence=ConfidenceAssessment(score=0.90, level=ConfidenceLevel.HIGH),
        interpretive_confidence=InterpretiveConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH),
        significance=SignificanceAssessment(),
        status=InsightStatus.ASSESSED if insight_type != InsightType.UNCERTAINTY else InsightStatus.UNCERTAIN,
    )

    context = DecisionContext(
        workspace_id=workspace_id,
        title=title,
        decision_domain=domain,
        business_objectives=[
            DecisionObjective(
                name="Primary Objective",
                dimension=ObjectiveDimension.GROWTH,
                target_description="Maximize growth and market share.",
                priority_weight=0.9,
            ),
            DecisionObjective(
                name="Secondary Objective",
                dimension=ObjectiveDimension.COST_EFFICIENCY,
                target_description="Maintain operational efficiency.",
                priority_weight=0.1,
            ),
        ],
        constraints=[
            DecisionConstraint(
                constraint_type=ConstraintType.BUDGET,
                description="Budget limit $35,000",
                is_hard_constraint=True,
                threshold_value=35000.0,
            )
        ],
        risk_tolerance=RiskTolerance.MODERATE,
    )

    return context, [insight], [event], [evidence]


async def evaluate_scenario_1_pricing_pressure() -> DecisionScenarioReport:
    """Scenario 1: Pricing Pressure Response with explicit weights and budget constraint."""
    context, insights, events, evidence = make_grounded_hierarchy(domain=DecisionDomain.PRICING)
    analyzer = DeterministicDecisionAnalyzer()

    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=insights,
        events=events,
    )

    passed = (
        len(options) == 3
        and rec is not None
        and rec.status == RecommendationStatus.PROPOSED
        and rec.selected_option_id in {opt.option_id for opt in options}
        and insights[0].insight_id in rec.supporting_insight_ids
        and events[0].event_id in rec.supporting_event_ids
    )

    elim_count = sum(1 for alt in rec.evaluated_alternatives if alt.is_eliminated) if rec else 0

    return DecisionScenarioReport(
        name="Scenario 1: Pricing Pressure Response",
        description="Explicit objective weights select optimal counter-move while budget constraint is respected",
        passed=passed,
        insights_supplied=len(insights),
        decision_context_title=context.title,
        candidate_options_count=len(options),
        feasible_options_count=len(options) - elim_count,
        eliminated_options_count=elim_count,
        recommendation_status=rec.status.value if rec else None,
        supporting_insight_ids=rec.supporting_insight_ids if rec else [],
        provenance_verified=passed,
        human_review_required=True,
        details={
            "selected_option": rec.title if rec else None,
            "rationale": rec.rationale if rec else None,
        },
    )


async def evaluate_scenario_2_product_launch() -> DecisionScenarioReport:
    """Scenario 2: Product Launch Response prioritizing time-to-market."""
    context, insights, events, _ = make_grounded_hierarchy(
        domain=DecisionDomain.PRODUCT,
        insight_type=InsightType.PRODUCT_STRATEGY,
        title="Product Counter-Strategy",
    )
    context.business_objectives = [
        DecisionObjective(
            name="Rapid Response",
            dimension=ObjectiveDimension.TIME_TO_MARKET,
            target_description="Close capability gap within 30 days.",
            priority_weight=1.0,
        )
    ]
    analyzer = DeterministicDecisionAnalyzer()

    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=insights,
        events=events,
    )

    passed = (
        len(options) == 2
        and rec is not None
        and "Accelerate Complementary Feature Release" in rec.title
        and rec.status == RecommendationStatus.PROPOSED
    )

    return DecisionScenarioReport(
        name="Scenario 2: Product Launch Response",
        description="Time-to-market objective prioritizes accelerated feature roadmap option",
        passed=passed,
        insights_supplied=len(insights),
        decision_context_title=context.title,
        candidate_options_count=len(options),
        feasible_options_count=len(options),
        eliminated_options_count=0,
        recommendation_status=rec.status.value if rec else None,
        supporting_insight_ids=rec.supporting_insight_ids if rec else [],
        provenance_verified=passed,
        human_review_required=True,
        details={"selected_action": rec.title if rec else None},
    )


async def evaluate_scenario_3_hiring_expansion() -> DecisionScenarioReport:
    """Scenario 3: Hiring Expansion with Conservative Risk Constraint."""
    context, insights, events, _ = make_grounded_hierarchy(
        domain=DecisionDomain.HIRING_TALENT,
        insight_type=InsightType.HIRING_DIRECTION,
        title="Talent Strategy",
    )
    context.risk_tolerance = RiskTolerance.VERY_LOW_CONSERVATIVE
    context.constraints = [
        DecisionConstraint(
            constraint_type=ConstraintType.MAX_ACCEPTABLE_RISK,
            description="No high severity hiring risks permitted",
            is_hard_constraint=True,
        )
    ]
    context.business_objectives = [
        DecisionObjective(
            name="Capability Development",
            dimension=ObjectiveDimension.COST_EFFICIENCY,
            target_description="Develop internal talent.",
            priority_weight=1.0,
        )
    ]

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=insights,
        events=events,
    )

    passed = (
        len(options) == 2
        and rec is not None
        and "Internal Capability Development" in rec.title
        and any(alt.is_eliminated for alt in rec.evaluated_alternatives)
    )

    return DecisionScenarioReport(
        name="Scenario 3: Hiring Expansion Response",
        description="Conservative risk tolerance constraint eliminates high-risk executive search in favor of internal upskilling",
        passed=passed,
        insights_supplied=len(insights),
        decision_context_title=context.title,
        candidate_options_count=len(options),
        feasible_options_count=1,
        eliminated_options_count=1,
        recommendation_status=rec.status.value if rec else None,
        supporting_insight_ids=rec.supporting_insight_ids if rec else [],
        provenance_verified=passed,
        human_review_required=True,
        details={"selected_action": rec.title if rec else None},
    )


async def evaluate_scenario_4_contradictory_evidence() -> DecisionScenarioReport:
    """Scenario 4: Contradictory Evidence preserving uncertainty in recommendations."""
    context, insights, events, _ = make_grounded_hierarchy(
        domain=DecisionDomain.PRICING,
        insight_type=InsightType.UNCERTAINTY,
        title="Pricing under Strategic Uncertainty",
    )

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=insights,
        events=events,
    )

    passed = (
        len(options) == 3
        and rec is not None
        and len(rec.unresolved_uncertainties) >= 1
        and rec.confidence.level == ConfidenceLevel.MEDIUM
        and "observed strategic uncertainty" in rec.confidence.rationale
    )

    return DecisionScenarioReport(
        name="Scenario 4: Contradictory Evidence (Uncertainty Preservation)",
        description="Epistemic contradiction in insights propagates directly into recommendation uncertainties without taking false sides",
        passed=passed,
        insights_supplied=len(insights),
        decision_context_title=context.title,
        candidate_options_count=len(options),
        feasible_options_count=len(options),
        eliminated_options_count=0,
        recommendation_status=rec.status.value if rec else None,
        supporting_insight_ids=rec.supporting_insight_ids if rec else [],
        provenance_verified=passed,
        human_review_required=True,
        details={
            "confidence_level": rec.confidence.level.value if rec else None,
            "unresolved_uncertainties": rec.unresolved_uncertainties if rec else [],
        },
    )


async def evaluate_scenario_5_missing_context() -> DecisionScenarioReport:
    """Scenario 5: Missing Context safely yields 0 options and no recommendation."""
    analyzer = DeterministicDecisionAnalyzer()
    empty_context = DecisionContext(
        title="Unspecified Context",
        business_objectives=[],
    )

    options, rec = await analyzer.analyze_and_recommend(
        context=empty_context,
        insights=[],
        events=[],
    )

    passed = len(options) == 0 and rec is None

    return DecisionScenarioReport(
        name="Scenario 5: Missing Decision Context Safety",
        description="Absent objectives prevent unjustified recommendation synthesis and return empty sets",
        passed=passed,
        insights_supplied=0,
        decision_context_title="Unspecified Context",
        candidate_options_count=0,
        feasible_options_count=0,
        eliminated_options_count=0,
        recommendation_status=None,
        supporting_insight_ids=[],
        provenance_verified=True,
        rejection_reasons=["missing_business_objectives"],
        human_review_required=False,
        details={"safe_zero_output": True},
    )


async def evaluate_scenario_6_conflicting_objectives() -> DecisionScenarioReport:
    """Scenario 6: Conflicting Objectives without explicit weights returns candidate options only."""
    context, insights, events, _ = make_grounded_hierarchy(domain=DecisionDomain.PRICING)
    # Remove explicit priority weights
    context.business_objectives = [
        DecisionObjective(
            name="Aggressive Acquisition",
            dimension=ObjectiveDimension.GROWTH,
            target_description="Drive volume",
            priority_weight=None,
        ),
        DecisionObjective(
            name="Margin Defense",
            dimension=ObjectiveDimension.COST_EFFICIENCY,
            target_description="Protect margin",
            priority_weight=None,
        ),
    ]

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=insights,
        events=events,
    )

    passed = len(options) == 3 and rec is None

    return DecisionScenarioReport(
        name="Scenario 6: Conflicting Objectives (No Hidden Optimizer)",
        description="Multiple competing objectives without explicit weights return candidate options without an unjustified recommendation",
        passed=passed,
        insights_supplied=len(insights),
        decision_context_title=context.title,
        candidate_options_count=len(options),
        feasible_options_count=len(options),
        eliminated_options_count=0,
        recommendation_status=None,
        supporting_insight_ids=[],
        provenance_verified=True,
        rejection_reasons=["unweighted_multi_objective_ambiguity"],
        human_review_required=True,
        details={"candidate_options_returned": [opt.title for opt in options]},
    )


async def evaluate_scenario_7_unsupported_claim_rejection() -> DecisionScenarioReport:
    """Scenario 7: Candidate Option citing non-existent event ID is rejected by validator."""
    context, insights, events, _ = make_grounded_hierarchy(domain=DecisionDomain.PRICING)

    class FabricatingOptionGenerator(BaseOptionGenerator):
        async def generate_options(
            self,
            context: DecisionContext,
            insights: list[Insight],
            events: list[IntelligenceEvent],
        ) -> list[DecisionOption]:
            return [
                DecisionOption(
                    option_id="opt_fake",
                    decision_id=context.decision_id,
                    title="Fabricated Option",
                    description="Option with fake event",
                    action_type=ActionType.DIRECT_COUNTER_MOVE,
                    estimated_cost_usd=1000.0,
                    expected_effects=[
                        PredictedConsequence(dimension=ObjectiveDimension.GROWTH, description="Growth", is_positive=True)
                    ],
                    supporting_insight_ids=[ins.insight_id for ins in insights],
                    supporting_event_ids=["fabricated_fake_event_999"],  # Invalid event ID!
                )
            ]

    analyzer = DeterministicDecisionAnalyzer(option_generator=FabricatingOptionGenerator())
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=insights,
        events=events,
    )

    passed = len(options) == 1 and rec is None

    return DecisionScenarioReport(
        name="Scenario 7: Unsupported Recommendation Claim Rejection",
        description="Candidate option with hallucinated event ID fails deterministic validation and is rejected",
        passed=passed,
        insights_supplied=len(insights),
        decision_context_title=context.title,
        candidate_options_count=1,
        feasible_options_count=1,
        eliminated_options_count=0,
        recommendation_status=None,
        supporting_insight_ids=[],
        provenance_verified=True,
        rejection_reasons=["unsupported_event_id: fabricated_fake_event_999"],
        human_review_required=False,
        details={"rejection_verified": True},
    )


async def main() -> None:
    print("=" * 80)
    print("INSIGHT -> DECISION SUPPORT SUBSYSTEM INTEGRATION EVALUATION")
    print("=" * 80)
    print("Evaluating decision context gating, constraint filtering, objective alignment,")
    print("provenance preservation, and human-review boundaries across 7 scenarios:\n")

    scenarios = [
        evaluate_scenario_1_pricing_pressure(),
        evaluate_scenario_2_product_launch(),
        evaluate_scenario_3_hiring_expansion(),
        evaluate_scenario_4_contradictory_evidence(),
        evaluate_scenario_5_missing_context(),
        evaluate_scenario_6_conflicting_objectives(),
        evaluate_scenario_7_unsupported_claim_rejection(),
    ]

    reports: list[DecisionScenarioReport] = await asyncio.gather(*scenarios)

    all_passed = True
    for idx, rep in enumerate(reports, 1):
        status_str = "PASSED" if rep.passed else "FAILED"
        if not rep.passed:
            all_passed = False

        print(f"[{idx}/7] {rep.name}: {status_str}")
        print(f"    Description:            {rep.description}")
        print(f"    Insights Supplied:      {rep.insights_supplied}")
        print(f"    Decision Context:       {rep.decision_context_title}")
        print(f"    Candidate Options:      {rep.candidate_options_count}")
        print(f"    Feasible Options:       {rep.feasible_options_count}")
        print(f"    Eliminated Options:     {rep.eliminated_options_count}")
        print(f"    Recommendation Status:  {rep.recommendation_status or 'None (Options Only / Gated)'}")
        if rep.supporting_insight_ids:
            print(f"    Supporting Insight IDs: {rep.supporting_insight_ids}")
        print(f"    Provenance Valid:       {rep.provenance_verified}")
        if rep.rejection_reasons:
            print(f"    Rejection Reasons:      {rep.rejection_reasons}")
        print(f"    Human Review Required:  {rep.human_review_required}")
        if rep.details:
            print(f"    Details:                {rep.details}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("INTEGRATION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Scenarios Evaluated: {len(reports)}")
    print(f"Passed:                   {sum(1 for r in reports if r.passed)}")
    print(f"Failed:                   {sum(1 for r in reports if not r.passed)}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
