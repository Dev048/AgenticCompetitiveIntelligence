"""Unit tests for Decision Support and Recommendation subsystem (Phase 5)."""

from datetime import datetime, timedelta, timezone
from typing import Any
import pytest

from competitive_intelligence.decision_support import (
    ActionType,
    BaseDecisionAnalyzer,
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
    Recommendation,
    RecommendationConfidenceAssessment,
    RecommendationStatus,
    RiskTolerance,
)
from competitive_intelligence.intelligence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EvidenceReference,
    EvidenceRole,
    Insight,
    InsightStatus,
    InsightType,
    IntelligenceEvent,
    IntelligenceEventType,
    InterpretiveConfidenceAssessment,
    SignificanceAssessment,
    SourcePerspective,
    TimePrecision,
)
from competitive_intelligence.intelligence.models import EventTime


def _create_test_event(
    event_id: str,
    company_id: str = "acme_corp",
    workspace_id: str = "default_workspace",
    event_type: IntelligenceEventType = IntelligenceEventType.PRICING_CHANGE,
    subject: str = "Enterprise Plan",
) -> IntelligenceEvent:
    ev_time = EventTime(
        occurred_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        precision=TimePrecision.DATE_ONLY,
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        detected_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
    )
    conf = ConfidenceAssessment(
        score=0.90,
        level=ConfidenceLevel.HIGH,
        rationale="Grounded event.",
    )
    evidence_item = EvidenceReference(
        source_id="src_test",
        document_id=f"doc_{event_id}",
        version_id=f"ver_{event_id}",
        chunk_id=f"chunk_{event_id}",
        source_url="https://acme.com/news",
        source_category="news",
        extracted_text="Evidence text.",
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        role=EvidenceRole.PRIMARY_DIRECT,
    )
    return IntelligenceEvent(
        event_id=event_id,
        workspace_id=workspace_id,
        company_id=company_id,
        event_type=event_type,
        title="Pricing Update",
        summary="Changed pricing.",
        subject=subject,
        event_time=ev_time,
        confidence=conf,
        evidence=[evidence_item],
        primary_perspective=SourcePerspective.FIRST_PARTY,
    )


def _create_test_insight(
    insight_id: str,
    workspace_id: str = "default_workspace",
    insight_type: InsightType = InsightType.PRICING_STRATEGY,
    supporting_event_ids: list[str] | None = None,
) -> Insight:
    ev_ids = supporting_event_ids or ["e1"]
    return Insight(
        insight_id=insight_id,
        workspace_id=workspace_id,
        company_id="acme_corp",
        insight_type=insight_type,
        title="Pricing Insight",
        summary="Summary of pricing insight.",
        statement="Enterprise pricing underwent repeated adjustments.",
        supporting_signal_ids=["sig1"],
        supporting_event_ids=ev_ids,
        supporting_evidence_ids=[f"ev_ref_{eid}" for eid in ev_ids],
        factual_confidence=ConfidenceAssessment(score=0.88, level=ConfidenceLevel.HIGH),
        interpretive_confidence=InterpretiveConfidenceAssessment(score=0.82, level=ConfidenceLevel.HIGH),
        significance=SignificanceAssessment(),
        status=InsightStatus.ASSESSED,
    )


@pytest.fixture
def analyzer() -> DeterministicDecisionAnalyzer:
    return DeterministicDecisionAnalyzer()


@pytest.mark.anyio
async def test_missing_context_produces_no_options_or_recommendation(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify that an empty DecisionContext with no objectives yields ([], None)."""
    context = DecisionContext(
        title="Empty Decision Context",
        business_objectives=[],  # Invalid/empty
    )
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[],
        events=[],
    )
    assert options == []
    assert rec is None


@pytest.mark.anyio
async def test_feasible_options_without_weights_produces_no_recommendation(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify that multiple feasible options without explicit weights yields (options, None)."""
    context = DecisionContext(
        title="Pricing Strategy Response",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(
                name="Revenue Preservation",
                dimension=ObjectiveDimension.COST_EFFICIENCY,
                target_description="Maintain gross margins.",
            ),
            DecisionObjective(
                name="Market Share Defense",
                dimension=ObjectiveDimension.GROWTH,
                target_description="Protect entry-tier signups.",
            ),
        ],
    )
    e = _create_test_event("e1")
    ins = _create_test_insight("ins1", supporting_event_ids=["e1"])

    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[ins],
        events=[e],
    )

    assert len(options) == 3
    # No implicit optimizer: without explicit weights, no recommendation is picked
    assert rec is None


@pytest.mark.anyio
async def test_explicit_objective_weights_produce_justified_recommendation(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify that explicit priority weights select the aligned preferred option."""
    context = DecisionContext(
        title="Prioritized Growth Response",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(
                name="Aggressive Growth",
                dimension=ObjectiveDimension.GROWTH,
                target_description="Maximize user acquisition.",
                priority_weight=0.9,
            ),
            DecisionObjective(
                name="Margin Maintenance",
                dimension=ObjectiveDimension.COST_EFFICIENCY,
                target_description="Maintain margins.",
                priority_weight=0.1,
            ),
        ],
    )
    e = _create_test_event("e1")
    ins = _create_test_insight("ins1", supporting_event_ids=["e1"])

    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[ins],
        events=[e],
    )

    assert len(options) == 3
    assert rec is not None
    assert rec.status == RecommendationStatus.PROPOSED
    assert "Targeted Entry-Tier Price Match" in rec.title
    assert "highest alignment score" in rec.rationale


@pytest.mark.anyio
async def test_hard_constraint_eliminates_infeasible_options(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify that hard budget constraints eliminate expensive options and select the remaining feasible option."""
    context = DecisionContext(
        title="Budget-Constrained Product Decision",
        decision_domain=DecisionDomain.PRODUCT,
        business_objectives=[
            DecisionObjective(
                name="Platform Evolution",
                dimension=ObjectiveDimension.STRATEGIC_POSITIONING,
                target_description="Maintain platform moat.",
            )
        ],
        constraints=[
            DecisionConstraint(
                constraint_type=ConstraintType.BUDGET,
                description="Maximum available budget is $10,000.",
                is_hard_constraint=True,
                threshold_value=10000.0,
            )
        ],
    )
    e = _create_test_event("e_prod", event_type=IntelligenceEventType.RELEASE)
    ins = _create_test_insight("ins_prod", insight_type=InsightType.PRODUCT_STRATEGY, supporting_event_ids=["e_prod"])

    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[ins],
        events=[e],
    )

    assert len(options) == 2
    assert rec is not None
    # Accelerated feature ($30k) was eliminated; Roadmap focus ($4k) is selected
    assert "Maintain Core Platform Roadmap Focus" in rec.title
    assert len(rec.evaluated_alternatives) == 2
    eliminated = [alt for alt in rec.evaluated_alternatives if alt.is_eliminated]
    assert len(eliminated) == 1
    assert "exceeds hard budget limit" in str(eliminated[0].elimination_reason)


@pytest.mark.anyio
async def test_all_options_eliminated_produces_no_recommendation(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify that when all candidate options exceed budget constraints, no recommendation is made."""
    context = DecisionContext(
        title="Extremely Constrained Decision",
        decision_domain=DecisionDomain.HIRING_TALENT,
        business_objectives=[
            DecisionObjective(
                name="Talent Sourcing",
                dimension=ObjectiveDimension.GROWTH,
                target_description="Scale executive talent.",
            )
        ],
        constraints=[
            DecisionConstraint(
                constraint_type=ConstraintType.BUDGET,
                description="Zero budget available ($0).",
                is_hard_constraint=True,
                threshold_value=0.0,
            )
        ],
    )
    e = _create_test_event("e_hire", event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT)
    ins = _create_test_insight("ins_hire", insight_type=InsightType.HIRING_DIRECTION, supporting_event_ids=["e_hire"])

    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[ins],
        events=[e],
    )

    assert len(options) == 2
    assert rec is None


@pytest.mark.anyio
async def test_alternative_audit_trail_in_recommendation(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify that the Recommendation contains a complete audit trail of all evaluated alternatives."""
    context = DecisionContext(
        title="Product Roadmap Response",
        decision_domain=DecisionDomain.PRODUCT,
        business_objectives=[
            DecisionObjective(
                name="Positioning",
                dimension=ObjectiveDimension.STRATEGIC_POSITIONING,
                target_description="Platform positioning.",
                priority_weight=1.0,
            )
        ],
    )
    e = _create_test_event("e1")
    ins = _create_test_insight("ins1", supporting_event_ids=["e1"])

    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[ins],
        events=[e],
    )

    assert rec is not None
    assert len(rec.evaluated_alternatives) == 2
    selected = [alt for alt in rec.evaluated_alternatives if alt.is_selected]
    unselected = [alt for alt in rec.evaluated_alternatives if not alt.is_selected]
    assert len(selected) == 1
    assert len(unselected) == 1
    assert unselected[0].tradeoff_vs_selected is not None


@pytest.mark.anyio
async def test_predicted_consequences_contain_unassessed_confidence(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify that PredictedConsequence defaults to unassessed confidence (None), not hardcoded numbers."""
    context = DecisionContext(
        title="Pricing Context",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(name="Revenue", dimension=ObjectiveDimension.REVENUE, target_description="Target")
        ],
    )
    options = await analyzer.option_generator.generate_options(context=context, insights=[], events=[])
    assert len(options) > 0
    for opt in options:
        for eff in opt.expected_effects:
            assert eff.confidence_score is None  # Explicit unassessed confidence
            assert len(eff.underlying_assumptions) > 0


@pytest.mark.anyio
async def test_decision_urgency_derived_from_context(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify decision urgency is derived from explicit context deadlines, not merely from Insight presence."""
    near_deadline = datetime.now(timezone.utc) + timedelta(days=3)
    context_urgent = DecisionContext(
        title="Urgent Response",
        decision_domain=DecisionDomain.PRICING,
        deadline=near_deadline,
        business_objectives=[
            DecisionObjective(name="Revenue", dimension=ObjectiveDimension.COST_EFFICIENCY, target_description="Target", priority_weight=1.0)
        ],
    )
    e = _create_test_event("e1")
    ins = _create_test_insight("ins1", supporting_event_ids=["e1"])

    _, rec = await analyzer.analyze_and_recommend(context=context_urgent, insights=[ins], events=[e])
    assert rec is not None
    assert rec.urgency == DecisionUrgency.URGENT


@pytest.mark.anyio
async def test_four_confidence_dimensions_separation(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify that factual, interpretive, option analysis, and recommendation confidences remain distinct."""
    context = DecisionContext(
        title="Multi-Confidence Check",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(name="Revenue", dimension=ObjectiveDimension.COST_EFFICIENCY, target_description="Target", priority_weight=1.0)
        ],
    )
    e = _create_test_event("e1")
    ins = _create_test_insight("ins1", supporting_event_ids=["e1"])

    _, rec = await analyzer.analyze_and_recommend(context=context, insights=[ins], events=[e])
    assert rec is not None
    # 1. Underlying event factual confidence
    assert e.confidence.score == 0.90
    # 2. Insight interpretive confidence
    assert ins.interpretive_confidence.score == 0.82
    # 3. Recommendation confidence
    assert rec.confidence.score == 0.85
    # 4. Option analysis confidence
    assert rec.confidence.option_analysis_confidence == 0.80


@pytest.mark.anyio
async def test_transitive_provenance_traceability(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify complete transitive provenance chain from Recommendation -> Options -> Insights -> Events -> Evidence."""
    context = DecisionContext(
        title="Provenance Test",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(name="Revenue", dimension=ObjectiveDimension.COST_EFFICIENCY, target_description="Target", priority_weight=1.0)
        ],
    )
    e = _create_test_event("e_prov")
    ins = _create_test_insight("ins_prov", supporting_event_ids=["e_prov"])

    _, rec = await analyzer.analyze_and_recommend(context=context, insights=[ins], events=[e])
    assert rec is not None
    assert rec.supporting_insight_ids == ["ins_prov"]
    assert "e_prov" in rec.supporting_event_ids
    assert len(rec.supporting_evidence_ids) >= 1
    assert e.evidence[0].evidence_id in rec.supporting_evidence_ids


def test_validator_rejects_unregistered_option_id() -> None:
    """Verify validator fails recommendation referencing non-existent option ID."""
    validator = DeterministicRecommendationValidator()
    context = DecisionContext(
        title="Test",
        business_objectives=[DecisionObjective(name="Test", dimension=ObjectiveDimension.GROWTH, target_description="Test")],
    )
    e = _create_test_event("e1")
    ins = _create_test_insight("ins1")

    opt = DecisionOption(
        option_id="opt_valid",
        decision_id=context.decision_id,
        title="Valid Option",
        description="Desc",
        action_type=ActionType.MONITOR_AND_WAIT,
    )

    invalid_rec = Recommendation(
        decision_id=context.decision_id,
        selected_option_id="opt_non_existent",  # Invalid ID
        title="Title",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="Test"),
        supporting_event_ids=["e1"],
    )

    is_valid, errors = validator.validate(invalid_rec, context=context, options=[opt], insights=[ins], events=[e])
    assert is_valid is False
    assert any("does not exist in evaluated candidate options" in err for err in errors)


def test_validator_rejects_missing_event_provenance() -> None:
    """Verify validator enforces supporting event provenance."""
    validator = DeterministicRecommendationValidator()
    context = DecisionContext(
        title="Test",
        business_objectives=[DecisionObjective(name="Test", dimension=ObjectiveDimension.GROWTH, target_description="Test")],
    )
    opt = DecisionOption(
        option_id="opt_1",
        decision_id=context.decision_id,
        title="Valid Option",
        description="Desc",
        action_type=ActionType.MONITOR_AND_WAIT,
    )

    invalid_rec = Recommendation(
        decision_id=context.decision_id,
        selected_option_id="opt_1",
        title="Title",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="Test"),
        supporting_event_ids=["non_existent_event"],  # Missing from context
    )

    is_valid, errors = validator.validate(invalid_rec, context=context, options=[opt], insights=[], events=[])
    assert is_valid is False
    assert any("does not exist in candidate events pool" in err for err in errors)


@pytest.mark.anyio
async def test_recommendation_lifecycle_state_starts_at_proposed(
    analyzer: DeterministicDecisionAnalyzer,
) -> None:
    """Verify recommendation status begins as PROPOSED."""
    context = DecisionContext(
        title="Lifecycle Check",
        decision_domain=DecisionDomain.PRODUCT,
        business_objectives=[
            DecisionObjective(name="Positioning", dimension=ObjectiveDimension.STRATEGIC_POSITIONING, target_description="Target", priority_weight=1.0)
        ],
    )
    e = _create_test_event("e1")
    ins = _create_test_insight("ins1", supporting_event_ids=["e1"])

    _, rec = await analyzer.analyze_and_recommend(context=context, insights=[ins], events=[e])
    assert rec is not None
    assert rec.status == RecommendationStatus.PROPOSED


def test_human_acceptance_state_transitions() -> None:
    """Verify human decision lifecycle transitions (PROPOSED -> ACCEPTED / REJECTED)."""
    rec = Recommendation(
        decision_id="dec_1",
        selected_option_id="opt_1",
        title="Title",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="Test"),
        supporting_event_ids=["e1"],
    )
    assert rec.status == RecommendationStatus.PROPOSED

    # Executive accepts recommendation
    rec.status = RecommendationStatus.ACCEPTED
    assert rec.status == RecommendationStatus.ACCEPTED


def test_superseding_recommendation_lineage() -> None:
    """Verify updating a recommendation populates supersedes_recommendation_id."""
    rec_orig = Recommendation(
        recommendation_id="rec_v1",
        decision_id="dec_1",
        selected_option_id="opt_1",
        title="Original Recommendation",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="Test"),
        supporting_event_ids=["e1"],
    )

    rec_updated = Recommendation(
        recommendation_id="rec_v2",
        decision_id="dec_1",
        selected_option_id="opt_2",
        title="Updated Recommendation",
        summary="Summary v2",
        rationale="Updated rationale",
        evaluated_alternatives=[],
        expected_benefits=["Updated Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH, rationale="Test"),
        supporting_event_ids=["e1"],
        supersedes_recommendation_id=rec_orig.recommendation_id,
    )

    assert rec_updated.supersedes_recommendation_id == "rec_v1"


@pytest.mark.anyio
async def test_custom_decision_analyzer_injection() -> None:
    """Verify dependency injection of custom BaseDecisionAnalyzer implementations."""

    class MockCustomDecisionAnalyzer(BaseDecisionAnalyzer):
        async def analyze_and_recommend(
            self,
            context: DecisionContext,
            insights: list[Insight],
            events: list[IntelligenceEvent],
            existing_recommendations: list[Recommendation] | None = None,
        ) -> tuple[list[DecisionOption], Recommendation | None]:
            opt = DecisionOption(
                option_id="opt_mock",
                decision_id=context.decision_id,
                title="Mock Option",
                description="Mock description",
                action_type=ActionType.OTHER,
            )
            rec = Recommendation(
                decision_id=context.decision_id,
                selected_option_id="opt_mock",
                title="Mock Recommendation",
                summary="Mock summary",
                rationale="Mock rationale",
                evaluated_alternatives=[],
                expected_benefits=["Mock Benefit"],
                confidence=RecommendationConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH, rationale="Mock"),
                supporting_event_ids=[events[0].event_id],
            )
            return [opt], rec

    custom_analyzer = MockCustomDecisionAnalyzer()
    context = DecisionContext(
        title="Custom Context",
        business_objectives=[DecisionObjective(name="Custom", dimension=ObjectiveDimension.GROWTH, target_description="Target")],
    )
    e = _create_test_event("e_custom")
    options, rec = await custom_analyzer.analyze_and_recommend(context=context, insights=[], events=[e])

    assert len(options) == 1
    assert rec is not None
    assert rec.title == "Mock Recommendation"
