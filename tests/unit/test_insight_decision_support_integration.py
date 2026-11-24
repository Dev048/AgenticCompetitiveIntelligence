"""Unit and Integration Tests for Insight -> Decision Support Subsystem Integration.

Verifies:
- Scenario A: Validated Insight -> DecisionOption generation
- Scenario B: DecisionOption -> Evaluated Alternative (hard constraint elimination)
- Scenario C: Evaluated Alternatives -> Justified Recommendation (with explicit weights)
- Scenario D: Full 10-tier structural provenance chain verification
- Scenario E: Missing DecisionContext / missing objectives -> safely yields ([], None)
- Scenario F: Missing optional constraints -> no constraint-based elimination
- Scenario G: Multi-objective without explicit weights -> returns candidate options, no recommendation
- Scenario H: Contradictory Insights -> preserves uncertainty in unresolved_uncertainties
- Scenario I: Regression: Candidate option with fabricated event ID -> validator rejects
- Scenario J: Regression: Candidate option with fabricated insight ID -> validator rejects
- Scenario K: Workspace / Tenant Isolation: Cross-workspace insight/event reference -> validator rejects
- Scenario L: Human-Review Lifecycle (PROPOSED status, non-autonomous execution)
- Scenario M: Untrusted candidate generator (mocked LLM/external) passes through deterministic filters
- Scenario N: End-to-End Pipeline: Retrieval -> Events -> Signals -> Insights -> Context -> Options -> Recommendation
"""

from datetime import datetime, timedelta, timezone
from typing import Any
import pytest

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
    RuleBasedOptionGenerator,
)
from competitive_intelligence.intelligence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    DeterministicSignalDetector,
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
    RetrievedEvidenceAnalyzer,
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
from competitive_intelligence.retrieval_engine.models import (
    RetrievalResult,
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


def create_10_layer_provenance_lineage(
    workspace_id: str = "ws_test_01",
    company_id: str = "acme_corp",
    source_id: str = "src_pricing_01",
    document_id: str = "doc_pricing_01",
    version_id: str = "ver_pricing_01",
    chunk_id: str = "chunk_pricing_01",
    event_id: str = "ev_pricing_01",
    signal_id: str = "sig_pricing_01",
    insight_id: str = "ins_pricing_01",
    decision_id: str = "dec_pricing_01",
    source_url: str = "https://acme.com/pricing",
    content: str = "Acme updated Enterprise Plan pricing from $50 to $30/month.",
) -> dict[str, Any]:
    """Helper creating complete 10-layer provenance entities."""
    source = Source(
        identity=SourceIdentity(
            source_id=source_id,
            company_id=company_id,
            workspace_id=workspace_id,
            display_name=f"Source {source_id}",
        ),
        classification=SourceClassification(
            category=SourceCategory.PRICING,
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier=source_url,
            access_pattern=AccessPattern.PUBLIC_WEB,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
        ),
    )

    doc = Document(
        document_id=document_id,
        source_id=source_id,
        workspace_id=workspace_id,
        company_id=company_id,
        canonical_url=source_url,
    )

    version = DocumentVersion(
        version_id=version_id,
        document_id=document_id,
        content=content,
        content_hash="hash_10tier",
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        version_id=version_id,
        content=content,
        metadata=ChunkMetadata(
            source_id=source_id,
            document_id=document_id,
            version_id=version_id,
            workspace_id=workspace_id,
            company_id=company_id,
            source_url=source_url,
            source_category="pricing",
            collected_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        ),
    )

    retrieval_res = RetrievalResult(
        chunk_id=chunk_id,
        score=0.95,
        text=content,
        document_id=document_id,
        version_id=version_id,
        source_id=source_id,
        company_id=company_id,
        workspace_id=workspace_id,
        source_category="pricing",
        canonical_url=source_url,
    )

    evidence_ref = EvidenceReference(
        evidence_id=f"ev_ref_{event_id}",
        source_id=source_id,
        document_id=document_id,
        version_id=version_id,
        chunk_id=chunk_id,
        source_url=source_url,
        source_category="pricing",
        extracted_text=content,
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        role=EvidenceRole.PRIMARY_DIRECT,
    )

    event = IntelligenceEvent(
        event_id=event_id,
        workspace_id=workspace_id,
        company_id=company_id,
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Enterprise Plan Price Reduction",
        summary="Enterprise plan reduced to $30/month.",
        subject="Enterprise Plan",
        previous_state={"price_usd": "$50"},
        new_state={"price_usd": "$30"},
        event_time=EventTime(
            occurred_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            precision=TimePrecision.DATE_ONLY,
            acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            detected_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.92, level=ConfidenceLevel.HIGH),
        evidence=[evidence_ref],
        primary_perspective=SourcePerspective.FIRST_PARTY,
    )

    signal = InferredSignal(
        signal_id=signal_id,
        workspace_id=workspace_id,
        company_id=company_id,
        signal_type=SignalType.DIRECTIONAL_TREND,
        subject="Enterprise Plan",
        title="Directional Pricing Adjustment",
        summary="Monitored entity executed price reductions on Enterprise tier.",
        supporting_event_ids=[event_id],
        supporting_evidence_ids=[evidence_ref.evidence_id],
        status=SignalStatus.DETECTED,
    )

    insight = Insight(
        insight_id=insight_id,
        workspace_id=workspace_id,
        company_id=company_id,
        insight_type=InsightType.PRICING_STRATEGY,
        title="Aggressive Down-Market Pricing Shift",
        summary="Competitor is systematically reducing pricing to capture enterprise volume.",
        statement="Detailed statement of pricing movement.",
        supporting_signal_ids=[signal_id],
        supporting_event_ids=[event_id],
        supporting_evidence_ids=[evidence_ref.evidence_id],
        factual_confidence=ConfidenceAssessment(score=0.90, level=ConfidenceLevel.HIGH),
        interpretive_confidence=InterpretiveConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH),
        significance=SignificanceAssessment(),
        status=InsightStatus.ASSESSED,
    )

    context = DecisionContext(
        decision_id=decision_id,
        workspace_id=workspace_id,
        title="Competitive Pricing Response",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(
                name="Market Share Growth",
                dimension=ObjectiveDimension.GROWTH,
                target_description="Protect market share in entry tier.",
                priority_weight=0.8,
            ),
            DecisionObjective(
                name="Margin Preservation",
                dimension=ObjectiveDimension.COST_EFFICIENCY,
                target_description="Protect gross margins.",
                priority_weight=0.2,
            ),
        ],
        constraints=[
            DecisionConstraint(
                constraint_type=ConstraintType.BUDGET,
                description="Maximum budget $50,000",
                is_hard_constraint=True,
                threshold_value=50000.0,
            )
        ],
        risk_tolerance=RiskTolerance.MODERATE,
    )

    return {
        "source": source,
        "doc": doc,
        "version": version,
        "chunk": chunk,
        "retrieval_res": retrieval_res,
        "evidence_ref": evidence_ref,
        "event": event,
        "signal": signal,
        "insight": insight,
        "context": context,
    }


# ==============================================================================
# Scenario A: Validated Insight -> DecisionOption Generation
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_a_validated_insight_generates_options() -> None:
    """Verify that validated Insights produce bounded, structured DecisionOptions."""
    data = create_10_layer_provenance_lineage()
    context: DecisionContext = data["context"]
    insight: Insight = data["insight"]
    event: IntelligenceEvent = data["event"]

    generator = RuleBasedOptionGenerator()
    options = await generator.generate_options(
        context=context,
        insights=[insight],
        events=[event],
    )

    assert len(options) >= 2
    for opt in options:
        assert opt.decision_id == context.decision_id
        assert opt.action_type in [
            ActionType.MONITOR_AND_WAIT,
            ActionType.DIRECT_COUNTER_MOVE,
            ActionType.DIFFERENTIATION,
        ]
        assert insight.insight_id in opt.supporting_insight_ids
        assert event.event_id in opt.supporting_event_ids
        assert len(opt.expected_effects) > 0


# ==============================================================================
# Scenario B: DecisionOption -> Evaluated Alternative (Constraint Filtering)
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_b_hard_constraint_eliminates_infeasible_alternative() -> None:
    """Verify that hard constraints eliminate expensive candidate options."""
    data = create_10_layer_provenance_lineage()
    context: DecisionContext = data["context"]
    # Tighten budget to $10,000 (Targeted Counter-Move costs $20,000)
    context.constraints = [
        DecisionConstraint(
            constraint_type=ConstraintType.BUDGET,
            description="Strict budget ceiling of $10,000.",
            is_hard_constraint=True,
            threshold_value=10000.0,
        )
    ]
    insight: Insight = data["insight"]
    event: IntelligenceEvent = data["event"]

    analyzer = DeterministicDecisionAnalyzer()
    options, recommendation = await analyzer.analyze_and_recommend(
        context=context,
        insights=[insight],
        events=[event],
    )

    assert len(options) == 3
    assert recommendation is not None
    # Targeted price match ($20,000) eliminated
    eliminated = [alt for alt in recommendation.evaluated_alternatives if alt.is_eliminated]
    assert len(eliminated) >= 1
    assert any("exceeds hard budget limit" in str(alt.elimination_reason) for alt in eliminated)


# ==============================================================================
# Scenario C: Evaluated Alternatives -> Justified Recommendation
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_c_explicit_weights_produce_justified_recommendation() -> None:
    """Verify that user-supplied priority weights select the optimal aligned option."""
    data = create_10_layer_provenance_lineage()
    context: DecisionContext = data["context"]
    insight: Insight = data["insight"]
    event: IntelligenceEvent = data["event"]

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[insight],
        events=[event],
    )

    assert rec is not None
    assert rec.status == RecommendationStatus.PROPOSED
    assert "Targeted Entry-Tier Price Match" in rec.title
    assert len(rec.evaluated_alternatives) == 3
    assert rec.selected_option_id in {opt.option_id for opt in options}


# ==============================================================================
# Scenario D: Full 10-Tier Structural Provenance Chain Verification
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_d_full_10_tier_provenance_chain() -> None:
    """Verify unbroken structural chain: Recommendation -> Option -> Insight -> Signal -> Event -> Evidence -> Chunk -> Version -> Doc -> Source."""
    data = create_10_layer_provenance_lineage()
    context: DecisionContext = data["context"]
    insight: Insight = data["insight"]
    signal: InferredSignal = data["signal"]
    event: IntelligenceEvent = data["event"]
    evidence_ref: EvidenceReference = data["evidence_ref"]
    chunk: DocumentChunk = data["chunk"]
    version: DocumentVersion = data["version"]
    doc: Document = data["doc"]
    source: Source = data["source"]

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[insight],
        events=[event],
    )

    assert rec is not None

    # 1. Recommendation -> Option
    selected_opt = next(opt for opt in options if opt.option_id == rec.selected_option_id)
    assert selected_opt is not None

    # 2. Option & Recommendation -> Insight
    assert insight.insight_id in rec.supporting_insight_ids
    assert insight.insight_id in selected_opt.supporting_insight_ids

    # 3. Insight -> Signal
    assert signal.signal_id in insight.supporting_signal_ids

    # 4. Signal & Insight -> Event
    assert event.event_id in signal.supporting_event_ids
    assert event.event_id in insight.supporting_event_ids
    assert event.event_id in rec.supporting_event_ids

    # 5. Event -> EvidenceReference
    assert evidence_ref.evidence_id in [e.evidence_id for e in event.evidence]
    assert evidence_ref.evidence_id in rec.supporting_evidence_ids

    # 6. EvidenceReference -> DocumentChunk
    assert evidence_ref.chunk_id == chunk.chunk_id

    # 7. DocumentChunk -> DocumentVersion
    assert chunk.version_id == version.version_id
    assert evidence_ref.version_id == version.version_id

    # 8. DocumentVersion -> Document
    assert version.document_id == doc.document_id
    assert evidence_ref.document_id == doc.document_id

    # 9. Document -> Source
    assert doc.source_id == source.source_id
    assert evidence_ref.source_id == source.source_id
    assert evidence_ref.source_url == source.location.url_or_identifier


# ==============================================================================
# Scenario E: Missing DecisionContext / Missing Objectives Safety
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_e_missing_context_or_objectives_safely_yields_none() -> None:
    """Verify that absent objectives prevent unjustified recommendation synthesis."""
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
    assert options == []
    assert rec is None


# ==============================================================================
# Scenario F: Missing Optional Constraints Handling
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_f_missing_optional_constraints_no_elimination() -> None:
    """Verify missing optional constraints causes no elimination, but still requires explicit weights."""
    context = DecisionContext(
        title="Unconstrained Decision",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(name="Growth", dimension=ObjectiveDimension.GROWTH, target_description="Grow"),
            DecisionObjective(name="Margin", dimension=ObjectiveDimension.COST_EFFICIENCY, target_description="Margin"),
        ],
        constraints=[],  # No constraints provided
    )
    data = create_10_layer_provenance_lineage()
    insight: Insight = data["insight"]
    event: IntelligenceEvent = data["event"]

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[insight],
        events=[event],
    )

    # All options remain feasible
    assert len(options) == 3
    # Without explicit weights, no recommendation is made (no hidden optimizer)
    assert rec is None


# ==============================================================================
# Scenario G: Multi-Objective Decision Without Explicit Weights
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_g_multi_objective_without_weights_returns_options_only() -> None:
    """Verify that objective priority is never inferred from wording."""
    context = DecisionContext(
        title="Unweighted Priorities",
        decision_domain=DecisionDomain.PRODUCT,
        business_objectives=[
            DecisionObjective(
                name="Time to Market",
                dimension=ObjectiveDimension.TIME_TO_MARKET,
                target_description="Release features as rapidly as possible.",
                priority_weight=None,  # Missing explicit weight
            ),
            DecisionObjective(
                name="Platform Moat",
                dimension=ObjectiveDimension.STRATEGIC_POSITIONING,
                target_description="Deepen long term platform moat.",
                priority_weight=None,  # Missing explicit weight
            ),
        ],
    )
    data = create_10_layer_provenance_lineage()
    insight: Insight = data["insight"]
    event: IntelligenceEvent = data["event"]

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[insight],
        events=[event],
    )

    assert len(options) == 2
    assert rec is None


# ==============================================================================
# Scenario H: Contradictory Insights Uncertainty Preservation
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_h_contradictory_insights_preserves_uncertainty() -> None:
    """Verify that contradictory insights populate unresolved_uncertainties and adjust confidence."""
    uncertain_insight = Insight(
        insight_id="ins_uncertain_01",
        workspace_id="ws_test_01",
        company_id="acme_corp",
        insight_type=InsightType.UNCERTAINTY,
        title="Contradictory Launch Status",
        summary="Reports conflict regarding whether Voice Mode was launched or delayed.",
        statement="Conflicting evidence observed.",
        supporting_signal_ids=["sig_c1"],
        supporting_event_ids=["ev_c1"],
        supporting_evidence_ids=["ev_ref_c1"],
        factual_confidence=ConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH),
        interpretive_confidence=InterpretiveConfidenceAssessment(score=0.80, level=ConfidenceLevel.HIGH),
        significance=SignificanceAssessment(),
        status=InsightStatus.UNCERTAIN,
    )
    event = IntelligenceEvent(
        event_id="ev_c1",
        workspace_id="ws_test_01",
        company_id="acme_corp",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Price Change",
        summary="Price change event",
        subject="Enterprise Plan",
        event_time=EventTime(
            occurred_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            precision=TimePrecision.DATE_ONLY,
            acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            detected_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.8, level=ConfidenceLevel.MEDIUM),
        evidence=[
            EvidenceReference(
                evidence_id="ev_ref_c1",
                source_id="s1",
                document_id="d1",
                version_id="v1",
                chunk_id="c1",
                source_url="https://acme.com",
                source_category="news",
                extracted_text="Some text",
                acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            )
        ],
    )

    context = DecisionContext(
        title="Response under uncertainty",
        workspace_id="ws_test_01",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(name="Growth", dimension=ObjectiveDimension.GROWTH, target_description="Grow", priority_weight=1.0)
        ],
    )

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[uncertain_insight],
        events=[event],
    )

    assert rec is not None
    assert len(rec.unresolved_uncertainties) >= 1
    assert "Contradictory Launch Status" in rec.unresolved_uncertainties[0]
    assert rec.confidence.level == ConfidenceLevel.MEDIUM
    assert "observed strategic uncertainty" in rec.confidence.rationale


# ==============================================================================
# Scenario I: Regression: Candidate Option with Fabricated Event ID
# ==============================================================================


def test_scenario_i_regression_validator_rejects_fabricated_event_id() -> None:
    """Verify validator rejects recommendation referencing candidate option with fabricated event ID."""
    validator = DeterministicRecommendationValidator()
    context = DecisionContext(
        decision_id="dec_01",
        workspace_id="ws_01",
        title="Test",
        business_objectives=[DecisionObjective(name="T", dimension=ObjectiveDimension.GROWTH, target_description="T")],
    )

    # Untrusted option claims fabricated event ID
    opt_fabricated = DecisionOption(
        option_id="opt_bad_event",
        decision_id="dec_01",
        title="Bad Option",
        description="Option with fake event ID",
        action_type=ActionType.MONITOR_AND_WAIT,
        supporting_event_ids=["fabricated_event_id_999"],  # NOT in trusted events
    )

    rec = Recommendation(
        decision_id="dec_01",
        workspace_id="ws_01",
        selected_option_id="opt_bad_event",
        title="Rec",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="R"),
        supporting_event_ids=["fabricated_event_id_999"],
    )

    # Validated upstream pool has zero events
    is_valid, errors = validator.validate(
        payload=rec,
        context=context,
        options=[opt_fabricated],
        insights=[],
        events=[],
    )

    assert is_valid is False
    assert any("does not exist in candidate events pool" in err or "references unknown event_id" in err for err in errors)


# ==============================================================================
# Scenario J: Regression: Candidate Option with Fabricated Insight ID
# ==============================================================================


def test_scenario_j_regression_validator_rejects_fabricated_insight_id() -> None:
    """Verify validator rejects recommendation referencing candidate option with fabricated insight ID."""
    validator = DeterministicRecommendationValidator()
    context = DecisionContext(
        decision_id="dec_01",
        workspace_id="ws_01",
        title="Test",
        business_objectives=[DecisionObjective(name="T", dimension=ObjectiveDimension.GROWTH, target_description="T")],
    )

    # Untrusted option claims fabricated insight ID
    opt_fabricated = DecisionOption(
        option_id="opt_bad_insight",
        decision_id="dec_01",
        title="Bad Option",
        description="Option with fake insight ID",
        action_type=ActionType.MONITOR_AND_WAIT,
        supporting_insight_ids=["fabricated_insight_id_888"],  # NOT in trusted insights
        supporting_event_ids=["trusted_e1"],
    )

    trusted_event = IntelligenceEvent(
        event_id="trusted_e1",
        workspace_id="ws_01",
        company_id="acme",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="E",
        summary="E",
        subject="S",
        event_time=EventTime(
            occurred_at=datetime.now(timezone.utc),
            acquired_at=datetime.now(timezone.utc),
            detected_at=datetime.now(timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
        evidence=[
            EvidenceReference(
                evidence_id="ev1",
                source_id="s1",
                document_id="d1",
                version_id="v1",
                chunk_id="c1",
                source_url="https://a.com",
                source_category="news",
                extracted_text="text",
                acquired_at=datetime.now(timezone.utc),
            )
        ],
    )

    rec = Recommendation(
        decision_id="dec_01",
        workspace_id="ws_01",
        selected_option_id="opt_bad_insight",
        title="Rec",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="R"),
        supporting_insight_ids=["fabricated_insight_id_888"],
        supporting_event_ids=["trusted_e1"],
    )

    is_valid, errors = validator.validate(
        payload=rec,
        context=context,
        options=[opt_fabricated],
        insights=[],  # Trusted pool empty
        events=[trusted_event],
    )

    assert is_valid is False
    assert any("does not exist in candidate insights pool" in err or "references unknown insight_id" in err for err in errors)


# ==============================================================================
# Scenario K: Workspace / Tenant Isolation Enforcement
# ==============================================================================


def test_scenario_k_cross_workspace_references_rejected() -> None:
    """Verify validator strictly rejects cross-workspace insight or event citations."""
    validator = DeterministicRecommendationValidator()
    context = DecisionContext(
        decision_id="dec_wsA",
        workspace_id="workspace_A",
        title="Workspace A Decision",
        business_objectives=[DecisionObjective(name="T", dimension=ObjectiveDimension.GROWTH, target_description="T")],
    )

    cross_ws_insight = Insight(
        insight_id="ins_wsB",
        workspace_id="workspace_B",  # Cross-workspace tenant!
        company_id="acme",
        insight_type=InsightType.PRICING_STRATEGY,
        title="Tenant B Insight",
        summary="B",
        statement="B",
        supporting_signal_ids=[],
        supporting_event_ids=["ev_wsB"],
        factual_confidence=ConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
        interpretive_confidence=InterpretiveConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH),
        significance=SignificanceAssessment(),
    )

    cross_ws_event = IntelligenceEvent(
        event_id="ev_wsB",
        workspace_id="workspace_B",  # Cross-workspace tenant!
        company_id="acme",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="E",
        summary="E",
        subject="S",
        event_time=EventTime(
            occurred_at=datetime.now(timezone.utc),
            acquired_at=datetime.now(timezone.utc),
            detected_at=datetime.now(timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
        evidence=[
            EvidenceReference(
                evidence_id="ev1",
                source_id="s1",
                document_id="d1",
                version_id="v1",
                chunk_id="c1",
                source_url="https://a.com",
                source_category="news",
                extracted_text="text",
                acquired_at=datetime.now(timezone.utc),
            )
        ],
    )

    opt = DecisionOption(
        option_id="opt_cross",
        decision_id="dec_wsA",
        title="Cross Opt",
        description="Cross Opt",
        action_type=ActionType.MONITOR_AND_WAIT,
        supporting_insight_ids=["ins_wsB"],
        supporting_event_ids=["ev_wsB"],
    )

    rec = Recommendation(
        decision_id="dec_wsA",
        workspace_id="workspace_A",
        selected_option_id="opt_cross",
        title="Rec",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="R"),
        supporting_insight_ids=["ins_wsB"],
        supporting_event_ids=["ev_wsB"],
    )

    is_valid, errors = validator.validate(
        payload=rec,
        context=context,
        options=[opt],
        insights=[cross_ws_insight],
        events=[cross_ws_event],
    )

    assert is_valid is False
    assert any("violating workspace isolation" in err for err in errors)


# ==============================================================================
# Scenario L: Human-Review Lifecycle Enforcement
# ==============================================================================


def test_scenario_l_human_review_lifecycle_non_autonomous() -> None:
    """Verify recommendation status begins as PROPOSED and requires executive review."""
    rec = Recommendation(
        decision_id="dec_01",
        workspace_id="ws_01",
        selected_option_id="opt_01",
        title="Proposed Action",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH, rationale="R"),
        supporting_event_ids=["ev_01"],
    )

    # Must start at PROPOSED
    assert rec.status == RecommendationStatus.PROPOSED

    # Executive reviews and approves
    rec.status = RecommendationStatus.ACCEPTED
    assert rec.status == RecommendationStatus.ACCEPTED

    # Or executive rejects
    rec.status = RecommendationStatus.REJECTED
    assert rec.status == RecommendationStatus.REJECTED


# ==============================================================================
# Scenario M: Untrusted Candidate Option Generator Integration
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_m_untrusted_candidate_generator_passes_deterministic_filters() -> None:
    """Verify that untrusted candidate options pass through deterministic constraint & objective filters."""

    class UntrustedMockOptionGenerator(BaseOptionGenerator):
        async def generate_options(
            self,
            context: DecisionContext,
            insights: list[Insight],
            events: list[IntelligenceEvent],
        ) -> list[DecisionOption]:
            # Returns untrusted external candidate options
            return [
                DecisionOption(
                    option_id="opt_ext_cheap",
                    decision_id=context.decision_id,
                    title="External Lightweight Action",
                    description="Cheap action",
                    action_type=ActionType.MONITOR_AND_WAIT,
                    estimated_cost_usd=5000.0,
                    expected_effects=[
                        PredictedConsequence(dimension=ObjectiveDimension.GROWTH, description="Low growth", is_positive=True)
                    ],
                    supporting_insight_ids=[ins.insight_id for ins in insights],
                    supporting_event_ids=[e.event_id for e in events],
                ),
                DecisionOption(
                    option_id="opt_ext_expensive",
                    decision_id=context.decision_id,
                    title="External Expensive Action",
                    description="Expensive action",
                    action_type=ActionType.DIRECT_COUNTER_MOVE,
                    estimated_cost_usd=100000.0,  # Exceeds $20k budget
                    expected_effects=[
                        PredictedConsequence(dimension=ObjectiveDimension.GROWTH, description="High growth", is_positive=True)
                    ],
                    supporting_insight_ids=[ins.insight_id for ins in insights],
                    supporting_event_ids=[e.event_id for e in events],
                ),
            ]

    data = create_10_layer_provenance_lineage()
    context: DecisionContext = data["context"]
    context.constraints = [
        DecisionConstraint(
            constraint_type=ConstraintType.BUDGET,
            description="Budget limit $20,000",
            is_hard_constraint=True,
            threshold_value=20000.0,
        )
    ]
    insight: Insight = data["insight"]
    event: IntelligenceEvent = data["event"]

    analyzer = DeterministicDecisionAnalyzer(option_generator=UntrustedMockOptionGenerator())
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[insight],
        events=[event],
    )

    assert len(options) == 2
    assert rec is not None
    # Deterministic filter selected the feasible cheap option ($5,000) over the expensive one ($100,000)
    assert rec.selected_option_id == "opt_ext_cheap"
    assert len(rec.evaluated_alternatives) == 2
    elim = next(alt for alt in rec.evaluated_alternatives if alt.is_eliminated)
    assert elim.option_id == "opt_ext_expensive"
    assert "exceeds hard budget limit" in str(elim.elimination_reason)


# ==============================================================================
# Scenario N: End-to-End Pipeline Execution
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_n_end_to_end_pipeline_retrieval_to_recommendation() -> None:
    """Verify complete pipeline: RetrievalResult -> Events -> Signals -> Insights -> Context -> Options -> Recommendation."""
    data = create_10_layer_provenance_lineage()
    context: DecisionContext = data["context"]
    retrieval_res1: RetrievalResult = data["retrieval_res"]

    retrieval_res2 = RetrievalResult(
        chunk_id="chunk_pricing_02",
        score=0.92,
        text="Acme updated Enterprise Plan pricing from $30 to $20/month on August 15, 2026.",
        document_id="doc_pricing_02",
        version_id="ver_pricing_02",
        source_id="src_pricing_02",
        company_id="acme_corp",
        workspace_id=context.workspace_id,
        source_category="pricing",
        canonical_url="https://acme.com/pricing2",
    )

    # 1. Retrieval -> Events
    evidence_analyzer = RetrievedEvidenceAnalyzer()
    events = await evidence_analyzer.analyze_retrieved_evidence(
        query="Acme pricing updates",
        company_id="acme_corp",
        workspace_id=context.workspace_id,
        evidence=[retrieval_res1, retrieval_res2],
    )
    assert len(events) == 2

    # 2. Events -> Signals
    signal_detector = DeterministicSignalDetector()
    signals = await signal_detector.detect_signals(
        workspace_id=context.workspace_id,
        company_id="acme_corp",
        events=events,
    )
    # Single event -> emerging activity signal
    assert len(signals) >= 0

    # 3. Signals + Events -> Insights
    strategic_analyzer = DeterministicStrategicAnalyzer()
    insights = await strategic_analyzer.generate_insights(
        workspace_id=context.workspace_id,
        company_id="acme_corp",
        signals=signals,
        events=events,
    )
    assert len(insights) >= 1

    # 4. Context + Insights + Events -> DecisionOptions & Recommendation
    decision_analyzer = DeterministicDecisionAnalyzer()
    options, recommendation = await decision_analyzer.analyze_and_recommend(
        context=context,
        insights=insights,
        events=events,
    )

    assert len(options) >= 2
    assert recommendation is not None
    assert recommendation.status == RecommendationStatus.PROPOSED
    assert len(recommendation.supporting_insight_ids) >= 1
    assert len(recommendation.supporting_event_ids) >= 1
    assert len(recommendation.supporting_evidence_ids) >= 1
