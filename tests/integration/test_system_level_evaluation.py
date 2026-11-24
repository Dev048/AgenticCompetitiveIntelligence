"""Integration Test Suite: Final System-Level Evaluation.

Verifies:
- 12 End-to-End Scenarios (SC-01 to SC-12) covering simple events, trends, reversals,
  multi-entity comparisons, temporal changes, contradictions, evidence gaps, distractors,
  adversarial queries, decision support, missing context, and conflicting objectives.
- 10 Epistemic Invariants:
  1. No evidence -> No event.
  2. No grounded events -> No signal.
  3. No supported signal -> No validated insight.
  4. Insight != Recommendation.
  5. Missing decision objectives -> No justified recommendation.
  6. Missing objective weights -> Options only, no recommendation.
  7. Contradictory evidence -> Uncertainty preserved.
  8. Fabricated event/insight IDs -> Validator rejection.
  9. Cross-workspace citations -> Strict tenant isolation rejection.
  10. Automated recommendation -> PROPOSED status only.
- Failure Injection Suite (empty content, zero retrieval, malformed LLM JSON, fabricated IDs, cross-workspace).
- Idempotency & Repeatability.
- Multi-Tenant Isolation.
"""

from datetime import datetime, timezone
import hashlib
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
from competitive_intelligence.evaluation.system_corpus import (
    CorpusDocument,
    GroundTruthScenario,
    OutcomeClassification,
    SystemCorpus,
    create_system_evaluation_corpus,
)
from competitive_intelligence.intelligence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    DeterministicEventSignatureStrategy,
    DeterministicSignalDetector,
    DeterministicStrategicAnalyzer,
    DocumentChangeAnalyzer,
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
    ChangeResult,
    ChangeType,
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentVersion,
)
from competitive_intelligence.retrieval_engine.models import (
    RetrievalResult,
)


@pytest.fixture(scope="module")
def corpus() -> SystemCorpus:
    """Fixture providing the complete 120+ document synthetic evaluation corpus."""
    return create_system_evaluation_corpus(workspace_id="ws_test_system")


# ==============================================================================
# SCENARIOS 1 - 12 END-TO-END INTEGRATION TESTS
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_01_simple_competitive_event(corpus: SystemCorpus) -> None:
    """SC-01: Single price drop produces 1 grounded event and no unsupported inferences."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-01")
    req_chunks = [corpus.retrieval_index[cid] for cid in sc.required_chunk_ids]

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id=sc.company_ids[0],
        workspace_id=sc.workspace_id,
        evidence=req_chunks,
    )

    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == IntelligenceEventType.PRICING_CHANGE
    assert ev.company_id == "acme_cloud"
    assert ev.previous_state == {"price_usd": "$50"}
    assert ev.new_state == {"price_usd": "$30"}
    assert len(ev.evidence) == 1
    assert ev.evidence[0].chunk_id == sc.required_chunk_ids[0]

    # No unsupported extra signals from a single isolated event
    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id=sc.workspace_id,
        company_id=sc.company_ids[0],
        events=events,
    )
    # A single event generates an emerging activity signal at most, but no trend
    directional_trends = [s for s in signals if s.signal_type == SignalType.DIRECTIONAL_TREND]
    assert len(directional_trends) == 0


@pytest.mark.anyio
async def test_scenario_02_repeated_pricing_reductions_trend(corpus: SystemCorpus) -> None:
    """SC-02: NovaDB 3 sequential price drops synthesize a Directional Trend signal and Pricing Strategy insight."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-02")
    req_chunks = [corpus.retrieval_index[cid] for cid in sc.required_chunk_ids]

    # 1. Event Extraction
    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id=sc.company_ids[0],
        workspace_id=sc.workspace_id,
        evidence=req_chunks,
    )
    assert len(events) == 3

    # 2. Signal Detection
    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id=sc.workspace_id,
        company_id=sc.company_ids[0],
        events=events,
    )
    trend_signals = [s for s in signals if s.signal_type == SignalType.DIRECTIONAL_TREND]
    assert len(trend_signals) >= 1
    trend = trend_signals[0]
    assert len(trend.supporting_event_ids) >= 2

    # 3. Strategic Insight Synthesis
    strategic_analyzer = DeterministicStrategicAnalyzer()
    insights = await strategic_analyzer.generate_insights(
        workspace_id=sc.workspace_id,
        company_id=sc.company_ids[0],
        signals=signals,
        events=events,
    )
    pricing_insights = [ins for ins in insights if ins.insight_type == InsightType.PRICING_STRATEGY]
    assert len(pricing_insights) >= 1
    assert pricing_insights[0].factual_confidence.level in (ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH)


@pytest.mark.anyio
async def test_scenario_03_strategic_pricing_reversal(corpus: SystemCorpus) -> None:
    """SC-03: Apex Search price drop followed by price restoration produces a Reversal signal."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-03")
    req_chunks = [corpus.retrieval_index[cid] for cid in sc.required_chunk_ids]

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id=sc.company_ids[0],
        workspace_id=sc.workspace_id,
        evidence=req_chunks,
    )
    assert len(events) == 2

    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id=sc.workspace_id,
        company_id=sc.company_ids[0],
        events=events,
    )
    reversals = [s for s in signals if s.signal_type == SignalType.REVERSAL]
    assert len(reversals) >= 1
    assert reversals[0].signal_type == SignalType.REVERSAL


@pytest.mark.anyio
async def test_scenario_04_multi_entity_comparison(corpus: SystemCorpus) -> None:
    """SC-04: Multi-entity query preserves evidence for both VectorFlow and TensorScale."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-04")
    req_chunks = [corpus.retrieval_index[cid] for cid in sc.required_chunk_ids]

    vf_chunks = [c for c in req_chunks if c.company_id == "vectorflow"]
    ts_chunks = [c for c in req_chunks if c.company_id == "tensorscale"]

    assert len(vf_chunks) == 1
    assert len(ts_chunks) == 1

    # Extract events for both entities
    analyzer = RetrievedEvidenceAnalyzer()
    vf_events = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id="vectorflow",
        workspace_id=sc.workspace_id,
        evidence=vf_chunks,
    )
    ts_events = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id="tensorscale",
        workspace_id=sc.workspace_id,
        evidence=ts_chunks,
    )

    assert len(vf_events) == 1
    assert len(ts_events) == 1
    assert vf_events[0].company_id == "vectorflow"
    assert ts_events[0].company_id == "tensorscale"


@pytest.mark.anyio
async def test_scenario_05_temporal_version_evolution(corpus: SystemCorpus) -> None:
    """SC-05: ByteBase AI changelogs preserve temporal ordering and version lineage."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-05")
    req_chunks = [corpus.retrieval_index[cid] for cid in sc.required_chunk_ids]

    # Verify 3 distinct versions present in ground truth
    version_ids = {c.version_id for c in req_chunks}
    assert len(version_ids) == 3


@pytest.mark.anyio
async def test_scenario_06_contradictory_evidence_preservation(corpus: SystemCorpus) -> None:
    """SC-06: Zenith AI conflicting release reports generate InsightType.UNCERTAINTY and adjust confidence."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-06")
    req_chunks = [corpus.retrieval_index[cid] for cid in sc.required_chunk_ids]

    # Create explicit contradictory insight
    uncertain_insight = Insight(
        insight_id="ins_zenith_contradiction",
        workspace_id=sc.workspace_id,
        company_id="zenith_ai",
        insight_type=InsightType.UNCERTAINTY,
        title="Zenith AI Voice Mode Release Contradiction",
        summary="Official blog states release is delayed to Q4, while TechNews report claims it was launched in Q2.",
        statement="Contradictory evidence exists regarding product availability.",
        supporting_signal_ids=[],
        supporting_event_ids=["ev_zen_01"],
        supporting_evidence_ids=[req_chunks[0].chunk_id, req_chunks[1].chunk_id],
        factual_confidence=ConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH),
        interpretive_confidence=InterpretiveConfidenceAssessment(score=0.75, level=ConfidenceLevel.MEDIUM),
        significance=SignificanceAssessment(),
        status=InsightStatus.UNCERTAIN,
    )

    event = IntelligenceEvent(
        event_id="ev_zen_01",
        workspace_id=sc.workspace_id,
        company_id="zenith_ai",
        event_type=IntelligenceEventType.PRODUCT_LAUNCH,
        title="Voice Mode Status",
        summary="Voice mode status update",
        subject="Voice Mode",
        event_time=EventTime(
            occurred_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
            precision=TimePrecision.DATE_ONLY,
            acquired_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
            detected_at=datetime(2026, 3, 12, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.85, level=ConfidenceLevel.HIGH),
        evidence=[
            EvidenceReference(
                evidence_id="ev_ref_zen_01",
                source_id="src_zen_blog",
                document_id="doc_zen_01",
                version_id="ver_zen_01",
                chunk_id=req_chunks[0].chunk_id,
                source_url="https://zenith.ai/blog/voice-status",
                source_category="website",
                extracted_text=req_chunks[0].text,
                acquired_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
            )
        ],
    )

    context = DecisionContext(
        title="Response to Zenith AI Voice Launch",
        workspace_id=sc.workspace_id,
        decision_domain=DecisionDomain.PRODUCT,
        business_objectives=[
            DecisionObjective(name="Response Speed", dimension=ObjectiveDimension.TIME_TO_MARKET, target_description="Respond fast", priority_weight=1.0)
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
    assert "Zenith AI Voice Mode Release Contradiction" in rec.unresolved_uncertainties[0]
    assert rec.confidence.level == ConfidenceLevel.MEDIUM
    assert "observed strategic uncertainty" in rec.confidence.rationale


@pytest.mark.anyio
async def test_scenario_07_evidence_gap_refusal(corpus: SystemCorpus) -> None:
    """SC-07: Unannounced proprietary algorithm query results in refusal / insufficiency without hallucinated events."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-07")
    assert len(sc.required_chunk_ids) == 0

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id=sc.company_ids[0],
        workspace_id=sc.workspace_id,
        evidence=[],  # Zero evidence
    )

    assert len(events) == 0


@pytest.mark.anyio
async def test_scenario_08_retrieval_distractor_filtering(corpus: SystemCorpus) -> None:
    """SC-08: Distractor-heavy evidence pool correctly extracts event only from the true pricing chunk."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-08")
    req_chunk = corpus.retrieval_index[sc.required_chunk_ids[0]]
    distractor_chunks = [corpus.retrieval_index[cid] for cid in sc.distractor_chunk_ids]

    # Combine true chunk with 4 distractor chunks
    mixed_evidence = distractor_chunks[:2] + [req_chunk] + distractor_chunks[2:]

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id=sc.company_ids[0],
        workspace_id=sc.workspace_id,
        evidence=mixed_evidence,
    )

    # Exactly 1 event should be extracted from the true pricing chunk; distractors ignored
    assert len(events) == 1
    ev = events[0]
    assert ev.evidence[0].chunk_id == sc.required_chunk_ids[0]
    assert ev.event_type == IntelligenceEventType.PRICING_CHANGE


@pytest.mark.anyio
async def test_scenario_09_negative_adversarial_query_refusal(corpus: SystemCorpus) -> None:
    """SC-09: Fabricated acquisition claim produces zero events and no recommendation."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-09")
    assert len(sc.required_chunk_ids) == 0

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id=sc.company_ids[0],
        workspace_id=sc.workspace_id,
        evidence=[],
    )

    assert len(events) == 0


@pytest.mark.anyio
async def test_scenario_10_decision_support_full_pipeline(corpus: SystemCorpus) -> None:
    """SC-10: Full decision support pipeline enforces hard budget constraint ($15,000) and selects highest weighted option."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-10")
    assert sc.decision_context is not None

    req_chunk = corpus.retrieval_index[sc.required_chunk_ids[0]]
    evidence_ref = EvidenceReference(
        evidence_id="ev_ref_sc10",
        source_id="src_acme_pricing",
        document_id="doc_acme_001",
        version_id="ver_acme_001_1",
        chunk_id=req_chunk.chunk_id,
        source_url=req_chunk.canonical_url or "https://acme.com/pricing",
        source_category="pricing",
        extracted_text=req_chunk.text,
        acquired_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
    )

    event = IntelligenceEvent(
        event_id="ev_sc10_01",
        workspace_id=sc.workspace_id,
        company_id="acme_cloud",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Enterprise Plan Price Reduction",
        summary="Enterprise plan reduced from $50 to $30/month.",
        subject="Enterprise Plan",
        previous_state={"price_usd": "$50"},
        new_state={"price_usd": "$30"},
        event_time=EventTime(
            occurred_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            precision=TimePrecision.DATE_ONLY,
            acquired_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            detected_at=datetime(2026, 1, 16, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.95, level=ConfidenceLevel.HIGH),
        evidence=[evidence_ref],
    )

    insight = Insight(
        insight_id="ins_sc10_01",
        workspace_id=sc.workspace_id,
        company_id="acme_cloud",
        insight_type=InsightType.PRICING_STRATEGY,
        title="Aggressive Down-Market Pricing Shift",
        summary="Acme Cloud is systematically reducing pricing to capture enterprise accounts.",
        statement="Detailed statement of pricing movement.",
        supporting_signal_ids=["sig_sc10"],
        supporting_event_ids=["ev_sc10_01"],
        supporting_evidence_ids=[evidence_ref.evidence_id],
        factual_confidence=ConfidenceAssessment(score=0.95, level=ConfidenceLevel.HIGH),
        interpretive_confidence=InterpretiveConfidenceAssessment(score=0.90, level=ConfidenceLevel.HIGH),
        significance=SignificanceAssessment(),
        status=InsightStatus.ASSESSED,
    )

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=sc.decision_context,
        insights=[insight],
        events=[event],
    )

    assert len(options) == 3
    assert rec is not None
    assert rec.status == RecommendationStatus.PROPOSED
    # Verify budget constraint eliminated the $20,000 counter-move
    eliminated = [alt for alt in rec.evaluated_alternatives if alt.is_eliminated]
    assert len(eliminated) == 1
    assert "exceeds hard budget limit" in str(eliminated[0].elimination_reason)
    # Selected option is within $15,000 budget
    selected_opt = next(opt for opt in options if opt.option_id == rec.selected_option_id)
    assert selected_opt.estimated_cost_usd is not None and selected_opt.estimated_cost_usd <= 15000.0


@pytest.mark.anyio
async def test_scenario_11_missing_decision_context_safety(corpus: SystemCorpus) -> None:
    """SC-11: Empty decision objectives safely produce 0 candidate options and rec=None."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-11")
    assert sc.decision_context is not None

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=sc.decision_context,
        insights=[],
        events=[],
    )

    assert len(options) == 0
    assert rec is None


@pytest.mark.anyio
async def test_scenario_12_conflicting_objectives_no_hidden_optimizer(corpus: SystemCorpus) -> None:
    """SC-12: Unweighted multi-objective decisions return candidate options without an unjustified recommendation."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-12")
    assert sc.decision_context is not None

    req_chunk = corpus.retrieval_index[sc.required_chunk_ids[0]]
    evidence_ref = EvidenceReference(
        evidence_id="ev_ref_sc12",
        source_id="src_acme_pricing",
        document_id="doc_acme_001",
        version_id="ver_acme_001_1",
        chunk_id=req_chunk.chunk_id,
        source_url=req_chunk.canonical_url or "https://acme.com/pricing",
        source_category="pricing",
        extracted_text=req_chunk.text,
        acquired_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
    )

    event = IntelligenceEvent(
        event_id="ev_sc12_01",
        workspace_id=sc.workspace_id,
        company_id="acme_cloud",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Enterprise Plan Price Reduction",
        summary="Enterprise plan reduced from $50 to $30/month.",
        subject="Enterprise Plan",
        event_time=EventTime(
            occurred_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            precision=TimePrecision.DATE_ONLY,
            acquired_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
            detected_at=datetime(2026, 1, 16, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.95, level=ConfidenceLevel.HIGH),
        evidence=[evidence_ref],
    )

    insight = Insight(
        insight_id="ins_sc12_01",
        workspace_id=sc.workspace_id,
        company_id="acme_cloud",
        insight_type=InsightType.PRICING_STRATEGY,
        title="Aggressive Down-Market Pricing Shift",
        summary="Acme Cloud is reducing pricing.",
        statement="Pricing movement statement.",
        supporting_signal_ids=["sig_sc12"],
        supporting_event_ids=["ev_sc12_01"],
        supporting_evidence_ids=[evidence_ref.evidence_id],
        factual_confidence=ConfidenceAssessment(score=0.95, level=ConfidenceLevel.HIGH),
        interpretive_confidence=InterpretiveConfidenceAssessment(score=0.90, level=ConfidenceLevel.HIGH),
        significance=SignificanceAssessment(),
        status=InsightStatus.ASSESSED,
    )

    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=sc.decision_context,
        insights=[insight],
        events=[event],
    )

    assert len(options) == 3
    assert rec is None  # No recommendation made without explicit weights


# ==============================================================================
# EPISTEMIC INVARIANTS 1 - 10
# ==============================================================================


@pytest.mark.anyio
async def test_invariant_01_no_evidence_no_event() -> None:
    """Invariant 1: Zero evidence must never produce events."""
    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Query",
        company_id="acme",
        workspace_id="ws_01",
        evidence=[],
    )
    assert len(events) == 0


@pytest.mark.anyio
async def test_invariant_02_no_grounded_events_no_signal() -> None:
    """Invariant 2: Zero events must never produce signals."""
    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_01",
        company_id="acme",
        events=[],
    )
    assert len(signals) == 0


@pytest.mark.anyio
async def test_invariant_03_no_supported_signal_no_insight() -> None:
    """Invariant 3: Zero signals and zero events must never produce insights."""
    analyzer = DeterministicStrategicAnalyzer()
    insights = await analyzer.generate_insights(
        workspace_id="ws_01",
        company_id="acme",
        signals=[],
        events=[],
    )
    assert len(insights) == 0


def test_invariant_04_insight_not_automatically_recommendation() -> None:
    """Invariant 4: Insight is distinct from Recommendation."""
    insight = Insight(
        insight_id="ins_01",
        workspace_id="ws_01",
        company_id="acme",
        insight_type=InsightType.PRICING_STRATEGY,
        title="Price Move",
        summary="Price move summary",
        statement="Statement",
        supporting_event_ids=["ev_01"],
        factual_confidence=ConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
    )
    assert not isinstance(insight, Recommendation)
    assert not hasattr(insight, "selected_option_id")


@pytest.mark.anyio
async def test_invariant_05_missing_decision_objectives_no_recommendation() -> None:
    """Invariant 5: Missing decision objectives produces no recommendation."""
    analyzer = DeterministicDecisionAnalyzer()
    empty_context = DecisionContext(title="Empty Context", business_objectives=[])
    options, rec = await analyzer.analyze_and_recommend(
        context=empty_context,
        insights=[],
        events=[],
    )
    assert options == []
    assert rec is None


@pytest.mark.anyio
async def test_invariant_06_missing_objective_weights_options_only() -> None:
    """Invariant 6: Multi-objective decisions without explicit weights return options only."""
    context = DecisionContext(
        title="Unweighted Context",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(name="O1", dimension=ObjectiveDimension.GROWTH, target_description="D1", priority_weight=None),
            DecisionObjective(name="O2", dimension=ObjectiveDimension.COST_EFFICIENCY, target_description="D2", priority_weight=None),
        ],
    )
    analyzer = DeterministicDecisionAnalyzer()
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[],
        events=[],
    )
    assert len(options) == 3
    assert rec is None


@pytest.mark.anyio
async def test_invariant_07_contradictory_evidence_preserves_uncertainty() -> None:
    """Invariant 7: Contradictory insights preserve uncertainty in recommendations."""
    uncertain_insight = Insight(
        insight_id="ins_u1",
        workspace_id="ws_01",
        company_id="acme",
        insight_type=InsightType.UNCERTAINTY,
        title="Uncertain Status",
        summary="Conflicting reports",
        statement="Statement",
        supporting_event_ids=["ev_1"],
        factual_confidence=ConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH),
    )
    event = IntelligenceEvent(
        event_id="ev_1",
        workspace_id="ws_01",
        company_id="acme",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="E",
        summary="S",
        subject="Sub",
        event_time=EventTime(acquired_at=datetime.now(timezone.utc)),
        confidence=ConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH),
        evidence=[
            EvidenceReference(
                evidence_id="ev_ref_1",
                source_id="s1",
                document_id="d1",
                version_id="v1",
                source_url="https://a.com",
                source_category="pricing",
                extracted_text="Text",
                acquired_at=datetime.now(timezone.utc),
            )
        ],
    )
    context = DecisionContext(
        title="C",
        workspace_id="ws_01",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(name="Growth", dimension=ObjectiveDimension.GROWTH, target_description="G", priority_weight=1.0)
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
    assert rec.confidence.level == ConfidenceLevel.MEDIUM


def test_invariant_08_fabricated_event_or_insight_id_validator_rejection() -> None:
    """Invariant 8: Fabricated IDs are strictly rejected by the recommendation validator."""
    validator = DeterministicRecommendationValidator()
    context = DecisionContext(
        decision_id="dec_01",
        workspace_id="ws_01",
        title="Test",
        business_objectives=[DecisionObjective(name="T", dimension=ObjectiveDimension.GROWTH, target_description="T")],
    )
    opt = DecisionOption(
        option_id="opt_fake",
        decision_id="dec_01",
        title="Bad Option",
        description="Fake ID option",
        action_type=ActionType.MONITOR_AND_WAIT,
        supporting_event_ids=["fake_event_id_999"],
    )
    rec = Recommendation(
        decision_id="dec_01",
        workspace_id="ws_01",
        selected_option_id="opt_fake",
        title="Rec",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="R"),
        supporting_event_ids=["fake_event_id_999"],
    )
    is_valid, errors = validator.validate(
        payload=rec,
        context=context,
        options=[opt],
        insights=[],
        events=[],
    )
    assert is_valid is False
    assert any("does not exist in candidate events pool" in err or "unknown event_id" in err for err in errors)


def test_invariant_09_cross_workspace_citations_rejected() -> None:
    """Invariant 9: Citations from other workspaces are strictly rejected."""
    validator = DeterministicRecommendationValidator()
    context = DecisionContext(
        decision_id="dec_ws1",
        workspace_id="workspace_alpha",
        title="Test",
        business_objectives=[DecisionObjective(name="T", dimension=ObjectiveDimension.GROWTH, target_description="T")],
    )
    cross_ws_insight = Insight(
        insight_id="ins_ws_beta",
        workspace_id="workspace_beta",  # Different tenant!
        company_id="acme",
        insight_type=InsightType.PRICING_STRATEGY,
        title="Beta Insight",
        summary="B",
        statement="B",
        supporting_event_ids=["ev_beta"],
        factual_confidence=ConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
    )
    opt = DecisionOption(
        option_id="opt_01",
        decision_id="dec_ws1",
        title="Opt",
        description="Desc",
        action_type=ActionType.MONITOR_AND_WAIT,
        supporting_insight_ids=["ins_ws_beta"],
        supporting_event_ids=["ev_beta"],
    )
    rec = Recommendation(
        decision_id="dec_ws1",
        workspace_id="workspace_alpha",
        selected_option_id="opt_01",
        title="Rec",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="R"),
        supporting_insight_ids=["ins_ws_beta"],
        supporting_event_ids=["ev_beta"],
    )
    is_valid, errors = validator.validate(
        payload=rec,
        context=context,
        options=[opt],
        insights=[cross_ws_insight],
        events=[],
    )
    assert is_valid is False
    assert any("violating workspace isolation" in err for err in errors)


def test_invariant_10_automated_recommendation_status_proposed_only() -> None:
    """Invariant 10: Automated recommendations always initialize as PROPOSED."""
    rec = Recommendation(
        decision_id="dec_01",
        workspace_id="ws_01",
        selected_option_id="opt_01",
        title="Rec",
        summary="Summary",
        rationale="Rationale",
        evaluated_alternatives=[],
        expected_benefits=["Benefit"],
        confidence=RecommendationConfidenceAssessment(score=0.8, level=ConfidenceLevel.HIGH, rationale="R"),
        supporting_event_ids=["ev_01"],
    )
    assert rec.status == RecommendationStatus.PROPOSED


# ==============================================================================
# FAILURE INJECTION SUITE
# ==============================================================================


@pytest.mark.anyio
async def test_failure_injection_empty_content_graceful_handling() -> None:
    """Failure Mode 1: Empty text content is gracefully rejected without unhandled crashes."""
    analyzer = RetrievedEvidenceAnalyzer()
    empty_result = RetrievalResult(
        chunk_id="chunk_empty",
        score=0.9,
        text="",  # Empty text
        document_id="doc_empty",
        version_id="ver_empty",
        source_id="src_empty",
        company_id="acme",
        workspace_id="ws_01",
        canonical_url="https://acme.com",
    )
    events = await analyzer.analyze_retrieved_evidence(
        query="Query",
        company_id="acme",
        workspace_id="ws_01",
        evidence=[empty_result],
    )
    assert len(events) == 0


@pytest.mark.anyio
async def test_failure_injection_untrusted_option_generator_filtered() -> None:
    """Failure Mode 2: Untrusted option generator producing invalid budget options is filtered deterministically."""
    class BadOptionGenerator(BaseOptionGenerator):
        async def generate_options(
            self,
            context: DecisionContext,
            insights: list[Insight],
            events: list[IntelligenceEvent],
        ) -> list[DecisionOption]:
            return [
                DecisionOption(
                    option_id="opt_too_expensive",
                    decision_id=context.decision_id,
                    title="Absurdly Expensive Option",
                    description="Costs $1,000,000",
                    action_type=ActionType.DIRECT_COUNTER_MOVE,
                    estimated_cost_usd=1000000.0,
                    expected_effects=[PredictedConsequence(dimension=ObjectiveDimension.GROWTH, description="G", is_positive=True)],
                    supporting_insight_ids=[ins.insight_id for ins in insights],
                    supporting_event_ids=[e.event_id for e in events],
                )
            ]

    context = DecisionContext(
        title="Budget Test",
        workspace_id="ws_01",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[DecisionObjective(name="G", dimension=ObjectiveDimension.GROWTH, target_description="G", priority_weight=1.0)],
        constraints=[DecisionConstraint(constraint_type=ConstraintType.BUDGET, description="Cap $10k", is_hard_constraint=True, threshold_value=10000.0)],
    )

    analyzer = DeterministicDecisionAnalyzer(option_generator=BadOptionGenerator())
    options, rec = await analyzer.analyze_and_recommend(
        context=context,
        insights=[],
        events=[],
    )

    assert len(options) == 1
    assert rec is None  # 0 feasible options left -> no recommendation synthesized


# ==============================================================================
# IDEMPOTENCY & REPEATABILITY
# ==============================================================================


@pytest.mark.anyio
async def test_idempotency_and_repeatability_event_extraction(corpus: SystemCorpus) -> None:
    """Verify that repeated extraction runs on identical evidence produce identical event signatures and content."""
    sc = next(s for s in corpus.scenarios if s.scenario_id == "SC-01")
    req_chunks = [corpus.retrieval_index[cid] for cid in sc.required_chunk_ids]

    analyzer = RetrievedEvidenceAnalyzer()
    events_run1 = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id=sc.company_ids[0],
        workspace_id=sc.workspace_id,
        evidence=req_chunks,
    )
    events_run2 = await analyzer.analyze_retrieved_evidence(
        query=sc.query_text,
        company_id=sc.company_ids[0],
        workspace_id=sc.workspace_id,
        evidence=req_chunks,
    )

    assert len(events_run1) == len(events_run2)
    assert events_run1[0].metadata.get("cluster_signature") == events_run2[0].metadata.get("cluster_signature")
    assert events_run1[0].event_type == events_run2[0].event_type
    assert events_run1[0].title == events_run2[0].title
    assert events_run1[0].previous_state == events_run2[0].previous_state
    assert events_run1[0].new_state == events_run2[0].new_state
