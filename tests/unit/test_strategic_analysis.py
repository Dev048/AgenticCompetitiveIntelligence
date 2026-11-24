"""Unit tests for Strategic Analysis and Insight Generation (Phase 4)."""

from datetime import datetime, timezone
from typing import Any
import pytest

from competitive_intelligence.intelligence import (
    BaseStrategicAnalyzer,
    ConfidenceAssessment,
    ConfidenceLevel,
    DeterministicInsightValidator,
    DeterministicStrategicAnalyzer,
    EvidenceReference,
    EvidenceRole,
    InferredSignal,
    Insight,
    InsightStatus,
    InsightType,
    IntelligenceEvent,
    IntelligenceEventType,
    RawInsightPayload,
    SignificanceStatus,
    SignalConfidenceAssessment,
    SignalStatus,
    SignalType,
    SourcePerspective,
    TimePrecision,
)
from competitive_intelligence.intelligence.models import EventTime


def _create_test_event(
    event_id: str,
    company_id: str = "acme_corp",
    event_type: IntelligenceEventType = IntelligenceEventType.PRICING_CHANGE,
    subject: str = "Enterprise Plan",
    title: str = "Pricing Update",
    summary: str = "Changed pricing.",
    previous_state: dict[str, Any] | str | None = None,
    new_state: dict[str, Any] | str | None = None,
    occurred_at: datetime | None = None,
    confidence_score: float = 0.90,
) -> IntelligenceEvent:
    """Helper to construct valid grounded IntelligenceEvent instances."""
    ev_time = EventTime(
        occurred_at=occurred_at,
        precision=TimePrecision.DATE_ONLY if occurred_at else TimePrecision.UNKNOWN,
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        detected_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
    )
    conf = ConfidenceAssessment(
        score=confidence_score,
        level=ConfidenceLevel.HIGH if confidence_score >= 0.8 else ConfidenceLevel.LOW,
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
        workspace_id="ws_default",
        company_id=company_id,
        event_type=event_type,
        title=title,
        summary=summary,
        subject=subject,
        previous_state=previous_state,
        new_state=new_state,
        event_time=ev_time,
        confidence=conf,
        evidence=[evidence_item],
        primary_perspective=SourcePerspective.FIRST_PARTY,
    )


def _create_test_signal(
    signal_id: str,
    company_id: str = "acme_corp",
    signal_type: SignalType = SignalType.REVERSAL,
    subject: str = "Enterprise Plan",
    title: str = "Pricing Reversal",
    summary: str = "Pricing reverted.",
    supporting_event_ids: list[str] | None = None,
    confidence_score: float = 0.85,
    status: SignalStatus = SignalStatus.RESOLVED,
) -> InferredSignal:
    """Helper to construct valid InferredSignal instances."""
    ev_ids = supporting_event_ids or ["e1"]
    return InferredSignal(
        signal_id=signal_id,
        workspace_id="ws_default",
        company_id=company_id,
        signal_type=signal_type,
        subject=subject,
        title=title,
        summary=summary,
        supporting_event_ids=ev_ids,
        supporting_evidence_ids=[f"ev_ref_{eid}" for eid in ev_ids],
        confidence=SignalConfidenceAssessment(
            score=confidence_score,
            level=ConfidenceLevel.HIGH if confidence_score >= 0.8 else ConfidenceLevel.LOW,
            rationale="Grounded signal.",
        ),
        status=status,
    )


@pytest.fixture
def analyzer() -> DeterministicStrategicAnalyzer:
    return DeterministicStrategicAnalyzer()


@pytest.mark.anyio
async def test_empty_input_produces_zero_insights(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify that empty inputs produce 0 Insights."""
    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[],
        events=[],
    )
    assert insights == []


@pytest.mark.anyio
async def test_pricing_pattern_insight_generation(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify that a pricing reversal signal produces a PRICING_STRATEGY insight without mind-reading."""
    e1 = _create_test_event("e_p1", subject="Enterprise Tier", occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    e2 = _create_test_event("e_p2", subject="Enterprise Tier", occurred_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
    sig = _create_test_signal(
        "sig_rev",
        signal_type=SignalType.REVERSAL,
        subject="Enterprise Tier",
        supporting_event_ids=["e_p1", "e_p2"],
    )

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig],
        events=[e1, e2],
    )

    assert len(insights) == 1
    ins = insights[0]
    assert ins.insight_type == InsightType.PRICING_STRATEGY
    assert ins.status == InsightStatus.ASSESSED
    assert "repeated adjustments" in ins.statement
    assert ins.supporting_signal_ids == ["sig_rev"]
    assert ins.supporting_event_ids == ["e_p1", "e_p2"]


@pytest.mark.anyio
async def test_product_strategy_acceleration_insight(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify that release acceleration produces a PRODUCT_STRATEGY insight."""
    e1 = _create_test_event("e_r1", event_type=IntelligenceEventType.RELEASE, subject="Vector Core")
    e2 = _create_test_event("e_r2", event_type=IntelligenceEventType.RELEASE, subject="Vector Core")
    sig = _create_test_signal(
        "sig_acc",
        signal_type=SignalType.ACCELERATION,
        subject="Vector Core",
        supporting_event_ids=["e_r1", "e_r2"],
        status=SignalStatus.ACTIVE,
    )

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig],
        events=[e1, e2],
    )

    assert len(insights) == 1
    ins = insights[0]
    assert ins.insight_type == InsightType.PRODUCT_STRATEGY
    assert "Product development cadence" in ins.statement


@pytest.mark.anyio
async def test_hiring_pattern_insight_generation(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify sustained executive hiring produces a HIRING_DIRECTION insight."""
    e1 = _create_test_event("e_h1", event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT, subject="Executive Leadership")
    e2 = _create_test_event("e_h2", event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT, subject="Executive Leadership")
    sig = _create_test_signal(
        "sig_hire",
        signal_type=SignalType.SUSTAINED_ACTIVITY,
        subject="Executive Leadership",
        supporting_event_ids=["e_h1", "e_h2"],
    )

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig],
        events=[e1, e2],
    )

    assert len(insights) == 1
    ins = insights[0]
    assert ins.insight_type == InsightType.HIRING_DIRECTION
    assert "leadership additions" in ins.statement


@pytest.mark.anyio
async def test_acquisition_pattern_insight_generation(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify notable acquisition signal produces a COMPETITIVE_MOVEMENT insight."""
    e_acq = _create_test_event(
        "e_acq",
        event_type=IntelligenceEventType.ACQUISITION,
        subject="VectorDB Inc",
        title="Acquisition: Acme Acquired VectorDB Inc",
    )
    sig = _create_test_signal(
        "sig_acq",
        signal_type=SignalType.EMERGING_ACTIVITY,
        subject="VectorDB Inc",
        supporting_event_ids=["e_acq"],
        status=SignalStatus.DETECTED,
    )

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig],
        events=[e_acq],
    )

    assert len(insights) == 1
    ins = insights[0]
    assert ins.insight_type == InsightType.COMPETITIVE_MOVEMENT
    assert "Acquisition activity completed" in ins.statement


@pytest.mark.anyio
async def test_contradictory_signals_produce_uncertainty_insight(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify contradictory signal produces an UNCERTAINTY insight with low interpretive confidence."""
    e1 = _create_test_event("e_d1", subject="Voice Feature")
    e2 = _create_test_event("e_d2", subject="Voice Feature")
    sig = _create_test_signal(
        "sig_conflict",
        signal_type=SignalType.CONTRADICTION,
        subject="Voice Feature",
        supporting_event_ids=["e_d1", "e_d2"],
        status=SignalStatus.UNCERTAIN,
    )

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig],
        events=[e1, e2],
    )

    assert len(insights) == 1
    ins = insights[0]
    assert ins.insight_type == InsightType.UNCERTAINTY
    assert ins.status == InsightStatus.UNCERTAIN
    assert ins.interpretive_confidence.score <= 0.60
    assert ins.interpretive_confidence.level == ConfidenceLevel.LOW


@pytest.mark.anyio
async def test_transitive_provenance_integrity(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify transitive provenance: Insight -> supporting_signal_ids, supporting_event_ids, supporting_evidence_ids."""
    e1 = _create_test_event("e_prov_1", subject="Enterprise Tier")
    e2 = _create_test_event("e_prov_2", subject="Enterprise Tier")
    sig = _create_test_signal(
        "sig_prov",
        signal_type=SignalType.REVERSAL,
        subject="Enterprise Tier",
        supporting_event_ids=["e_prov_1", "e_prov_2"],
    )

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig],
        events=[e1, e2],
    )

    assert len(insights) == 1
    ins = insights[0]
    assert ins.supporting_signal_ids == ["sig_prov"]
    assert ins.supporting_event_ids == ["e_prov_1", "e_prov_2"]
    assert len(ins.supporting_evidence_ids) == 2
    assert ins.supporting_evidence_ids[0] == e1.evidence[0].evidence_id
    assert ins.supporting_evidence_ids[1] == e2.evidence[0].evidence_id


@pytest.mark.anyio
async def test_factual_vs_interpretive_confidence_separation(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify that factual confidence and interpretive confidence remain independently maintained."""
    # Low factual confidence events
    e1 = _create_test_event("e_low_1", subject="Enterprise Tier", confidence_score=0.50)
    e2 = _create_test_event("e_low_2", subject="Enterprise Tier", confidence_score=0.55)
    sig = _create_test_signal(
        "sig_rev_low",
        signal_type=SignalType.REVERSAL,
        subject="Enterprise Tier",
        supporting_event_ids=["e_low_1", "e_low_2"],
    )

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig],
        events=[e1, e2],
    )

    assert len(insights) == 1
    ins = insights[0]
    # Factual confidence reflects low event confidence
    assert ins.factual_confidence.score <= 0.60
    assert ins.factual_confidence.level == ConfidenceLevel.LOW
    # Interpretive confidence reflects pattern deduction
    assert ins.interpretive_confidence.score >= 0.80
    assert ins.interpretive_confidence.level == ConfidenceLevel.HIGH


@pytest.mark.anyio
async def test_strategic_significance_remains_unassessed(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify that strategic significance remains UNASSESSED in Phase 4."""
    e1 = _create_test_event("e1", subject="Enterprise Tier")
    sig = _create_test_signal("sig1", subject="Enterprise Tier", supporting_event_ids=["e1"])

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig],
        events=[e1],
    )

    assert len(insights) == 1
    ins = insights[0]
    assert ins.significance.status == SignificanceStatus.UNASSESSED
    assert ins.significance.score is None


@pytest.mark.anyio
async def test_temporal_scope_bounds_accuracy(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify TemporalScope bounds accurately reflect supporting event timestamps."""
    t_start = datetime(2026, 1, 15, tzinfo=timezone.utc)
    t_end = datetime(2026, 6, 20, tzinfo=timezone.utc)
    e1 = _create_test_event("e_t1", subject="Enterprise Tier", occurred_at=t_start)
    e2 = _create_test_event("e_t2", subject="Enterprise Tier", occurred_at=t_end)
    sig = _create_test_signal("sig_t", subject="Enterprise Tier", supporting_event_ids=["e_t1", "e_t2"])

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig],
        events=[e1, e2],
    )

    assert len(insights) == 1
    ins = insights[0]
    assert ins.temporal_scope.start_date == t_start
    assert ins.temporal_scope.end_date == t_end


def test_validator_rejects_ungrounded_signal_ids() -> None:
    """Verify validator rejects candidate payloads referencing non-existent signal IDs."""
    validator = DeterministicInsightValidator()
    e1 = _create_test_event("e1")
    sig1 = _create_test_signal("sig1", supporting_event_ids=["e1"])

    invalid_payload = RawInsightPayload(
        insight_type=InsightType.PRICING_STRATEGY,
        title="Invalid Insight",
        summary="Summary",
        statement="Statement",
        supporting_signal_ids=["non_existent_sig"],
        supporting_event_ids=["e1"],
    )

    is_valid, errors = validator.validate(invalid_payload, signals=[sig1], events=[e1])
    assert is_valid is False
    assert any("does not exist in the candidate signals pool" in err for err in errors)


def test_validator_rejects_missing_event_provenance() -> None:
    """Verify validator rejects candidate payloads referencing non-existent event IDs."""
    validator = DeterministicInsightValidator()
    e1 = _create_test_event("e1")
    sig1 = _create_test_signal("sig1", supporting_event_ids=["e1"])

    invalid_payload = RawInsightPayload(
        insight_type=InsightType.PRICING_STRATEGY,
        title="Invalid Insight",
        summary="Summary",
        statement="Statement",
        supporting_signal_ids=["sig1"],
        supporting_event_ids=["non_existent_event"],
    )

    is_valid, errors = validator.validate(invalid_payload, signals=[sig1], events=[e1])
    assert is_valid is False
    assert any("does not exist in the candidate events pool" in err for err in errors)


@pytest.mark.anyio
async def test_unrelated_signals_separation(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify disjoint signals on different subjects generate distinct insights."""
    e_price = _create_test_event("e_price", subject="Enterprise Tier")
    e_hire = _create_test_event("e_hire", event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT, subject="Executive Leadership")

    sig_price = _create_test_signal("sig_p", signal_type=SignalType.REVERSAL, subject="Enterprise Tier", supporting_event_ids=["e_price"])
    sig_hire = _create_test_signal("sig_h", signal_type=SignalType.SUSTAINED_ACTIVITY, subject="Executive Leadership", supporting_event_ids=["e_hire"])

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig_price, sig_hire],
        events=[e_price, e_hire],
    )

    assert len(insights) == 2
    types = {ins.insight_type for ins in insights}
    assert InsightType.PRICING_STRATEGY in types
    assert InsightType.HIRING_DIRECTION in types


@pytest.mark.anyio
async def test_weak_evidence_produces_zero_insights(
    analyzer: DeterministicStrategicAnalyzer,
) -> None:
    """Verify that unclassified or weak signals produce 0 insights."""
    e_misc = _create_test_event("e_misc", event_type=IntelligenceEventType.OTHER, subject="Cosmetic UI")
    sig_misc = _create_test_signal("sig_misc", signal_type=SignalType.OTHER, subject="Cosmetic UI", supporting_event_ids=["e_misc"])

    insights = await analyzer.generate_insights(
        workspace_id="ws_default",
        company_id="acme_corp",
        signals=[sig_misc],
        events=[e_misc],
    )
    assert insights == []


@pytest.mark.anyio
async def test_custom_strategic_analyzer_injection() -> None:
    """Verify dependency injection of custom BaseStrategicAnalyzer implementations."""

    class MockCustomStrategicAnalyzer(BaseStrategicAnalyzer):
        async def generate_insights(
            self,
            workspace_id: str,
            company_id: str,
            signals: list[InferredSignal],
            events: list[IntelligenceEvent],
            existing_insights: list[Insight] | None = None,
        ) -> list[Insight]:
            return [
                Insight(
                    workspace_id=workspace_id,
                    company_id=company_id,
                    insight_type=InsightType.OTHER,
                    title="Mock Strategic Finding",
                    summary="Mock summary",
                    statement="Mock bounded takeaway.",
                    supporting_event_ids=[events[0].event_id],
                    factual_confidence=events[0].confidence,
                )
            ]

    custom_analyzer = MockCustomStrategicAnalyzer()
    e = _create_test_event("e_custom")
    res = await custom_analyzer.generate_insights(
        workspace_id="ws_1", company_id="acme", signals=[], events=[e]
    )

    assert len(res) == 1
    assert res[0].title == "Mock Strategic Finding"
