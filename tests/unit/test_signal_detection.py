"""Unit tests for Event History and Signal Detection (Phase 3)."""

from datetime import datetime, timezone
from typing import Any
import pytest

from competitive_intelligence.intelligence import (
    BaseSignalDetector,
    ConfidenceAssessment,
    ConfidenceLevel,
    DeterministicEventSequenceTracker,
    DeterministicSignalDetector,
    EventRelationshipType,
    EventSequence,
    EventTime,
    EvidenceReference,
    EvidenceRole,
    InferredSignal,
    IntelligenceEvent,
    IntelligenceEventType,
    NumericStateComparator,
    SignalPolicy,
    SignalStatus,
    SignalType,
    SourcePerspective,
    TimePrecision,
)


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
    acquired_at: datetime | None = None,
    confidence_score: float = 0.90,
    evidence_role: EvidenceRole = EvidenceRole.PRIMARY_DIRECT,
) -> IntelligenceEvent:
    """Helper to construct valid grounded IntelligenceEvent instances for testing."""
    ev_time = EventTime(
        occurred_at=occurred_at,
        precision=TimePrecision.DATE_ONLY if occurred_at else TimePrecision.UNKNOWN,
        acquired_at=acquired_at or datetime(2026, 8, 1, tzinfo=timezone.utc),
        detected_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
    )
    conf = ConfidenceAssessment(
        score=confidence_score,
        level=ConfidenceLevel.HIGH if confidence_score >= 0.8 else ConfidenceLevel.LOW,
        rationale="Test evidence grounded.",
    )
    evidence_item = EvidenceReference(
        source_id="src_test",
        document_id=f"doc_{event_id}",
        version_id=f"ver_{event_id}",
        chunk_id=f"chunk_{event_id}",
        source_url="https://acme.com/news",
        source_category="news",
        extracted_text="Event evidence text.",
        acquired_at=acquired_at or datetime(2026, 8, 1, tzinfo=timezone.utc),
        role=evidence_role,
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


@pytest.fixture
def detector() -> DeterministicSignalDetector:
    return DeterministicSignalDetector()


@pytest.mark.anyio
async def test_empty_events_produces_zero_signals(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify that an empty list of events produces 0 InferredSignals."""
    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[],
    )
    assert signals == []


@pytest.mark.anyio
async def test_default_single_event_produces_zero_signals(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify conservative policy: by default, a single isolated event produces 0 signals."""
    single_event = _create_test_event(
        event_id="e1",
        event_type=IntelligenceEventType.PRODUCT_CHANGE,
        subject="Dark Mode",
        title="Dark Mode Added",
        summary="Added dark mode support.",
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[single_event],
    )
    assert signals == []


@pytest.mark.anyio
async def test_configured_single_event_produces_detected_signal(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify that an acquisition event explicitly authorized by SignalPolicy emits an EMERGING_ACTIVITY signal."""
    acq_event = _create_test_event(
        event_id="e_acq",
        event_type=IntelligenceEventType.ACQUISITION,
        subject="VectorDB Inc",
        title="Acquisition: Acme Acquired VectorDB Inc",
        summary="Acme acquired VectorDB Inc for $50M.",
        new_state={"deal_size": "$50M"},
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[acq_event],
    )

    assert len(signals) == 1
    sig = signals[0]
    assert sig.signal_type == SignalType.EMERGING_ACTIVITY
    assert sig.status == SignalStatus.DETECTED
    assert sig.subject == "VectorDB Inc"
    assert sig.supporting_event_ids == ["e_acq"]
    assert len(sig.supporting_evidence_ids) == 1


@pytest.mark.anyio
async def test_pricing_reversal_detection(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify state reversal ($20 -> $30 -> $20) produces a REVERSAL signal with RESOLVED status."""
    e1 = _create_test_event(
        event_id="e_price_1",
        subject="Enterprise Tier",
        previous_state={"price_usd": 20.0},
        new_state={"price_usd": 30.0},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    e2 = _create_test_event(
        event_id="e_price_2",
        subject="Enterprise Tier",
        previous_state={"price_usd": 30.0},
        new_state={"price_usd": 20.0},
        occurred_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e1, e2],
    )

    assert len(signals) == 1
    sig = signals[0]
    assert sig.signal_type == SignalType.REVERSAL
    assert sig.status == SignalStatus.RESOLVED
    assert sig.subject == "Enterprise Tier"
    assert sig.supporting_event_ids == ["e_price_1", "e_price_2"]
    assert "reverted to initial state" in sig.summary


@pytest.mark.anyio
async def test_monotonic_directional_trend(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify monotonic pricing progression ($10 -> $15 -> $20) emits a DIRECTIONAL_TREND signal."""
    e1 = _create_test_event(
        event_id="e_trend_1",
        subject="Pro Tier",
        previous_state={"price_usd": 10.0},
        new_state={"price_usd": 15.0},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    e2 = _create_test_event(
        event_id="e_trend_2",
        subject="Pro Tier",
        previous_state={"price_usd": 15.0},
        new_state={"price_usd": 20.0},
        occurred_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e1, e2],
    )

    assert len(signals) == 1
    sig = signals[0]
    assert sig.signal_type == SignalType.DIRECTIONAL_TREND
    assert sig.status == SignalStatus.ACTIVE
    assert "expanding" in sig.title or "upward" in sig.title
    assert sig.supporting_event_ids == ["e_trend_1", "e_trend_2"]


@pytest.mark.anyio
async def test_unorderable_states_do_not_produce_false_trend(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify arbitrary unorderable dictionaries do not emit false directional trends."""
    e1 = _create_test_event(
        event_id="e_unorder_1",
        subject="Custom Config",
        new_state={"theme": "dark", "layout": "grid"},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    e2 = _create_test_event(
        event_id="e_unorder_2",
        subject="Custom Config",
        new_state={"theme": "blue", "layout": "flex"},
        occurred_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e1, e2],
    )

    # Should fall back to REPEATED_CHANGE or 0 signals, but NEVER DIRECTIONAL_TREND
    for sig in signals:
        assert sig.signal_type != SignalType.DIRECTIONAL_TREND


@pytest.mark.anyio
async def test_release_acceleration_cadence(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify rapid cadence of 3 software releases within 10 days emits ACCELERATION."""
    e1 = _create_test_event(
        event_id="e_rel_1",
        event_type=IntelligenceEventType.RELEASE,
        subject="Core Engine",
        occurred_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    e2 = _create_test_event(
        event_id="e_rel_2",
        event_type=IntelligenceEventType.RELEASE,
        subject="Core Engine",
        occurred_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
    )
    e3 = _create_test_event(
        event_id="e_rel_3",
        event_type=IntelligenceEventType.RELEASE,
        subject="Core Engine",
        occurred_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e1, e2, e3],
    )

    assert len(signals) == 1
    sig = signals[0]
    assert sig.signal_type == SignalType.ACCELERATION
    assert sig.status == SignalStatus.ACTIVE
    assert len(sig.supporting_event_ids) == 3


@pytest.mark.anyio
async def test_sustained_hiring_activity(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify spaced executive hires across 90 days emit SUSTAINED_ACTIVITY."""
    e1 = _create_test_event(
        event_id="e_hire_1",
        event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT,
        subject="Leadership",
        title="Hired VP AI",
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    e2 = _create_test_event(
        event_id="e_hire_2",
        event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT,
        subject="Leadership",
        title="Hired VP Sales",
        occurred_at=datetime(2026, 2, 15, tzinfo=timezone.utc),
    )
    e3 = _create_test_event(
        event_id="e_hire_3",
        event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT,
        subject="Leadership",
        title="Hired VP Marketing",
        occurred_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e1, e2, e3],
    )

    assert len(signals) == 1
    sig = signals[0]
    assert sig.signal_type == SignalType.SUSTAINED_ACTIVITY
    assert sig.status == SignalStatus.ACTIVE


@pytest.mark.anyio
async def test_contradictory_events_handling(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify conflicting reports emit CONTRADICTION with UNCERTAIN status and capped confidence."""
    e_launch = _create_test_event(
        event_id="e_launch",
        event_type=IntelligenceEventType.PRODUCT_LAUNCH,
        subject="Voice Mode",
        title="Voice Mode Launched",
        evidence_role=EvidenceRole.PRIMARY_DIRECT,
        occurred_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    e_delay = _create_test_event(
        event_id="e_delay",
        event_type=IntelligenceEventType.PRODUCT_CHANGE,
        subject="Voice Mode",
        title="Voice Mode Delayed",
        evidence_role=EvidenceRole.CONTRADICTING,
        occurred_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e_launch, e_delay],
    )

    assert len(signals) == 1
    sig = signals[0]
    assert sig.signal_type == SignalType.CONTRADICTION
    assert sig.status == SignalStatus.UNCERTAIN
    assert sig.confidence.score <= 0.60
    assert sig.confidence.level == ConfidenceLevel.LOW
    assert "Dispute" in sig.title or "Contradiction" in sig.title


@pytest.mark.anyio
async def test_out_of_order_chronological_sorting(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify that events ingested out of order sort chronologically by occurred_at."""
    e_later = _create_test_event(
        event_id="e_later",
        subject="Pricing Tier",
        previous_state={"price_usd": 30.0},
        new_state={"price_usd": 20.0},
        occurred_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    e_earlier = _create_test_event(
        event_id="e_earlier",
        subject="Pricing Tier",
        previous_state={"price_usd": 20.0},
        new_state={"price_usd": 30.0},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    # Ingest e_later before e_earlier
    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e_later, e_earlier],
    )

    assert len(signals) == 1
    sig = signals[0]
    assert sig.signal_type == SignalType.REVERSAL
    # First observed should be Jan 1, last observed should be June 1
    assert sig.first_observed_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert sig.last_observed_at == datetime(2026, 6, 1, tzinfo=timezone.utc)


@pytest.mark.anyio
async def test_unrelated_events_separation(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify disjoint subjects for the same company produce separate or zero signals."""
    e_price = _create_test_event(
        event_id="e_price",
        subject="Pricing Tier",
        previous_state={"price_usd": 20.0},
        new_state={"price_usd": 30.0},
    )
    e_hiring = _create_test_event(
        event_id="e_hiring",
        subject="VP Sales Hiring",
        event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT,
    )

    # Isolated minor events on disjoint subjects -> 0 signals
    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e_price, e_hiring],
    )
    assert signals == []


@pytest.mark.anyio
async def test_full_provenance_preservation(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify transitive provenance: InferredSignal -> supporting_event_ids & supporting_evidence_ids."""
    e1 = _create_test_event(
        event_id="e_prov_1",
        subject="API Gateway",
        previous_state={"price_usd": 10.0},
        new_state={"price_usd": 20.0},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    e2 = _create_test_event(
        event_id="e_prov_2",
        subject="API Gateway",
        previous_state={"price_usd": 20.0},
        new_state={"price_usd": 30.0},
        occurred_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e1, e2],
    )

    assert len(signals) == 1
    sig = signals[0]
    assert sig.supporting_event_ids == ["e_prov_1", "e_prov_2"]
    assert len(sig.supporting_evidence_ids) == 2
    assert e1.evidence[0].chunk_id == "chunk_e_prov_1"
    assert e2.evidence[0].chunk_id == "chunk_e_prov_2"


def test_event_sequence_tracker_integration() -> None:
    """Verify EventSequence structural state transition tracking."""
    tracker = DeterministicEventSequenceTracker()
    e1 = _create_test_event(
        event_id="e_seq_1",
        subject="Enterprise Tier",
        previous_state={"price_usd": 20.0},
        new_state={"price_usd": 30.0},
    )
    seq = tracker.append_event(None, e1)
    assert seq.initial_state == {"price_usd": 20.0}
    assert seq.current_state == {"price_usd": 30.0}
    assert seq.event_ids == ["e_seq_1"]

    e2 = _create_test_event(
        event_id="e_seq_2",
        subject="Enterprise Tier",
        previous_state={"price_usd": 30.0},
        new_state={"price_usd": 20.0},
    )
    updated_seq = tracker.append_event(seq, e2)
    assert updated_seq.initial_state == {"price_usd": 20.0}
    assert updated_seq.current_state == {"price_usd": 20.0}
    assert updated_seq.event_ids == ["e_seq_1", "e_seq_2"]


@pytest.mark.anyio
async def test_low_confidence_events_do_not_produce_high_signal_confidence(
    detector: DeterministicSignalDetector,
) -> None:
    """Verify that multiple low-confidence events do not artificially produce high signal confidence."""
    e1 = _create_test_event(
        event_id="e_low_1",
        subject="Rumor Feature",
        confidence_score=0.40,
        previous_state={"price_usd": 10.0},
        new_state={"price_usd": 20.0},
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    e2 = _create_test_event(
        event_id="e_low_2",
        subject="Rumor Feature",
        confidence_score=0.45,
        previous_state={"price_usd": 20.0},
        new_state={"price_usd": 30.0},
        occurred_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    signals = await detector.detect_signals(
        workspace_id="ws_default",
        company_id="acme_corp",
        events=[e1, e2],
    )

    assert len(signals) == 1
    sig = signals[0]
    # Confidence must remain bounded by low event scores
    assert sig.confidence.score <= 0.55
    assert sig.confidence.level == ConfidenceLevel.LOW


@pytest.mark.anyio
async def test_custom_signal_detector_injection() -> None:
    """Verify dependency injection of custom BaseSignalDetector implementations."""

    class MockCustomSignalDetector(BaseSignalDetector):
        async def detect_signals(
            self,
            workspace_id: str,
            company_id: str,
            events: list[IntelligenceEvent],
            existing_signals: list[InferredSignal] | None = None,
        ) -> list[InferredSignal]:
            return [
                InferredSignal(
                    workspace_id=workspace_id,
                    company_id=company_id,
                    signal_type=SignalType.OTHER,
                    subject="Custom Strategy",
                    title="Mock Signal",
                    summary="Mock detector emitted finding.",
                    supporting_event_ids=[events[0].event_id],
                )
            ]

    custom_detector = MockCustomSignalDetector()
    e = _create_test_event(event_id="e_custom")
    res = await custom_detector.detect_signals(
        workspace_id="ws_1", company_id="acme", events=[e]
    )

    assert len(res) == 1
    assert res[0].title == "Mock Signal"
