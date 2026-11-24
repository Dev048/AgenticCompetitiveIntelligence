"""Unit tests for Phase 1 Intelligence Analysis domain models and interfaces."""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from competitive_intelligence.intelligence import (
    BaseEventSequenceTracker,
    BaseIntelligenceAnalyzer,
    ConfidenceAssessment,
    ConfidenceLevel,
    EventSequence,
    EventTime,
    EvidenceReference,
    EvidenceRole,
    IntelligenceEvent,
    IntelligenceEventType,
    SignificanceAssessment,
    SignificanceLevel,
    SignificanceStatus,
    SourcePerspective,
    TimePrecision,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChangeResult,
    ChangeType,
    DocumentVersion,
)
from competitive_intelligence.retrieval_engine.models import (
    RetrievalResult,
    ScoredDocumentChunk,
)


@pytest.fixture
def base_evidence() -> EvidenceReference:
    return EvidenceReference(
        source_id="src_acme_pricing",
        document_id="doc_pricing_01",
        version_id="ver_2026_08",
        chunk_id="chunk_pricing_table",
        source_url="https://acme.com/pricing",
        source_category="pricing",
        source_perspective=SourcePerspective.FIRST_PARTY,
        extracted_text="Pro Plan increased to $30/month",
        acquired_at=datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc),
        role=EvidenceRole.PRIMARY_DIRECT,
    )


def test_pricing_change_event_with_state_transition(base_evidence: EvidenceReference) -> None:
    """Verify pricing change event with explicit previous_state and new_state."""
    event = IntelligenceEvent(
        workspace_id="ws_default",
        company_id="acme",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Acme Pro Tier Increased to $30",
        summary="Acme updated Pro tier pricing from $20/mo to $30/mo on their official pricing page.",
        subject="Pro Tier",
        previous_state={"price_usd": 20.0, "billing": "monthly"},
        new_state={"price_usd": 30.0, "billing": "monthly"},
        event_time=EventTime(
            occurred_at=None,
            precision=TimePrecision.UNKNOWN,
            acquired_at=datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc),
            detected_at=datetime(2026, 8, 15, 12, 5, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(
            score=1.0,
            level=ConfidenceLevel.VERY_HIGH,
            rationale="Deterministic table diff from official 1st-party pricing page.",
        ),
        significance=SignificanceAssessment(status=SignificanceStatus.UNASSESSED),
        evidence=[base_evidence],
        primary_perspective=SourcePerspective.FIRST_PARTY,
    )

    assert event.event_type == IntelligenceEventType.PRICING_CHANGE
    assert event.previous_state == {"price_usd": 20.0, "billing": "monthly"}
    assert event.new_state == {"price_usd": 30.0, "billing": "monthly"}
    assert event.confidence.score == 1.0
    assert event.confidence.level == ConfidenceLevel.VERY_HIGH
    assert event.significance.status == SignificanceStatus.UNASSESSED
    assert event.significance.level == SignificanceLevel.UNKNOWN
    assert len(event.evidence) == 1


def test_non_state_change_events(base_evidence: EvidenceReference) -> None:
    """Verify events that do not represent a before/after state transition."""
    # 1. Product Launch
    launch_event = IntelligenceEvent(
        workspace_id="ws_default",
        company_id="openai",
        event_type=IntelligenceEventType.PRODUCT_LAUNCH,
        title="OpenAI Announced GPT-4o",
        summary="OpenAI announced the release of GPT-4o with multimodal capabilities.",
        subject="GPT-4o",
        previous_state=None,
        new_state=None,
        event_time=EventTime(
            occurred_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
            precision=TimePrecision.DATE_ONLY,
            acquired_at=datetime(2026, 5, 13, 14, 0, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.98, level=ConfidenceLevel.VERY_HIGH),
        evidence=[base_evidence],
    )
    assert launch_event.previous_state is None
    assert launch_event.new_state is None
    assert launch_event.event_type == IntelligenceEventType.PRODUCT_LAUNCH

    # 2. Executive Hiring
    hiring_event = IntelligenceEvent(
        workspace_id="ws_default",
        company_id="snowflake",
        event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT,
        title="Snowflake Appoints New Chief AI Architect",
        summary="Snowflake hired a new VP of AI Systems to lead enterprise infrastructure.",
        subject="Executive Leadership",
        previous_state=None,
        new_state=None,
        event_time=EventTime(
            approximate_period="August 2026",
            precision=TimePrecision.MONTH_YEAR,
            acquired_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.90, level=ConfidenceLevel.HIGH),
        evidence=[base_evidence],
    )
    assert hiring_event.previous_state is None
    assert hiring_event.event_time.approximate_period == "August 2026"


def test_multi_source_evidence_references() -> None:
    """Verify an event supported by multiple sources with distinct roles and perspectives."""
    ev_first_party = EvidenceReference(
        source_id="src_official_blog",
        document_id="doc_blog_post",
        version_id="ver_01",
        source_url="https://company.com/blog/announcement",
        source_category="blog",
        source_perspective=SourcePerspective.FIRST_PARTY,
        extracted_text="We are acquiring VectorDB Inc for $50M.",
        acquired_at=datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc),
        role=EvidenceRole.PRIMARY_DIRECT,
    )
    ev_news = EvidenceReference(
        source_id="src_techcrunch",
        document_id="doc_tc_article",
        version_id="ver_01",
        source_url="https://techcrunch.com/2026/08/10/vectordb-acquisition",
        source_category="news",
        source_perspective=SourcePerspective.THIRD_PARTY_NEWS,
        extracted_text="Company X finalized the purchase of VectorDB Inc.",
        acquired_at=datetime(2026, 8, 10, 8, 30, tzinfo=timezone.utc),
        role=EvidenceRole.CONFIRMATORY,
    )

    event = IntelligenceEvent(
        workspace_id="ws_default",
        company_id="company_x",
        event_type=IntelligenceEventType.ACQUISITION,
        title="Company X Acquired VectorDB Inc",
        summary="Company X agreed to acquire VectorDB Inc for $50M in cash and stock.",
        subject="VectorDB Inc",
        event_time=EventTime(
            occurred_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
            precision=TimePrecision.DATE_ONLY,
            acquired_at=datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc),
        ),
        confidence=ConfidenceAssessment(
            score=0.99,
            level=ConfidenceLevel.VERY_HIGH,
            is_corroborated=True,
            rationale="Direct 1st-party official blog corroborated by 3rd-party journalism.",
        ),
        evidence=[ev_first_party, ev_news],
        primary_perspective=SourcePerspective.FIRST_PARTY,
    )

    assert len(event.evidence) == 2
    assert event.evidence[0].role == EvidenceRole.PRIMARY_DIRECT
    assert event.evidence[0].source_perspective == SourcePerspective.FIRST_PARTY
    assert event.evidence[1].role == EvidenceRole.CONFIRMATORY
    assert event.evidence[1].source_perspective == SourcePerspective.THIRD_PARTY_NEWS
    assert event.confidence.is_corroborated is True


def test_temporal_representation_separation() -> None:
    """Verify distinct separation of occurred_at, acquired_at, and detected_at timestamps."""
    occurred_time = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
    acquired_time = datetime(2026, 8, 2, 14, 30, tzinfo=timezone.utc)
    detected_time = datetime(2026, 8, 2, 14, 35, tzinfo=timezone.utc)

    event_time = EventTime(
        occurred_at=occurred_time,
        precision=TimePrecision.EXACT_TIMESTAMP,
        acquired_at=acquired_time,
        detected_at=detected_time,
    )

    assert event_time.occurred_at == occurred_time
    assert event_time.acquired_at == acquired_time
    assert event_time.detected_at == detected_time
    assert event_time.occurred_at != event_time.acquired_at
    assert event_time.precision == TimePrecision.EXACT_TIMESTAMP


def test_confidence_and_significance_independence(base_evidence: EvidenceReference) -> None:
    """Verify extraction confidence and strategic significance are strictly decoupled."""
    # High confidence extraction, but low/unassessed strategic significance
    event = IntelligenceEvent(
        workspace_id="ws_default",
        company_id="acme",
        event_type=IntelligenceEventType.RELEASE,
        title="Acme Released Patch v1.2.1",
        summary="Minor bug fix for button alignment in UI.",
        subject="v1.2.1",
        event_time=EventTime(acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc)),
        confidence=ConfidenceAssessment(
            score=1.0,
            level=ConfidenceLevel.VERY_HIGH,
            rationale="Exact changelog diff.",
        ),
        significance=SignificanceAssessment(
            status=SignificanceStatus.ASSESSED,
            level=SignificanceLevel.LOW,
            score=0.1,
            rationale="Routine minor bugfix patch with negligible competitive impact.",
            evaluated_at=datetime(2026, 8, 1, 1, 0, tzinfo=timezone.utc),
        ),
        evidence=[base_evidence],
    )

    assert event.confidence.level == ConfidenceLevel.VERY_HIGH
    assert event.confidence.score == 1.0
    assert event.significance.level == SignificanceLevel.LOW
    assert event.significance.score == 0.1


def test_historical_state_transition_sequence(base_evidence: EvidenceReference) -> None:
    """Verify $20 -> $30 -> $20 sequence preserves individual events and updates EventSequence."""
    # Event 1: $20 -> $30
    e1 = IntelligenceEvent(
        workspace_id="ws_default",
        company_id="acme",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Acme Pro Tier: $20 to $30",
        summary="Price increase to $30.",
        subject="Pro Tier",
        previous_state={"price": 20.0},
        new_state={"price": 30.0},
        event_time=EventTime(acquired_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
        confidence=ConfidenceAssessment(score=1.0, level=ConfidenceLevel.VERY_HIGH),
        evidence=[base_evidence],
        sequence_id="seq_acme_pro",
    )

    # Event 2: $30 -> $20 (price reverted)
    e2 = IntelligenceEvent(
        workspace_id="ws_default",
        company_id="acme",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Acme Pro Tier: $30 reverted to $20",
        summary="Price reverted to $20.",
        subject="Pro Tier",
        previous_state={"price": 30.0},
        new_state={"price": 20.0},
        event_time=EventTime(acquired_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
        confidence=ConfidenceAssessment(score=1.0, level=ConfidenceLevel.VERY_HIGH),
        evidence=[base_evidence],
        sequence_id="seq_acme_pro",
        preceding_event_id=e1.event_id,
    )

    # Track sequence
    seq = EventSequence(
        sequence_id="seq_acme_pro",
        company_id="acme",
        subject="Pro Tier",
        event_ids=[e1.event_id, e2.event_id],
        initial_state={"price": 20.0},
        current_state={"price": 20.0},
    )

    assert len(seq.event_ids) == 2
    assert e2.preceding_event_id == e1.event_id
    assert e1.previous_state == {"price": 20.0}
    assert e1.new_state == {"price": 30.0}
    assert e2.previous_state == {"price": 30.0}
    assert e2.new_state == {"price": 20.0}


def test_pydantic_validation_rules(base_evidence: EvidenceReference) -> None:
    """Verify strict validation for non-empty fields, URLs, and timezones."""
    # 1. Empty title raises ValueError
    with pytest.raises(ValidationError):
        IntelligenceEvent(
            workspace_id="ws_default",
            company_id="acme",
            event_type=IntelligenceEventType.PRICING_CHANGE,
            title="   ",
            summary="Valid summary",
            subject="Valid subject",
            event_time=EventTime(acquired_at=datetime.now(timezone.utc)),
            confidence=ConfidenceAssessment(score=1.0, level=ConfidenceLevel.HIGH),
            evidence=[base_evidence],
        )

    # 2. Empty evidence list raises ValidationError (min_length=1)
    with pytest.raises(ValidationError):
        IntelligenceEvent(
            workspace_id="ws_default",
            company_id="acme",
            event_type=IntelligenceEventType.PRICING_CHANGE,
            title="Valid title",
            summary="Valid summary",
            subject="Valid subject",
            event_time=EventTime(acquired_at=datetime.now(timezone.utc)),
            confidence=ConfidenceAssessment(score=1.0, level=ConfidenceLevel.HIGH),
            evidence=[],
        )

    # 3. Invalid URL in EvidenceReference raises ValueError
    with pytest.raises(ValidationError):
        EvidenceReference(
            source_id="src_1",
            document_id="doc_1",
            version_id="ver_1",
            source_url="not-a-valid-url",
            source_category="pricing",
            extracted_text="Some text",
            acquired_at=datetime.now(timezone.utc),
        )

    # 4. Naive datetime raises ValueError
    with pytest.raises(ValidationError):
        EventTime(
            acquired_at=datetime(2026, 1, 1),  # Naive without tzinfo
        )


@pytest.mark.anyio
async def test_base_intelligence_analyzer_contract_compliance(
    base_evidence: EvidenceReference,
) -> None:
    """Verify concrete subclass implementation of BaseIntelligenceAnalyzer contract."""

    class MockPricingAnalyzer(BaseIntelligenceAnalyzer):
        async def analyze_document_change(
            self,
            change: ChangeResult,
            current_version: DocumentVersion,
            previous_version: DocumentVersion | None,
            context_evidence: list[RetrievalResult] | None = None,
        ) -> list[IntelligenceEvent]:
            if change.change_type == ChangeType.CHANGED:
                return [
                    IntelligenceEvent(
                        workspace_id="ws_default",
                        company_id="acme",
                        event_type=IntelligenceEventType.PRICING_CHANGE,
                        title="Acme Price Change",
                        summary="Detected price change.",
                        subject="Pricing",
                        event_time=EventTime(acquired_at=current_version.acquired_at),
                        confidence=ConfidenceAssessment(
                            score=1.0, level=ConfidenceLevel.VERY_HIGH
                        ),
                        evidence=[
                            EvidenceReference(
                                source_id=current_version.metadata["source_id"],
                                document_id=current_version.document_id,
                                version_id=current_version.version_id,
                                source_url="https://acme.com/pricing",
                                source_category="pricing",
                                extracted_text=current_version.content[:50],
                                acquired_at=current_version.acquired_at,
                            )
                        ],
                    )
                ]
            return []

        async def analyze_retrieved_evidence(
            self,
            query: str,
            company_id: str,
            workspace_id: str,
            evidence: list[RetrievalResult],
        ) -> list[IntelligenceEvent]:
            return []

    analyzer = MockPricingAnalyzer()
    v1 = DocumentVersion(
        document_id="doc_1",
        content="Price $20",
        content_hash="hash_1",
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        metadata={"source_id": "src_1"},
    )
    v2 = DocumentVersion(
        document_id="doc_1",
        content="Price $30",
        content_hash="hash_2",
        acquired_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        metadata={"source_id": "src_1"},
    )
    change = ChangeResult(
        document_id="doc_1",
        current_version_id=v2.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.CHANGED,
        current_content_hash="hash_2",
        previous_content_hash="hash_1",
    )

    events = await analyzer.analyze_document_change(
        change=change,
        current_version=v2,
        previous_version=v1,
    )
    assert len(events) == 1
    assert events[0].event_type == IntelligenceEventType.PRICING_CHANGE
    assert events[0].evidence[0].version_id == v2.version_id
