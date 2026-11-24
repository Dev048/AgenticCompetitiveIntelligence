"""Unit tests for DocumentChangeAnalyzer and RuleBasedDiffInterpreter."""

from datetime import datetime, timezone
import pytest

from competitive_intelligence.intelligence import (
    BaseDiffInterpreter,
    ConfidenceLevel,
    DocumentChangeAnalyzer,
    EvidenceRole,
    EvidenceSnippet,
    IntelligenceEventType,
    RawEventPayload,
    RuleBasedDiffInterpreter,
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


@pytest.fixture
def default_metadata() -> dict[str, str]:
    return {
        "workspace_id": "ws_test",
        "company_id": "acme_corp",
        "source_id": "src_acme_web",
        "canonical_url": "https://acme.com/pricing",
        "source_category": "pricing",
        "source_perspective": "first_party",
    }


@pytest.fixture
def analyzer() -> DocumentChangeAnalyzer:
    return DocumentChangeAnalyzer()


@pytest.mark.anyio
async def test_unchanged_document_produces_zero_events(
    analyzer: DocumentChangeAnalyzer, default_metadata: dict[str, str]
) -> None:
    """Verify that an UNCHANGED change result produces 0 IntelligenceEvents."""
    v1 = DocumentVersion(
        document_id="doc_1",
        content="Pro Plan: $20/month",
        content_hash="hash_123",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
    )
    change = ChangeResult(
        document_id="doc_1",
        current_version_id=v1.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.UNCHANGED,
        current_content_hash="hash_123",
        previous_content_hash="hash_123",
    )

    events = await analyzer.analyze_document_change(
        change=change,
        current_version=v1,
        previous_version=v1,
    )
    assert events == []


@pytest.mark.anyio
async def test_identical_hash_changed_type_produces_zero_events(
    analyzer: DocumentChangeAnalyzer, default_metadata: dict[str, str]
) -> None:
    """Verify that if hashes are identical, no event is created even if flagged as changed."""
    v1 = DocumentVersion(
        document_id="doc_1",
        content="Pro Plan: $20/month",
        content_hash="hash_identical",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
    )
    v2 = DocumentVersion(
        document_id="doc_1",
        content="Pro Plan: $20/month",
        content_hash="hash_identical",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc),
    )
    change = ChangeResult(
        document_id="doc_1",
        current_version_id=v2.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.CHANGED,
        current_content_hash="hash_identical",
        previous_content_hash="hash_identical",
    )

    events = await analyzer.analyze_document_change(
        change=change,
        current_version=v2,
        previous_version=v1,
    )
    assert events == []


@pytest.mark.anyio
async def test_cosmetic_diff_produces_zero_events(
    analyzer: DocumentChangeAnalyzer, default_metadata: dict[str, str]
) -> None:
    """Verify that cosmetic diffs (copyright year update or whitespace) produce 0 events."""
    v1 = DocumentVersion(
        document_id="doc_1",
        content="Acme Solutions\n© 2025 Acme Inc. All rights reserved.",
        content_hash="hash_prev",
        metadata=default_metadata,
        acquired_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    v2 = DocumentVersion(
        document_id="doc_1",
        content="Acme Solutions\n© 2026 Acme Inc. All rights reserved.",
        content_hash="hash_curr",
        metadata=default_metadata,
        acquired_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    change = ChangeResult(
        document_id="doc_1",
        current_version_id=v2.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.CHANGED,
        current_content_hash="hash_curr",
        previous_content_hash="hash_prev",
    )

    events = await analyzer.analyze_document_change(
        change=change,
        current_version=v2,
        previous_version=v1,
    )
    assert events == []


@pytest.mark.anyio
async def test_pricing_change_with_state_transition(
    analyzer: DocumentChangeAnalyzer, default_metadata: dict[str, str]
) -> None:
    """Verify pricing increase extraction with previous_state and new_state and evidence provenance."""
    v1 = DocumentVersion(
        document_id="doc_pricing",
        content="Acme Cloud Pricing\nStarter Plan: $10/mo\nPro Plan: $20/mo\nEnterprise: Contact Us",
        content_hash="hash_p1",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc),
    )
    v2 = DocumentVersion(
        document_id="doc_pricing",
        content="Acme Cloud Pricing\nStarter Plan: $10/mo\nPro Plan: $30/mo\nEnterprise: Contact Us",
        content_hash="hash_p2",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc),
    )
    change = ChangeResult(
        document_id="doc_pricing",
        current_version_id=v2.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.CHANGED,
        current_content_hash="hash_p2",
        previous_content_hash="hash_p1",
    )

    events = await analyzer.analyze_document_change(
        change=change,
        current_version=v2,
        previous_version=v1,
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == IntelligenceEventType.PRICING_CHANGE
    assert event.subject == "Pro"
    assert event.previous_state == {"price_usd": 20.0, "billing_period": "mo"}
    assert event.new_state == {"price_usd": 30.0, "billing_period": "mo"}
    assert event.confidence.level == ConfidenceLevel.VERY_HIGH
    assert event.confidence.score == 1.0
    assert event.significance.status == SignificanceStatus.UNASSESSED
    assert event.significance.level == SignificanceLevel.UNKNOWN
    assert event.primary_perspective == SourcePerspective.FIRST_PARTY

    # Verify both current and previous evidence references
    assert len(event.evidence) == 2
    curr_ev = next(e for e in event.evidence if e.version_id == v2.version_id)
    prev_ev = next(e for e in event.evidence if e.version_id == v1.version_id)
    assert "$30" in curr_ev.extracted_text
    assert "$20" in prev_ev.extracted_text
    assert curr_ev.source_id == "src_acme_web"


@pytest.mark.anyio
async def test_product_launch_non_state_event(
    analyzer: DocumentChangeAnalyzer, default_metadata: dict[str, str]
) -> None:
    """Verify product launch announcement produces an event with previous_state=None and new_state=None."""
    default_metadata["source_category"] = "blog"
    v1 = DocumentVersion(
        document_id="doc_blog",
        content="Acme Engineering Blog\nWelcome to our quarterly engineering roundup.",
        content_hash="hash_b1",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
    )
    v2 = DocumentVersion(
        document_id="doc_blog",
        content="Acme Engineering Blog\nWelcome to our quarterly engineering roundup.\nIntroducing Acme Vector Search, our ultra-fast embedding retrieval engine.",
        content_hash="hash_b2",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc),
    )
    change = ChangeResult(
        document_id="doc_blog",
        current_version_id=v2.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.CHANGED,
        current_content_hash="hash_b2",
        previous_content_hash="hash_b1",
    )

    events = await analyzer.analyze_document_change(
        change=change,
        current_version=v2,
        previous_version=v1,
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == IntelligenceEventType.PRODUCT_LAUNCH
    assert event.subject == "Acme Vector Search"
    assert event.previous_state is None
    assert event.new_state is None
    assert "Acme Vector Search" in event.evidence[0].extracted_text


@pytest.mark.anyio
async def test_software_release_with_explicit_date(
    analyzer: DocumentChangeAnalyzer, default_metadata: dict[str, str]
) -> None:
    """Verify release changelog extracts explicit date into occurred_at with DATE_ONLY precision."""
    default_metadata["source_category"] = "documentation"
    v1 = DocumentVersion(
        document_id="doc_release",
        content="Changelog\n### v2.0.0 - Released June 1, 2026\n- Initial 2.0 release",
        content_hash="hash_r1",
        metadata=default_metadata,
        acquired_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    v2 = DocumentVersion(
        document_id="doc_release",
        content="Changelog\n### v2.1.0 - Released August 15, 2026\n- Added Hybrid Reranking\n\n### v2.0.0 - Released June 1, 2026\n- Initial 2.0 release",
        content_hash="hash_r2",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
    )
    change = ChangeResult(
        document_id="doc_release",
        current_version_id=v2.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.CHANGED,
        current_content_hash="hash_r2",
        previous_content_hash="hash_r1",
    )

    events = await analyzer.analyze_document_change(
        change=change,
        current_version=v2,
        previous_version=v1,
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == IntelligenceEventType.RELEASE
    assert event.subject == "v2.1.0"
    assert event.event_time.occurred_at == datetime(2026, 8, 15, tzinfo=timezone.utc)
    assert event.event_time.precision == TimePrecision.DATE_ONLY
    assert event.event_time.acquired_at == v2.acquired_at


@pytest.mark.anyio
async def test_executive_hiring_appointment(
    analyzer: DocumentChangeAnalyzer, default_metadata: dict[str, str]
) -> None:
    """Verify leadership hiring update extraction."""
    default_metadata["source_category"] = "careers"
    v1 = DocumentVersion(
        document_id="doc_team",
        content="Our Leadership Team\nCEO: John Smith",
        content_hash="hash_t1",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    v2 = DocumentVersion(
        document_id="doc_team",
        content="Our Leadership Team\nCEO: John Smith\nWelcoming Alice Cooper as Chief Technology Officer.",
        content_hash="hash_t2",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
    )
    change = ChangeResult(
        document_id="doc_team",
        current_version_id=v2.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.CHANGED,
        current_content_hash="hash_t2",
        previous_content_hash="hash_t1",
    )

    events = await analyzer.analyze_document_change(
        change=change,
        current_version=v2,
        previous_version=v1,
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == IntelligenceEventType.HIRING_OR_HEADCOUNT
    assert event.subject == "Chief Technology Officer"
    assert "Alice Cooper" in event.title


@pytest.mark.anyio
async def test_multi_event_document_update(
    analyzer: DocumentChangeAnalyzer, default_metadata: dict[str, str]
) -> None:
    """Verify that a single document update containing multiple distinct changes produces N events."""
    v1 = DocumentVersion(
        document_id="doc_multi",
        content="Acme Platform Status\nPro Plan: $20/mo\nREST API v1 is active.",
        content_hash="hash_m1",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    v2 = DocumentVersion(
        document_id="doc_multi",
        content="Acme Platform Status\nPro Plan: $30/mo\nDeprecating REST API v1 in favor of v2 GraphQL API.",
        content_hash="hash_m2",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    change = ChangeResult(
        document_id="doc_multi",
        current_version_id=v2.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.CHANGED,
        current_content_hash="hash_m2",
        previous_content_hash="hash_m1",
    )

    events = await analyzer.analyze_document_change(
        change=change,
        current_version=v2,
        previous_version=v1,
    )

    assert len(events) == 2
    types = {e.event_type for e in events}
    assert IntelligenceEventType.PRICING_CHANGE in types
    assert IntelligenceEventType.PRODUCT_DEPRECATION in types


@pytest.mark.anyio
async def test_initial_document_selective_event_emission(
    analyzer: DocumentChangeAnalyzer, default_metadata: dict[str, str]
) -> None:
    """Verify that INITIAL documents only produce events if content represents an identifiable event."""
    # 1. Standard static landing page: produces 0 events
    v_static = DocumentVersion(
        document_id="doc_static",
        content="Welcome to Acme Corporation. We build cloud software for enterprise businesses.",
        content_hash="hash_static",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    change_static = ChangeResult(
        document_id="doc_static",
        current_version_id=v_static.version_id,
        change_type=ChangeType.INITIAL,
        current_content_hash="hash_static",
    )
    events_static = await analyzer.analyze_document_change(
        change=change_static,
        current_version=v_static,
        previous_version=None,
    )
    assert events_static == []

    # 2. Standalone press release launch announcement: produces 1 event
    v_announcement = DocumentVersion(
        document_id="doc_pr",
        content="Acme Corporation Press Release\nAnnouncing Acme Vector DB for high-speed embeddings.",
        content_hash="hash_pr",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    change_pr = ChangeResult(
        document_id="doc_pr",
        current_version_id=v_announcement.version_id,
        change_type=ChangeType.INITIAL,
        current_content_hash="hash_pr",
    )
    events_pr = await analyzer.analyze_document_change(
        change=change_pr,
        current_version=v_announcement,
        previous_version=None,
    )
    assert len(events_pr) == 1
    assert events_pr[0].event_type == IntelligenceEventType.PRODUCT_LAUNCH


@pytest.mark.anyio
async def test_custom_diff_interpreter_injection(default_metadata: dict[str, str]) -> None:
    """Verify DocumentChangeAnalyzer supports injecting custom BaseDiffInterpreter implementations."""

    class MockCustomInterpreter(BaseDiffInterpreter):
        async def interpret_diff(
            self,
            previous_content: str | None,
            current_content: str,
            source_category: str,
            source_perspective: SourcePerspective,
            metadata: dict[str, str],
        ) -> list[RawEventPayload]:
            return [
                RawEventPayload(
                    event_type=IntelligenceEventType.STRATEGIC_ANNOUNCEMENT,
                    title="Mock Strategic Pivot",
                    summary="Custom mock analyzer detected pivot.",
                    subject="Strategy",
                    confidence_score=0.99,
                    confidence_level=ConfidenceLevel.VERY_HIGH,
                    evidence_snippets=[
                        EvidenceSnippet(
                            extracted_text="Custom diff statement",
                            is_current_version=True,
                            role=EvidenceRole.PRIMARY_DIRECT,
                        )
                    ],
                )
            ]

    custom_analyzer = DocumentChangeAnalyzer(diff_interpreter=MockCustomInterpreter())
    v1 = DocumentVersion(
        document_id="doc_custom",
        content="Old content",
        content_hash="hash_1",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    v2 = DocumentVersion(
        document_id="doc_custom",
        content="New content",
        content_hash="hash_2",
        metadata=default_metadata,
        acquired_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    change = ChangeResult(
        document_id="doc_custom",
        current_version_id=v2.version_id,
        previous_version_id=v1.version_id,
        change_type=ChangeType.CHANGED,
        current_content_hash="hash_2",
        previous_content_hash="hash_1",
    )

    events = await custom_analyzer.analyze_document_change(
        change=change,
        current_version=v2,
        previous_version=v1,
    )
    assert len(events) == 1
    assert events[0].event_type == IntelligenceEventType.STRATEGIC_ANNOUNCEMENT
