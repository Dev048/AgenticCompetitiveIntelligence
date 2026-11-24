"""Unit tests for RetrievedEvidenceAnalyzer and RuleBasedEvidenceInterpreter."""

from datetime import datetime, timezone
import pytest

from competitive_intelligence.intelligence import (
    BaseEvidenceInterpreter,
    ConfidenceLevel,
    EvidenceRole,
    IntelligenceEventType,
    RetrievedEvidenceAnalyzer,
    RetrievedEvidenceSnippet,
    RuleBasedEvidenceInterpreter,
    SignificanceLevel,
    SignificanceStatus,
    SourcePerspective,
    TimePrecision,
)
from competitive_intelligence.intelligence.analyzers.retrieved_evidence_analyzer import (
    RawEventPayload,
)
from competitive_intelligence.retrieval_engine.models import RetrievalResult


@pytest.fixture
def analyzer() -> RetrievedEvidenceAnalyzer:
    return RetrievedEvidenceAnalyzer()


@pytest.mark.anyio
async def test_empty_evidence_produces_zero_events(analyzer: RetrievedEvidenceAnalyzer) -> None:
    """Verify that an empty list of retrieval results produces 0 IntelligenceEvents."""
    events = await analyzer.analyze_retrieved_evidence(
        query="What are Acme's recent announcements?",
        company_id="acme_corp",
        workspace_id="ws_default",
        evidence=[],
    )
    assert events == []


@pytest.mark.anyio
async def test_pure_contextual_chunks_produce_zero_events(
    analyzer: RetrievedEvidenceAnalyzer,
) -> None:
    """Verify that generic background/contextual chunks with no factual event emit 0 events."""
    chunks = [
        RetrievalResult(
            chunk_id="chunk_bg_1",
            score=0.85,
            text="Acme Corporation is a cloud software company headquartered in San Francisco.",
            document_id="doc_about",
            version_id="ver_01",
            source_id="src_acme",
            canonical_url="https://acme.com/about",
            source_category="website",
        ),
        RetrievalResult(
            chunk_id="chunk_bg_2",
            score=0.72,
            text="We offer support 24/7 for all our valued business customers worldwide.",
            document_id="doc_support",
            version_id="ver_01",
            source_id="src_acme",
            canonical_url="https://acme.com/support",
            source_category="website",
        ),
    ]

    events = await analyzer.analyze_retrieved_evidence(
        query="Tell me about Acme",
        company_id="acme_corp",
        workspace_id="ws_default",
        evidence=chunks,
    )
    assert events == []


@pytest.mark.anyio
async def test_single_first_party_announcement(analyzer: RetrievedEvidenceAnalyzer) -> None:
    """Verify official 1st-party product launch announcement extraction and provenance."""
    chunks = [
        RetrievalResult(
            chunk_id="chunk_launch_01",
            score=0.92,
            text="Announcing Acme Vector Engine: our new ultra-fast vector retrieval engine.",
            document_id="doc_blog_99",
            version_id="ver_99",
            source_id="src_acme_blog",
            canonical_url="https://acme.com/blog/announcing-vector-engine",
            source_category="blog",
            metadata={"source_perspective": "first_party"},
        )
    ]

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme new products",
        company_id="acme_corp",
        workspace_id="ws_default",
        evidence=chunks,
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == IntelligenceEventType.PRODUCT_LAUNCH
    assert event.subject == "Acme Vector Engine"
    assert event.primary_perspective == SourcePerspective.FIRST_PARTY
    assert event.confidence.level == ConfidenceLevel.VERY_HIGH
    assert event.significance.status == SignificanceStatus.UNASSESSED
    assert len(event.evidence) == 1
    assert event.evidence[0].chunk_id == "chunk_launch_01"
    assert event.evidence[0].source_url == "https://acme.com/blog/announcing-vector-engine"


@pytest.mark.anyio
async def test_multi_source_deduplication_clusters_same_event(
    analyzer: RetrievedEvidenceAnalyzer,
) -> None:
    """Verify that multiple retrieved chunks describing the same event merge into 1 event with distinct evidence."""
    chunks = [
        RetrievalResult(
            chunk_id="chunk_blog",
            score=0.95,
            text="We are thrilled to announce that Acme acquired VectorDB Inc for $50M.",
            document_id="doc_blog",
            version_id="ver_01",
            source_id="src_acme_blog",
            canonical_url="https://acme.com/blog/acquisition",
            source_category="blog",
            metadata={"source_perspective": "first_party"},
        ),
        RetrievalResult(
            chunk_id="chunk_news",
            score=0.88,
            text="TechCrunch reports: Acme acquired VectorDB Inc for $50M in cash.",
            document_id="doc_tc",
            version_id="ver_02",
            source_id="src_techcrunch",
            canonical_url="https://techcrunch.com/acme-vectordb",
            source_category="news",
            metadata={"source_perspective": "third_party_news"},
        ),
        RetrievalResult(
            chunk_id="chunk_reuters",
            score=0.82,
            text="Acme acquired VectorDB Inc to bolster its AI infrastructure portfolio.",
            document_id="doc_reuters",
            version_id="ver_03",
            source_id="src_reuters",
            canonical_url="https://reuters.com/acme-deal",
            source_category="news",
            metadata={"source_perspective": "third_party_news"},
        ),
    ]

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme acquisitions",
        company_id="acme_corp",
        workspace_id="ws_default",
        evidence=chunks,
    )

    # 3 chunks for the same acquisition should deduplicate into 1 IntelligenceEvent with 3 EvidenceReferences
    assert len(events) == 1
    event = events[0]
    assert event.event_type == IntelligenceEventType.ACQUISITION
    assert event.subject == "VectorDB Inc"
    assert len(event.evidence) == 3
    chunk_ids = {e.chunk_id for e in event.evidence}
    assert chunk_ids == {"chunk_blog", "chunk_news", "chunk_reuters"}
    # is_corroborated remains False pending dedicated downstream independence analysis
    assert event.confidence.is_corroborated is False


@pytest.mark.anyio
async def test_conflicting_evidence_recording(analyzer: RetrievedEvidenceAnalyzer) -> None:
    """Verify that conflicting / delay statements are recorded with CONTRADICTING role and lower confidence."""
    chunks = [
        RetrievalResult(
            chunk_id="chunk_delay",
            score=0.89,
            text="Internal sources confirm rollout delayed for Advanced Voice Mode due to safety review.",
            document_id="doc_news_delay",
            version_id="ver_01",
            source_id="src_news",
            canonical_url="https://news.com/delay",
            source_category="news",
        )
    ]

    events = await analyzer.analyze_retrieved_evidence(
        query="Advanced Voice Mode status",
        company_id="openai",
        workspace_id="ws_default",
        evidence=chunks,
    )

    assert len(events) == 1
    event = events[0]
    assert event.evidence[0].role == EvidenceRole.CONTRADICTING
    assert event.confidence.level == ConfidenceLevel.LOW
    assert "Contradicting statement" in (event.confidence.rationale or "")


@pytest.mark.anyio
async def test_community_speculation_low_confidence(
    analyzer: RetrievedEvidenceAnalyzer,
) -> None:
    """Verify that community discussions from Reddit/social forums produce LOW confidence events."""
    chunks = [
        RetrievalResult(
            chunk_id="chunk_reddit",
            score=0.75,
            text="Rumor on r/SaaS: Acme might be discontinuing their free tier next month.",
            document_id="doc_reddit_post",
            version_id="ver_01",
            source_id="src_reddit",
            canonical_url="https://reddit.com/r/SaaS/comments/123",
            source_category="reddit",
            metadata={"source_perspective": "community_discussion"},
        )
    ]

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme pricing rumors",
        company_id="acme_corp",
        workspace_id="ws_default",
        evidence=chunks,
    )

    assert len(events) == 1
    event = events[0]
    assert event.primary_perspective == SourcePerspective.COMMUNITY_DISCUSSION
    assert event.confidence.level == ConfidenceLevel.LOW
    assert "Community forum" in (event.confidence.rationale or "")


@pytest.mark.anyio
async def test_multi_event_retrieval_set(analyzer: RetrievedEvidenceAnalyzer) -> None:
    """Verify that a multi-topic retrieval set produces N distinct IntelligenceEvents."""
    chunks = [
        # Event 1: Product launch
        RetrievalResult(
            chunk_id="chunk_e1",
            score=0.91,
            text="Introducing Acme AI Studio for generative workflows.",
            document_id="doc_1",
            version_id="ver_1",
            source_id="src_1",
            canonical_url="https://acme.com/studio",
            source_category="blog",
        ),
        # Event 2: Executive hire
        RetrievalResult(
            chunk_id="chunk_e2",
            score=0.88,
            text="Welcoming David Miller as Chief Operating Officer.",
            document_id="doc_2",
            version_id="ver_2",
            source_id="src_2",
            canonical_url="https://acme.com/careers",
            source_category="careers",
        ),
        # Event 3: Software release
        RetrievalResult(
            chunk_id="chunk_e3",
            score=0.84,
            text="Release notes for v4.1.0 published on August 20, 2026.",
            document_id="doc_3",
            version_id="ver_3",
            source_id="src_3",
            canonical_url="https://acme.com/changelog",
            source_category="documentation",
        ),
    ]

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme recent updates",
        company_id="acme_corp",
        workspace_id="ws_default",
        evidence=chunks,
    )

    assert len(events) == 3
    types = {e.event_type for e in events}
    assert IntelligenceEventType.PRODUCT_LAUNCH in types
    assert IntelligenceEventType.HIRING_OR_HEADCOUNT in types
    assert IntelligenceEventType.RELEASE in types


@pytest.mark.anyio
async def test_explicit_vs_relative_vs_unknown_temporal_handling(
    analyzer: RetrievedEvidenceAnalyzer,
) -> None:
    """Verify handling of explicit dates, coarse periods, relative dates, and unknown time."""
    chunks = [
        # 1. Explicit date
        RetrievalResult(
            chunk_id="chunk_t1",
            score=0.90,
            text="Release v1.2.0 on August 15, 2026.",
            document_id="doc_t1",
            version_id="ver_1",
            canonical_url="https://acme.com/r1",
            source_category="documentation",
        ),
        # 2. Coarse quarter period
        RetrievalResult(
            chunk_id="chunk_t2",
            score=0.85,
            text="Announcing Acme Cloud Q3 2026 preview.",
            document_id="doc_t2",
            version_id="ver_2",
            canonical_url="https://acme.com/r2",
            source_category="blog",
        ),
        # 3. Relative date expression
        RetrievalResult(
            chunk_id="chunk_t3",
            score=0.80,
            text="Appointed Jane Doe as VP of AI yesterday.",
            document_id="doc_t3",
            version_id="ver_3",
            canonical_url="https://acme.com/r3",
            source_category="careers",
        ),
        # 4. Unknown date
        RetrievalResult(
            chunk_id="chunk_t4",
            score=0.82,
            text="Strategic partnership with Snowflake Enterprise.",
            document_id="doc_t4",
            version_id="ver_4",
            canonical_url="https://acme.com/r4",
            source_category="website",
        ),
    ]

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme updates",
        company_id="acme_corp",
        workspace_id="ws_default",
        evidence=chunks,
    )

    assert len(events) == 4

    # 1. Explicit
    e_explicit = next(e for e in events if e.event_type == IntelligenceEventType.RELEASE)
    assert e_explicit.event_time.occurred_at == datetime(2026, 8, 15, tzinfo=timezone.utc)
    assert e_explicit.event_time.precision == TimePrecision.DATE_ONLY

    # 2. Coarse
    e_coarse = next(e for e in events if e.event_type == IntelligenceEventType.PRODUCT_LAUNCH)
    assert e_coarse.event_time.precision == TimePrecision.QUARTER_YEAR
    assert e_coarse.event_time.approximate_period == "Q3 2026"

    # 3. Relative
    e_rel = next(e for e in events if e.event_type == IntelligenceEventType.HIRING_OR_HEADCOUNT)
    assert e_rel.event_time.precision == TimePrecision.APPROXIMATE
    assert e_rel.event_time.approximate_period == "yesterday"

    # 4. Unknown
    e_unk = next(e for e in events if e.event_type == IntelligenceEventType.PARTNERSHIP)
    assert e_unk.event_time.occurred_at is None
    assert e_unk.event_time.precision == TimePrecision.UNKNOWN


@pytest.mark.anyio
async def test_external_review_opinion(analyzer: RetrievedEvidenceAnalyzer) -> None:
    """Verify 3rd-party user review / external opinion extraction."""
    chunks = [
        RetrievalResult(
            chunk_id="chunk_review",
            score=0.80,
            text="Acme Dashboard review: excellent querying performance but UI is sluggish.",
            document_id="doc_g2",
            version_id="ver_01",
            source_id="src_g2",
            canonical_url="https://g2.com/products/acme/reviews",
            source_category="review",
            metadata={"source_perspective": "external_opinion"},
        )
    ]

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme reviews",
        company_id="acme_corp",
        workspace_id="ws_default",
        evidence=chunks,
    )

    assert len(events) == 1
    event = events[0]
    assert event.event_type == IntelligenceEventType.EXTERNAL_OPINION
    assert event.primary_perspective == SourcePerspective.EXTERNAL_OPINION
    assert event.confidence.level == ConfidenceLevel.MEDIUM


@pytest.mark.anyio
async def test_custom_evidence_interpreter_injection() -> None:
    """Verify dependency injection of custom BaseEvidenceInterpreter implementations."""

    class MockCustomInterpreter(BaseEvidenceInterpreter):
        async def interpret_evidence(
            self,
            query: str,
            company_id: str,
            workspace_id: str,
            evidence: list[RetrievalResult],
            context_metadata: dict[str, str] | None = None,
        ) -> list[RawEventPayload]:
            return [
                RawEventPayload(
                    event_type=IntelligenceEventType.STRATEGIC_ANNOUNCEMENT,
                    title="Mock Search Finding",
                    summary="Custom mock analyzer detected finding.",
                    subject="Strategy",
                    confidence_score=0.99,
                    confidence_level=ConfidenceLevel.VERY_HIGH,
                    evidence_snippets=[
                        RetrievedEvidenceSnippet(
                            chunk_id=evidence[0].chunk_id,
                            document_id=evidence[0].document_id,
                            version_id=evidence[0].version_id,
                            extracted_text=evidence[0].text,
                        )
                    ],
                )
            ]

    custom_analyzer = RetrievedEvidenceAnalyzer(evidence_interpreter=MockCustomInterpreter())
    chunk = RetrievalResult(
        chunk_id="chunk_c1",
        score=0.90,
        text="Custom text",
        document_id="doc_c1",
        version_id="ver_c1",
    )

    events = await custom_analyzer.analyze_retrieved_evidence(
        query="custom query",
        company_id="acme",
        workspace_id="ws_1",
        evidence=[chunk],
    )

    assert len(events) == 1
    assert events[0].event_type == IntelligenceEventType.STRATEGIC_ANNOUNCEMENT
    assert events[0].evidence[0].chunk_id == "chunk_c1"
