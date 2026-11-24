"""Unit and integration tests for Retrieval-to-Intelligence subsystem integration.

Verifies:
- Scenario A: Retrieved evidence -> event extraction (pricing, launches, releases)
- Scenario B: End-to-end structural provenance preservation (Insight -> Signal -> Event -> Evidence -> Chunk -> Version -> Doc -> Source)
- Scenario C: Event -> deterministic signal inference (repeated change, reversal, trend)
- Scenario D: Signal -> strategic insight synthesis with transitive audit trail
- Scenario E: Unsupported extraction rejection (LLM citing invalid chunk IDs is rejected)
- Scenario F: Insufficient evidence safety (generic background text -> 0 events, 0 signals, 0 insights)
- Scenario G: Deduplication across overlapping multi-hop chunks via DeterministicEventSignatureStrategy
- Scenario H: Contradictory evidence handling (CONTRADICTION signal -> UNCERTAINTY insight)
- Scenario I: Missing provenance validation rejection
- Scenario J: LLM failure observable fallback (fallback_used=True, fallback_reason recorded)
- Scenario K: End-to-end SelectiveAgenticRetriever -> Intelligence pipeline execution
"""

from datetime import datetime, timezone
from typing import Any
import pytest

from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.selective import (
    DeterministicEscalationPolicy,
    SelectiveAgenticRetriever,
    SelectiveRetrieverConfig,
)
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.agentic_rag.llm.errors import LLMUnavailableError
from competitive_intelligence.intelligence import (
    ConfidenceAssessment,
    ConfidenceLevel,
    DeterministicInsightValidator,
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
    LLMExtractedEventItem,
    LLMExtractionResponse,
    LLMEvidenceInterpreter,
    RetrievedEvidenceAnalyzer,
    RuleBasedEvidenceInterpreter,
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
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
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




# ==============================================================================
# Helper Mock Retriever
# ==============================================================================


class StaticMockRetriever(BaseRetriever):
    """Simple test retriever returning preconfigured RetrievalResult items."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return self.results[:query.top_k]


# ==============================================================================
# Test Fixtures & Data Builders
# ==============================================================================


def create_provenance_lineage(
    workspace_id: str = "ws_test",
    company_id: str = "acme_corp",
    source_id: str = "src_pricing_01",
    document_id: str = "doc_pricing_01",
    version_id: str = "ver_pricing_01",
    chunk_id: str = "chunk_pricing_01",
    source_url: str = "https://acme.com/pricing",
    content: str = "Acme updated Enterprise Plan pricing from $50 to $30/month.",
    score: float = 0.95,
) -> tuple[Source, Document, DocumentVersion, DocumentChunk, RetrievalResult]:
    """Helper to build a complete 7-tier structural lineage for test assertions."""
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
        content_hash="hash_123",
        acquired_at=datetime.now(timezone.utc),
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
            collected_at=datetime.now(timezone.utc),
        ),

    )
    retrieval_result = RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=content,
        document_id=document_id,
        version_id=version_id,
        source_id=source_id,
        company_id=company_id,
        workspace_id=workspace_id,
        source_category="pricing",
        canonical_url=source_url,
        metadata={
            "selective_initial_strategy": "adaptive",
            "selective_escalated": True,
            "selective_reasons": ["pricing_investigation"],
            "selective_final_source": "agentic",
        },
    )
    return source, doc, version, chunk, retrieval_result


# ==============================================================================
# Scenario A: Retrieved Evidence -> Event Extraction
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_a_retrieved_evidence_to_pricing_event() -> None:
    """Verify conversion of retrieved pricing change evidence into a structured IntelligenceEvent."""
    _, _, _, _, ret_res = create_provenance_lineage(
        content="Acme updated Enterprise Plan pricing from $50 to $30/month on August 10, 2026."
    )
    analyzer = RetrievedEvidenceAnalyzer()

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme pricing updates",
        company_id="acme_corp",
        workspace_id="ws_test",
        evidence=[ret_res],
    )

    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == IntelligenceEventType.PRICING_CHANGE
    assert ev.subject == "Enterprise Plan"
    assert ev.previous_state == {"price_usd": "$50"}
    assert ev.new_state == {"price_usd": "$30"}
    assert len(ev.evidence) == 1
    assert ev.evidence[0].chunk_id == "chunk_pricing_01"
    assert ev.evidence[0].document_id == "doc_pricing_01"
    assert ev.evidence[0].version_id == "ver_pricing_01"
    assert ev.evidence[0].source_id == "src_pricing_01"
    assert ev.evidence[0].metadata["selective_escalated"] is True


# ==============================================================================
# Scenario B: End-to-End Structural Provenance Preservation
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_b_end_to_end_structural_provenance_chain() -> None:
    """Verify complete 7-tier structural provenance: Insight -> Signal -> Event -> EvidenceRef -> Chunk -> Version -> Doc -> Source."""
    source, doc, version, chunk, ret_res1 = create_provenance_lineage(
        source_id="src_p1",
        document_id="doc_p1",
        version_id="ver_p1",
        chunk_id="chunk_p1",
        content="Acme updated Enterprise Plan pricing from $50 to $40/month on August 01, 2026.",
    )
    _, _, _, _, ret_res2 = create_provenance_lineage(
        source_id="src_p2",
        document_id="doc_p2",
        version_id="ver_p2",
        chunk_id="chunk_p2",
        content="Acme updated Enterprise Plan pricing from $40 to $30/month on August 15, 2026.",
    )

    # 1. Evidence -> Events
    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Acme pricing history",
        company_id="acme_corp",
        workspace_id="ws_test",
        evidence=[ret_res1, ret_res2],
    )
    assert len(events) == 2

    # 2. Events -> Signals
    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_test",
        company_id="acme_corp",
        events=events,
    )
    assert len(signals) >= 1
    signal = signals[0]

    # 3. Signals -> Insights
    strategic_analyzer = DeterministicStrategicAnalyzer()
    insights = await strategic_analyzer.generate_insights(
        workspace_id="ws_test",
        company_id="acme_corp",
        signals=signals,
        events=events,
    )
    assert len(insights) >= 1
    insight = insights[0]

    # Structural Provenance Verification
    # Insight -> Signal
    assert signal.signal_id in insight.supporting_signal_ids

    # Insight -> Event
    for eid in insight.supporting_event_ids:
        assert eid in [e.event_id for e in events]

    # Signal -> Event
    for eid in signal.supporting_event_ids:
        assert eid in [e.event_id for e in events]

    # Event -> EvidenceReference
    all_evidence_refs = [ref for ev in events for ref in ev.evidence]
    evidence_ids = {ref.evidence_id for ref in all_evidence_refs}
    for ev_id in insight.supporting_evidence_ids:
        assert ev_id in evidence_ids

    # EvidenceReference -> DocumentChunk -> DocumentVersion -> Document -> Source
    ref1 = next(r for r in all_evidence_refs if r.chunk_id == chunk.chunk_id)
    assert ref1.chunk_id == chunk.chunk_id
    assert ref1.version_id == version.version_id
    assert ref1.document_id == doc.document_id
    assert ref1.source_id == source.source_id
    assert ref1.source_url == source.location.url_or_identifier


# ==============================================================================
# Scenario C: Event -> Deterministic Signal Detection
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_c_deterministic_signal_detection_reversal() -> None:
    """Verify that state transition sequences trigger deterministic REVERSAL signals."""
    _, _, _, _, ret1 = create_provenance_lineage(
        chunk_id="chunk_r1",
        content="Acme updated Pro Plan pricing from $20 to $30 on July 01, 2026.",
    )
    _, _, _, _, ret2 = create_provenance_lineage(
        chunk_id="chunk_r2",
        content="Acme reverted Pro Plan pricing from $30 back to $20 on August 01, 2026.",
    )

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Acme Pro pricing",
        company_id="acme_corp",
        workspace_id="ws_test",
        evidence=[ret1, ret2],
    )
    assert len(events) == 2

    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_test",
        company_id="acme_corp",
        events=events,
    )

    signal_types = {s.signal_type for s in signals}
    assert SignalType.REVERSAL in signal_types or SignalType.REPEATED_CHANGE in signal_types
    reversal_sig = next(s for s in signals if s.signal_type == SignalType.REVERSAL)
    assert len(reversal_sig.supporting_event_ids) == 2


# ==============================================================================
# Scenario D: Signal -> Insight Grounding & Validation
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_d_signal_to_insight_grounding() -> None:
    """Verify that validated Insight requires supporting signals/events and rejects fabricated IDs."""
    ev = IntelligenceEvent(
        workspace_id="ws_test",
        company_id="acme_corp",
        event_type=IntelligenceEventType.PRICING_CHANGE,
        title="Pricing Update",
        summary="Adjusted pricing",
        subject="Enterprise Plan",
        event_time=EventTime(
            precision=TimePrecision.DATE_ONLY,
            acquired_at=datetime.now(timezone.utc),
            detected_at=datetime.now(timezone.utc),
        ),
        confidence=ConfidenceAssessment(score=0.90, level=ConfidenceLevel.HIGH),
        evidence=[
            EvidenceReference(
                source_id="src_1",
                document_id="doc_1",
                version_id="ver_1",
                chunk_id="chunk_1",
                source_url="https://acme.com",
                source_category="pricing",
                extracted_text="Pricing updated",
                acquired_at=datetime.now(timezone.utc),
            )
        ],
    )
    sig = InferredSignal(
        workspace_id="ws_test",
        company_id="acme_corp",
        signal_type=SignalType.REPEATED_CHANGE,
        subject="Enterprise Plan",
        title="Repeated Pricing Adjustments",
        summary="Observed multiple price adjustments.",
        supporting_event_ids=[ev.event_id],
        supporting_evidence_ids=[ev.evidence[0].evidence_id],
    )

    analyzer = DeterministicStrategicAnalyzer()
    insights = await analyzer.generate_insights(
        workspace_id="ws_test",
        company_id="acme_corp",
        signals=[sig],
        events=[ev],
    )

    assert len(insights) >= 1
    insight = insights[0]
    assert insight.insight_type == InsightType.PRICING_STRATEGY
    assert insight.factual_confidence.score >= 0.80
    assert insight.supporting_event_ids == [ev.event_id]


# ==============================================================================
# Scenario E: Unsupported / Hallucinated LLM Extraction Rejection
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_e_unsupported_extraction_rejection() -> None:
    """Verify that candidate LLM extractions citing invalid/hallucinated chunk IDs are rejected."""

    class HallucinatingLLMClient(BaseLLMClient):
        async def generate_structured(
            self,
            prompt: str,
            response_schema: type[Any],
            system_prompt: str | None = None,
            temperature: float = 0.0,
            budget_tracker: LLMBudgetTracker | None = None,
        ) -> Any:
            # LLM hallucinating chunk ID "chunk_fake_999" not in evidence
            return LLMExtractionResponse(
                events=[
                    LLMExtractedEventItem(
                        event_type=IntelligenceEventType.PRODUCT_LAUNCH,
                        title="Hallucinated Rocket Ship",
                        summary="Acme launched a rocket ship.",
                        subject="Rocket Ship",
                        confidence_score=0.99,
                        confidence_level=ConfidenceLevel.VERY_HIGH,
                        supporting_chunk_ids=["chunk_fake_999"],  # Invalid chunk ID!
                    )
                ]
            )

        async def generate_text(
            self,
            prompt: str,
            system_prompt: str | None = None,
            temperature: float = 0.0,
            budget_tracker: LLMBudgetTracker | None = None,
        ) -> str:
            return ""

    _, _, _, _, valid_chunk = create_provenance_lineage(
        chunk_id="chunk_real_01",
        content="Acme updated their office address in San Francisco.",
    )

    llm_interpreter = LLMEvidenceInterpreter(llm_client=HallucinatingLLMClient())
    analyzer = RetrievedEvidenceAnalyzer(evidence_interpreter=llm_interpreter)

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme products",
        company_id="acme_corp",
        workspace_id="ws_test",
        evidence=[valid_chunk],
    )

    # Hallucinated event must be rejected: 0 accepted events
    assert events == []
    diag = analyzer.last_diagnostics
    assert diag["candidate_events_count"] == 1
    assert diag["accepted_events_count"] == 0
    assert diag["rejected_events_count"] == 1
    assert any("cited no valid chunk IDs" in r for r in diag["rejection_reasons"])


# ==============================================================================
# Scenario F: Insufficient Evidence Handling
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_f_insufficient_evidence_safely_yields_zero_intelligence() -> None:
    """Verify that generic background text without factual events yields 0 events, 0 signals, 0 insights."""
    generic_chunk = RetrievalResult(
        chunk_id="chunk_bg_01",
        score=0.70,
        text="Acme is a software provider dedicated to excellence, innovation, and client satisfaction.",
        document_id="doc_about",
        version_id="ver_01",
        canonical_url="https://acme.com/about",
        source_category="website",
    )

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Acme updates",
        company_id="acme_corp",
        workspace_id="ws_test",
        evidence=[generic_chunk],
    )
    assert events == []

    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_test",
        company_id="acme_corp",
        events=events,
    )
    assert signals == []

    strategic_analyzer = DeterministicStrategicAnalyzer()
    insights = await strategic_analyzer.generate_insights(
        workspace_id="ws_test",
        company_id="acme_corp",
        signals=signals,
        events=events,
    )
    assert insights == []


# ==============================================================================
# Scenario G: Duplicate Evidence Deduplication via Deterministic Signature
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_g_duplicate_evidence_deduplicated_by_signature() -> None:
    """Verify that multiple identical or overlapping chunks from multi-hop retrieval deduplicate to 1 event."""
    _, _, _, _, chunk1 = create_provenance_lineage(
        chunk_id="hop1_chunk",
        source_id="src_blog",
        source_url="https://acme.com/blog/announcing-studio",
        content="Announcing Acme AI Studio for generative workflows on August 12, 2026.",
    )
    _, _, _, _, chunk2 = create_provenance_lineage(
        chunk_id="hop2_chunk",
        source_id="src_news",
        source_url="https://techcrunch.com/acme-studio",
        content="Announcing Acme AI Studio for generative workflows on August 12, 2026.",
    )

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Acme AI Studio launch",
        company_id="acme_corp",
        workspace_id="ws_test",
        evidence=[chunk1, chunk2],
    )

    # Clustered by DeterministicEventSignatureStrategy into 1 IntelligenceEvent with 2 EvidenceReferences
    assert len(events) == 1
    ev = events[0]
    assert ev.subject == "Acme AI Studio"
    assert len(ev.evidence) == 2
    assert {ref.chunk_id for ref in ev.evidence} == {"hop1_chunk", "hop2_chunk"}


# ==============================================================================
# Scenario H: Contradictory Evidence Handling
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_h_contradictory_evidence_creates_uncertainty_insight() -> None:
    """Verify that contradictory evidence reports create a CONTRADICTION signal and UNCERTAINTY insight."""
    _, _, _, _, launch_chunk = create_provenance_lineage(
        chunk_id="chunk_l",
        content="Announcing Voice Mode for all enterprise customers.",
    )
    _, _, _, _, delay_chunk = create_provenance_lineage(
        chunk_id="chunk_d",
        content="Internal reports confirm rollout delayed for Voice Mode pending review.",
    )

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Voice Mode status",
        company_id="acme_corp",
        workspace_id="ws_test",
        evidence=[launch_chunk, delay_chunk],
    )
    assert len(events) == 2

    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_test",
        company_id="acme_corp",
        events=events,
    )
    assert any(s.signal_type == SignalType.CONTRADICTION for s in signals)
    contra_sig = next(s for s in signals if s.signal_type == SignalType.CONTRADICTION)

    strategic_analyzer = DeterministicStrategicAnalyzer()
    insights = await strategic_analyzer.generate_insights(
        workspace_id="ws_test",
        company_id="acme_corp",
        signals=[contra_sig],
        events=events,
    )
    assert len(insights) >= 1
    uncertainty_ins = next(i for i in insights if i.insight_type == InsightType.UNCERTAINTY)
    assert uncertainty_ins.status == InsightStatus.UNCERTAIN


# ==============================================================================
# Scenario I: Missing Provenance Validation Rejection
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_i_missing_provenance_validation_rejection() -> None:
    """Verify that an Insight referencing non-existent event IDs is rejected by DeterministicInsightValidator."""
    validator = DeterministicInsightValidator()
    from competitive_intelligence.intelligence.insights.models import RawInsightPayload

    unsupported_payload = RawInsightPayload(
        insight_type=InsightType.PRICING_STRATEGY,
        title="Unsubstantiated Pricing Shift",
        summary="Summary of fake shift.",
        statement="Detailed fake statement.",
        supporting_signal_ids=[],
        supporting_event_ids=["non_existent_event_id_999"],  # Missing from events pool!
    )

    is_valid, errors = validator.validate(
        payload=unsupported_payload,
        signals=[],
        events=[],
    )
    assert is_valid is False
    assert any("does not exist in the candidate events pool" in err for err in errors)


# ==============================================================================
# Scenario J: LLM Failure Observable Fallback
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_j_llm_failure_observable_fallback() -> None:
    """Verify that LLM failure triggers observable fallback with fallback_used=True and diagnostic recording."""

    class FailingLLMClient(BaseLLMClient):
        async def generate_structured(
            self,
            prompt: str,
            response_schema: type[Any],
            system_prompt: str | None = None,
            temperature: float = 0.0,
            budget_tracker: LLMBudgetTracker | None = None,
        ) -> Any:
            raise LLMUnavailableError("OpenAI API endpoint connection timed out.")

        async def generate_text(
            self,
            prompt: str,
            system_prompt: str | None = None,
            temperature: float = 0.0,
            budget_tracker: LLMBudgetTracker | None = None,
        ) -> str:
            raise LLMUnavailableError("OpenAI API endpoint connection timed out.")

    _, _, _, _, chunk = create_provenance_lineage(
        content="Acme acquired VectorDB Inc for $50M on August 18, 2026."
    )

    llm_interpreter = LLMEvidenceInterpreter(llm_client=FailingLLMClient())
    analyzer = RetrievedEvidenceAnalyzer(evidence_interpreter=llm_interpreter)

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme acquisitions",
        company_id="acme_corp",
        workspace_id="ws_test",
        evidence=[chunk],
    )

    # Fallback must extract the acquisition successfully
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == IntelligenceEventType.ACQUISITION
    assert ev.subject == "VectorDB Inc"

    # Fallback must be explicitly observable in metadata and diagnostics
    assert ev.metadata.get("fallback_used") is True
    assert "LLMUnavailableError" in ev.metadata.get("fallback_reason", "")
    assert "[Fallback: LLMUnavailableError]" in (ev.confidence.rationale or "")

    diag = analyzer.last_diagnostics
    assert diag["fallback_used"] is True
    assert "LLMUnavailableError" in str(diag["fallback_reason"])


# ==============================================================================
# Scenario K: End-to-End SelectiveAgenticRetriever -> Intelligence Pipeline
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_k_selective_agentic_retrieval_to_intelligence_pipeline() -> None:
    """Verify integration from SelectiveAgenticRetriever -> Retrieved Evidence -> Events -> Signals -> Insights."""
    _, _, _, _, ret_item1 = create_provenance_lineage(
        chunk_id="chunk_sel_01",
        content="Acme updated Enterprise Plan pricing from $100 to $80 on July 01, 2026.",
    )
    _, _, _, _, ret_item2 = create_provenance_lineage(
        chunk_id="chunk_sel_02",
        content="Acme updated Enterprise Plan pricing from $80 to $60 on August 01, 2026.",
    )

    adaptive_retriever = StaticMockRetriever([ret_item1])
    agentic_retriever = StaticMockRetriever([ret_item1, ret_item2])

    selective_retriever = SelectiveAgenticRetriever(
        adaptive_retriever=adaptive_retriever,
        agentic_retriever=agentic_retriever,
        config=SelectiveRetrieverConfig(),
    )

    # 1. Retrieval
    query = RetrievalQuery(
        query_text="Compare Pinecone and Acme pricing changes over time",
        metadata_filter=MetadataFilter(workspace_id="ws_test", company_ids=["acme_corp"]),
        top_k=5,
    )
    retrieved_results = await selective_retriever.retrieve(query)
    assert len(retrieved_results) >= 1

    # 2. Event Analysis
    evidence_analyzer = RetrievedEvidenceAnalyzer()
    events = await evidence_analyzer.analyze_retrieved_evidence(
        query=query.query_text,
        company_id="acme_corp",
        workspace_id="ws_test",
        evidence=retrieved_results,
    )
    assert len(events) >= 1

    # 3. Deterministic Signal Detection
    signal_detector = DeterministicSignalDetector()
    signals = await signal_detector.detect_signals(
        workspace_id="ws_test",
        company_id="acme_corp",
        events=events,
    )

    # 4. Strategic Insight Synthesis
    strategic_analyzer = DeterministicStrategicAnalyzer()
    insights = await strategic_analyzer.generate_insights(
        workspace_id="ws_test",
        company_id="acme_corp",
        signals=signals,
        events=events,
    )

    # Epistemic hierarchy validation
    for ins in insights:
        assert ins.supporting_event_ids
        for eid in ins.supporting_event_ids:
            matching_event = next(e for e in events if e.event_id == eid)
            assert matching_event.company_id == "acme_corp"
            assert len(matching_event.evidence) >= 1
