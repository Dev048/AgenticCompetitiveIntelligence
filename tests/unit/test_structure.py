"""Structural and contractual unit tests verifying subsystem integrity and domain invariants."""

from datetime import datetime, timezone

import pytest

from competitive_intelligence.agentic_rag import (
    AgenticRAGState,
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.application import (
    TrackedCompetitor,
    User,
    Workspace,
)
from competitive_intelligence.core import (
    BudgetExceededError,
    LLMBudgetTracker,
    Settings,
)
from competitive_intelligence.intelligence import (
    Alert,
    DetectedEvent,
    EventCategory,
    GeneratedInsight,
    InferredSignal,
    IntelligenceReport,
    RawEvidence,
    SeverityLevel,
    SignalType,
)
from competitive_intelligence.knowledge_pipeline import (
    ChunkingStrategy,
    ChunkMetadata,
    DocumentChunk,
    IngestionJob,
    IngestionStatus,
)
from competitive_intelligence.retrieval_engine import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    ScoredDocumentChunk,
)
from competitive_intelligence.source_acquisition import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    ContentType,
    GovernanceState,
    LifecycleState,
    LifecycleStatus,
    MonitoringPolicy,
    PlatformType,
    Source,
    SourceCategory,
    SourceClassification,
    SourceGovernance,
    SourceIdentity,
    SourceLifecycle,
    SourceLocation,
    SourceReliability,
    SourceTier,
    ToolChain,
    ToSStatus,
)


def test_core_budget_tracker_enforces_limits() -> None:
    """Test that LLMBudgetTracker limits calls, tokens, and retrieval hops."""
    tracker = LLMBudgetTracker(
        task_id="test-task-1",
        max_llm_calls=2,
        max_retrieval_hops=1,
        max_tokens=1000,
    )

    # Valid step
    tracker.record_llm_call(prompt_tokens=200, completion_tokens=100)
    tracker.record_retrieval_hop()

    assert tracker.usage.llm_calls == 1
    assert tracker.usage.total_tokens == 300
    assert tracker.usage.retrieval_hops == 1

    # Second valid call
    tracker.record_llm_call(prompt_tokens=200, completion_tokens=100)
    assert tracker.usage.llm_calls == 2

    # Exceeding LLM call limit
    with pytest.raises(BudgetExceededError, match="exceeded max LLM calls limit"):
        tracker.record_llm_call(prompt_tokens=10, completion_tokens=10)


def test_source_acquisition_models_instantiation() -> None:
    """Test that Source instantiates with correct invariants and defaults."""
    source = Source(
        identity=SourceIdentity(
            company_id="comp_123",
            workspace_id="ws_456",
            display_name="Competitor Pricing",
        ),
        classification=SourceClassification(
            category=SourceCategory.PRICING,
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier="https://competitor.com/pricing",
            access_pattern=AccessPattern.PUBLIC_WEB,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
        ),
    )
    assert source.source_id is not None
    assert source.classification.category == SourceCategory.PRICING
    assert source.governance.terms_of_service_status == ToSStatus.UNKNOWN
    assert source.reliability.historical_accuracy is None
    assert source.lifecycle.status == LifecycleStatus.ACTIVE


def test_provenance_chain_across_subsystems() -> None:
    """Verify that DocumentChunk, RawEvidence, Event, Signal, and Insight connect via IDs."""
    source_id = "src_1"
    doc_id = "doc_1"
    version_id = "ver_1"
    timestamp = datetime.now(timezone.utc)

    chunk_meta = ChunkMetadata(
        source_id=source_id,
        document_id=doc_id,
        version_id=version_id,
        company_id="comp_1",
        workspace_id="ws_1",
        source_url="https://competitor.com/pricing",
        source_category="pricing",
        collected_at=timestamp,
    )

    chunk = DocumentChunk(
        content="Enterprise tier now includes SSO and audit logs.",
        metadata=chunk_meta,
        content_hash="hash_123",
    )

    evidence = RawEvidence(
        chunk_id=chunk.chunk_id,
        source_id=source_id,
        source_url=chunk_meta.source_url,
        company_id=chunk_meta.company_id,
        extracted_text=chunk.content,
        timestamp=chunk_meta.collected_at,
    )

    event = DetectedEvent(
        company_id="comp_1",
        category=EventCategory.FEATURE_UPDATE,
        title="Enterprise SSO Added",
        summary="Competitor added SSO to enterprise plan",
        evidence_ids=[evidence.evidence_id],
    )

    signal = InferredSignal(
        company_ids=["comp_1"],
        signal_type=SignalType.ENTERPRISE_EXPANSION,
        summary="Aggressive push upmarket to enterprise customers",
        supporting_event_ids=[event.event_id],
        supporting_evidence_ids=[evidence.evidence_id],
    )

    insight = GeneratedInsight(
        workspace_id="ws_1",
        title="Competitor Enterprise Pivot",
        narrative="Evidence shows rapid enterprise feature rollout.",
        supporting_signal_ids=[signal.signal_id],
        supporting_evidence_ids=[evidence.evidence_id],
    )

    assert chunk.metadata.source_id == source_id
    assert evidence.chunk_id == chunk.chunk_id
    assert event.evidence_ids[0] == evidence.evidence_id
    assert signal.supporting_event_ids[0] == event.event_id
    assert insight.supporting_signal_ids[0] == signal.signal_id
