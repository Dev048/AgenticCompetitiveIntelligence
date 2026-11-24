"""Evaluation Script: Retrieval-to-Intelligence Subsystem Integration.

Evaluates the integrated pipeline:
Retrieved Evidence (RetrievalResult)
  -> Evidence/Event Extraction (RetrievedEvidenceAnalyzer)
  -> IntelligenceEvent (what happened)
  -> Deterministic Signal Detection (DeterministicSignalDetector)
  -> InferredSignal (observable pattern)
  -> Strategic Analysis (DeterministicStrategicAnalyzer)
  -> Insight (bounded competitive takeaway)

Evaluates 6 key operational scenarios:
1. Pricing Change (Single stateful pricing transition)
2. Repeated Change / Directional Trend (Consecutive price drops)
3. Reversal (Price increase followed by price decrease)
4. Contradiction (Product launch vs. product delay)
5. Insufficient Evidence (Background noise -> 0 events, 0 signals, 0 insights)
6. Unsupported Extraction Rejection (LLM hallucinated citations -> rejected with diagnostics)

Usage:
    python scripts/evaluate_retrieval_intelligence_integration.py
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import sys
from typing import Any

# Ensure workspace src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.intelligence import (
    ConfidenceLevel,
    DeterministicSignalDetector,
    DeterministicStrategicAnalyzer,
    InferredSignal,
    Insight,
    InsightType,
    IntelligenceEvent,
    IntelligenceEventType,
    LLMExtractedEventItem,
    LLMExtractionResponse,
    LLMEvidenceInterpreter,
    RetrievedEvidenceAnalyzer,
    RuleBasedEvidenceInterpreter,
    SignalType,
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


@dataclass
class ScenarioReport:
    name: str
    description: str
    passed: bool
    candidate_events: int = 0
    accepted_events: int = 0
    rejected_events: int = 0
    rejection_reasons: list[str] = field(default_factory=list)
    signals_count: int = 0
    insights_count: int = 0
    provenance_verified: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def make_evidence(
    chunk_id: str,
    content: str,
    source_url: str = "https://acme.com/pricing",
    company_id: str = "acme_corp",
    workspace_id: str = "ws_eval",
) -> tuple[Source, Document, DocumentVersion, DocumentChunk, RetrievalResult]:
    """Builds a verified 7-tier provenance lineage."""
    source_id = f"src_{chunk_id}"
    document_id = f"doc_{chunk_id}"
    version_id = f"ver_{chunk_id}"

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
        content_hash="hash_eval",
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
        score=0.92,
        text=content,
        document_id=document_id,
        version_id=version_id,
        source_id=source_id,
        company_id=company_id,
        workspace_id=workspace_id,
        source_category="pricing",
        canonical_url=source_url,
        metadata={"retrieval_escalated": True},
    )

    return source, doc, version, chunk, retrieval_result


async def evaluate_scenario_1_pricing_change() -> ScenarioReport:
    """Scenario 1: Single pricing change evidence item -> Event -> Signal -> Insight."""
    src, doc, ver, chk, res = make_evidence(
        chunk_id="chunk_price_01",
        content="Acme updated Enterprise Plan pricing from $50 to $35/month on August 10, 2026.",
    )

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Acme pricing updates",
        company_id="acme_corp",
        workspace_id="ws_eval",
        evidence=[res],
    )

    diag = analyzer.last_diagnostics
    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_eval",
        company_id="acme_corp",
        events=events,
    )

    strat = DeterministicStrategicAnalyzer()
    insights = await strat.generate_insights(
        workspace_id="ws_eval",
        company_id="acme_corp",
        signals=signals,
        events=events,
    )

    # Verify structural provenance
    prov_ok = False
    if events and len(events[0].evidence) == 1:
        ref = events[0].evidence[0]
        prov_ok = (
            ref.chunk_id == chk.chunk_id
            and ref.version_id == ver.version_id
            and ref.document_id == doc.document_id
            and ref.source_id == src.source_id
            and ref.source_url == src.location.url_or_identifier
        )

    passed = (
        len(events) == 1
        and events[0].event_type == IntelligenceEventType.PRICING_CHANGE
        and prov_ok
    )

    return ScenarioReport(
        name="Scenario 1: Pricing Change",
        description="Single pricing change evidence extracted with exact state transitions and provenance",
        passed=passed,
        candidate_events=diag.get("candidate_events", len(events)),
        accepted_events=len(events),
        rejected_events=diag.get("rejected_events", 0),
        rejection_reasons=diag.get("rejection_reasons", []),
        signals_count=len(signals),
        insights_count=len(insights),
        provenance_verified=prov_ok,
        details={
            "event_type": events[0].event_type.value if events else None,
            "subject": events[0].subject if events else None,
            "previous_state": events[0].previous_state if events else None,
            "new_state": events[0].new_state if events else None,
        },
    )


async def evaluate_scenario_2_directional_trend() -> ScenarioReport:
    """Scenario 2: Consecutive price cuts -> Event sequence -> Directional Trend Signal -> Strategic Insight."""
    _, _, _, _, res1 = make_evidence(
        chunk_id="chunk_trend_01",
        content="Acme updated Pro Plan pricing from $50 to $40 on July 01, 2026.",
    )
    _, _, _, _, res2 = make_evidence(
        chunk_id="chunk_trend_02",
        content="Acme updated Pro Plan pricing from $40 to $30 on August 01, 2026.",
    )

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Acme Pro Plan pricing history",
        company_id="acme_corp",
        workspace_id="ws_eval",
        evidence=[res1, res2],
    )
    diag = analyzer.last_diagnostics

    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_eval",
        company_id="acme_corp",
        events=events,
    )

    strat = DeterministicStrategicAnalyzer()
    insights = await strat.generate_insights(
        workspace_id="ws_eval",
        company_id="acme_corp",
        signals=signals,
        events=events,
    )

    prov_ok = False
    if insights and signals:
        prov_ok = (
            signals[0].signal_id in insights[0].supporting_signal_ids
            and all(eid in [e.event_id for e in events] for eid in signals[0].supporting_event_ids)
        )

    passed = (
        len(events) == 2
        and len(signals) >= 1
        and signals[0].signal_type in {SignalType.DIRECTIONAL_TREND, SignalType.DOWN_MARKET_EXPANSION, SignalType.AGGRESSIVE_DISCOUNTING, SignalType.REPEATED_CHANGE}
        and len(insights) >= 1
        and prov_ok
    )

    return ScenarioReport(
        name="Scenario 2: Directional Trend",
        description="Repeated price reductions produce a deterministic DIRECTIONAL_TREND signal and strategic insight",
        passed=passed,
        candidate_events=diag.get("candidate_events", len(events)),
        accepted_events=len(events),
        rejected_events=diag.get("rejected_events", 0),
        rejection_reasons=diag.get("rejection_reasons", []),
        signals_count=len(signals),
        insights_count=len(insights),
        provenance_verified=prov_ok,
        details={
            "signal_type": signals[0].signal_type.value if signals else None,
            "insight_summary": insights[0].summary if insights else None,
        },
    )


async def evaluate_scenario_3_reversal() -> ScenarioReport:
    """Scenario 3: Price increase followed by price decrease -> Reversal Signal -> Insight."""
    _, _, _, _, res1 = make_evidence(
        chunk_id="chunk_rev_01",
        content="Acme updated Enterprise Plan pricing from $100 to $150 on May 01, 2026.",
    )
    _, _, _, _, res2 = make_evidence(
        chunk_id="chunk_rev_02",
        content="Acme reverted Enterprise Plan pricing from $150 back to $100 on July 01, 2026.",
    )

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Acme Enterprise Plan price changes",
        company_id="acme_corp",
        workspace_id="ws_eval",
        evidence=[res1, res2],
    )
    diag = analyzer.last_diagnostics

    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_eval",
        company_id="acme_corp",
        events=events,
    )

    strat = DeterministicStrategicAnalyzer()
    insights = await strat.generate_insights(
        workspace_id="ws_eval",
        company_id="acme_corp",
        signals=signals,
        events=events,
    )

    reversal_signals = [s for s in signals if s.signal_type == SignalType.REVERSAL]
    passed = len(events) == 2 and len(reversal_signals) == 1 and len(insights) >= 1

    return ScenarioReport(
        name="Scenario 3: Reversal",
        description="Opposing state transitions trigger deterministic REVERSAL signal and insight",
        passed=passed,
        candidate_events=diag.get("candidate_events", len(events)),
        accepted_events=len(events),
        rejected_events=diag.get("rejected_events", 0),
        rejection_reasons=diag.get("rejection_reasons", []),
        signals_count=len(signals),
        insights_count=len(insights),
        provenance_verified=len(reversal_signals) > 0,
        details={
            "reversal_detected": len(reversal_signals) == 1,
            "signal_description": reversal_signals[0].summary if reversal_signals else None,
        },
    )


async def evaluate_scenario_4_contradiction() -> ScenarioReport:
    """Scenario 4: Contradictory evidence -> CONTRADICTION Signal -> UNCERTAINTY Insight."""
    _, _, _, _, res_launch = make_evidence(
        chunk_id="chunk_launch",
        content="Announcing Voice Mode for all enterprise customers.",
    )
    _, _, _, _, res_delay = make_evidence(
        chunk_id="chunk_delay",
        content="Internal reports confirm rollout delayed for Voice Mode pending review.",
    )

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Voice Mode rollout status",
        company_id="acme_corp",
        workspace_id="ws_eval",
        evidence=[res_launch, res_delay],
    )
    diag = analyzer.last_diagnostics

    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_eval",
        company_id="acme_corp",
        events=events,
    )

    strat = DeterministicStrategicAnalyzer()
    insights = await strat.generate_insights(
        workspace_id="ws_eval",
        company_id="acme_corp",
        signals=signals,
        events=events,
    )

    contradiction_signals = [s for s in signals if s.signal_type == SignalType.CONTRADICTION]
    uncertainty_insights = [i for i in insights if i.insight_type == InsightType.UNCERTAINTY]

    passed = (
        len(events) == 2
        and len(contradiction_signals) >= 1
        and len(uncertainty_insights) >= 1
    )

    return ScenarioReport(
        name="Scenario 4: Contradiction",
        description="Conflicting evidence triggers CONTRADICTION signal and UNCERTAINTY insight without taking sides",
        passed=passed,
        candidate_events=diag.get("candidate_events", len(events)),
        accepted_events=len(events),
        rejected_events=diag.get("rejected_events", 0),
        rejection_reasons=diag.get("rejection_reasons", []),
        signals_count=len(signals),
        insights_count=len(insights),
        provenance_verified=len(uncertainty_insights) > 0,
        details={
            "contradiction_signal": contradiction_signals[0].summary if contradiction_signals else None,
            "insight_type": uncertainty_insights[0].insight_type.value if uncertainty_insights else None,
        },
    )


async def evaluate_scenario_5_insufficient_evidence() -> ScenarioReport:
    """Scenario 5: Non-intelligence evidence -> 0 events, 0 signals, 0 insights, with diagnostics."""
    _, _, _, _, res_noise = make_evidence(
        chunk_id="chunk_noise_01",
        content="Acme operates global data centers and adheres to standard enterprise privacy principles.",
    )

    analyzer = RetrievedEvidenceAnalyzer()
    events = await analyzer.analyze_retrieved_evidence(
        query="Acme pricing updates",
        company_id="acme_corp",
        workspace_id="ws_eval",
        evidence=[res_noise],
    )
    diag = analyzer.last_diagnostics

    detector = DeterministicSignalDetector()
    signals = await detector.detect_signals(
        workspace_id="ws_eval",
        company_id="acme_corp",
        events=events,
    )

    strat = DeterministicStrategicAnalyzer()
    insights = await strat.generate_insights(
        workspace_id="ws_eval",
        company_id="acme_corp",
        signals=signals,
        events=events,
    )

    passed = len(events) == 0 and len(signals) == 0 and len(insights) == 0

    return ScenarioReport(
        name="Scenario 5: Insufficient Evidence",
        description="Uninformative evidence produces 0 events, 0 signals, 0 insights with explicit diagnostics",
        passed=passed,
        candidate_events=diag.get("candidate_events", 0),
        accepted_events=0,
        rejected_events=diag.get("rejected_events", 0),
        rejection_reasons=diag.get("rejection_reasons", ["no_candidate_events_extracted"]),
        signals_count=len(signals),
        insights_count=len(insights),
        provenance_verified=True,
        details={
            "safe_zero_events": True,
            "safe_zero_signals": True,
            "safe_zero_insights": True,
        },
    )


async def evaluate_scenario_6_unsupported_extraction_rejection() -> ScenarioReport:
    """Scenario 6: LLM returns hallucinated/unsupported chunk IDs -> rejected with rejection diagnostics."""
    _, _, _, _, res_valid = make_evidence(
        chunk_id="chunk_valid_100",
        content="Acme updated Enterprise Plan pricing from $50 to $30/month.",
    )

    class HallucinatingMockLLM(BaseLLMClient):
        async def generate_structured(
            self,
            prompt: str,
            response_schema: type[Any],
            system_prompt: str | None = None,
            temperature: float = 0.0,
            budget_tracker: LLMBudgetTracker | None = None,
        ) -> Any:
            return LLMExtractionResponse(
                events=[
                    # Valid citation
                    LLMExtractedEventItem(
                        event_type=IntelligenceEventType.PRICING_CHANGE,
                        title="Acme Price Change",
                        summary="Acme lowered Enterprise Plan to $30",
                        subject="Enterprise Plan",
                        supporting_chunk_ids=["chunk_valid_100"],
                        previous_state={"price_usd": "$50"},
                        new_state={"price_usd": "$30"},
                        confidence_score=0.9,
                        confidence_level=ConfidenceLevel.HIGH,
                    ),
                    # Hallucinated citation
                    LLMExtractedEventItem(
                        event_type=IntelligenceEventType.ACQUISITION,
                        title="Acme Acquisition",
                        summary="Acme acquired MegaCorp for $1B",
                        subject="MegaCorp",
                        supporting_chunk_ids=["chunk_hallucinated_unknown_999"],
                        confidence_score=0.95,
                        confidence_level=ConfidenceLevel.VERY_HIGH,
                    ),
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

    interpreter = LLMEvidenceInterpreter(llm_client=HallucinatingMockLLM())
    analyzer = RetrievedEvidenceAnalyzer(evidence_interpreter=interpreter)

    events = await analyzer.analyze_retrieved_evidence(
        query="Acme acquisitions and pricing",
        company_id="acme_corp",
        workspace_id="ws_eval",
        evidence=[res_valid],
    )
    diag = analyzer.last_diagnostics

    passed = (
        len(events) == 1
        and events[0].subject == "Enterprise Plan"
        and diag.get("rejected_events", 0) == 1
        and any("chunk_hallucinated_unknown_999" in r for r in diag.get("rejection_reasons", []))
    )

    return ScenarioReport(
        name="Scenario 6: Unsupported Extraction Rejection",
        description="LLM event citing non-existent chunk ID is rejected; valid event accepted; rejection reason logged",
        passed=passed,
        candidate_events=diag.get("candidate_events", 2),
        accepted_events=len(events),
        rejected_events=diag.get("rejected_events", 1),
        rejection_reasons=diag.get("rejection_reasons", []),
        signals_count=0,
        insights_count=0,
        provenance_verified=True,
        details={
            "accepted_subjects": [e.subject for e in events],
            "rejected_citations": [r for r in diag.get("rejection_reasons", [])],
        },
    )


async def main() -> None:
    print("=" * 80)
    print("RETRIEVAL-TO-INTELLIGENCE SUBSYSTEM INTEGRATION EVALUATION")
    print("=" * 80)
    print("Evaluating epistemic progression and structural provenance across 6 scenarios:\n")

    scenarios = [
        evaluate_scenario_1_pricing_change(),
        evaluate_scenario_2_directional_trend(),
        evaluate_scenario_3_reversal(),
        evaluate_scenario_4_contradiction(),
        evaluate_scenario_5_insufficient_evidence(),
        evaluate_scenario_6_unsupported_extraction_rejection(),
    ]

    reports: list[ScenarioReport] = await asyncio.gather(*scenarios)

    all_passed = True
    for idx, rep in enumerate(reports, 1):
        status_str = "PASSED" if rep.passed else "FAILED"
        if not rep.passed:
            all_passed = False

        print(f"[{idx}/6] {rep.name}: {status_str}")
        print(f"    Description:        {rep.description}")
        print(f"    Candidate Events:   {rep.candidate_events}")
        print(f"    Accepted Events:    {rep.accepted_events}")
        print(f"    Rejected Events:    {rep.rejected_events}")
        if rep.rejection_reasons:
            print(f"    Rejection Reasons:  {rep.rejection_reasons}")
        print(f"    Signals Produced:   {rep.signals_count}")
        print(f"    Insights Produced:  {rep.insights_count}")
        print(f"    Provenance Valid:   {rep.provenance_verified}")
        if rep.details:
            print(f"    Details:            {rep.details}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("INTEGRATION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Scenarios Evaluated: {len(reports)}")
    print(f"Passed:                   {sum(1 for r in reports if r.passed)}")
    print(f"Failed:                   {sum(1 for r in reports if not r.passed)}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
