"""Strategic Analyzer and reference RuleBasedStrategicInterpreter components."""

from datetime import datetime, timezone
import re
from typing import Any

from competitive_intelligence.intelligence.insights.interfaces import (
    BaseInsightValidator,
    BaseStrategicAnalyzer,
    BaseStrategicInterpreter,
)
from competitive_intelligence.intelligence.insights.models import (
    RawInsightPayload,
    StrategicAnalysisPolicy,
)
from competitive_intelligence.intelligence.insights.validator import (
    DeterministicInsightValidator,
)
from competitive_intelligence.intelligence.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    InferredSignal,
    Insight,
    InsightStatus,
    InsightType,
    IntelligenceEvent,
    IntelligenceEventType,
    InterpretiveConfidenceAssessment,
    SignificanceAssessment,
    SignificanceLevel,
    SignificanceStatus,
    SignalType,
    TemporalScope,
)


class RuleBasedStrategicInterpreter(BaseStrategicInterpreter):
    """Reference deterministic interpreter implementing BaseStrategicInterpreter.

    Synthesizes bounded, evidence-grounded strategic interpretation statements.
    Strictly avoids mind-reading, ungrounded motives, or speculative business recommendations.
    """

    def __init__(self, policy: StrategicAnalysisPolicy | None = None) -> None:
        self.policy = policy or StrategicAnalysisPolicy()

    async def interpret(
        self,
        workspace_id: str,
        company_id: str,
        signals: list[InferredSignal],
        events: list[IntelligenceEvent],
        context_metadata: dict[str, Any] | None = None,
    ) -> list[RawInsightPayload]:
        candidate_payloads: list[RawInsightPayload] = []

        # 1. Evaluate individual signals
        for sig in signals:
            event_ids = list(sig.supporting_event_ids)
            if not event_ids:
                continue

            # A. Contradictions & Disputed Reports
            if sig.signal_type == SignalType.CONTRADICTION:
                candidate_payloads.append(
                    RawInsightPayload(
                        insight_type=InsightType.UNCERTAINTY,
                        title=f"Market Uncertainty regarding {sig.subject}",
                        summary=f"Conflicting reports observed regarding {sig.subject}.",
                        statement=f"Conflicting reports exist regarding the status and timeline of {sig.subject}.",
                        supporting_signal_ids=[sig.signal_id],
                        supporting_event_ids=event_ids,
                        interpretive_confidence_score=0.50,
                        interpretive_confidence_level=ConfidenceLevel.LOW,
                        interpretive_rationale="Signal contains conflicting or contradictory evidence.",
                        status=InsightStatus.UNCERTAIN,
                    )
                )
                continue

            # B. Pricing Strategy (Reversal, Directional Trend, Repeated Change)
            if sig.signal_type in (
                SignalType.REVERSAL,
                SignalType.DIRECTIONAL_TREND,
                SignalType.REPEATED_CHANGE,
            ) and self._is_pricing_subject(sig.subject, events):
                trend_desc = "reversal" if sig.signal_type == SignalType.REVERSAL else "directional progression"
                candidate_payloads.append(
                    RawInsightPayload(
                        insight_type=InsightType.PRICING_STRATEGY,
                        title=f"Pricing Strategy for {sig.subject}",
                        summary=f"Pricing for {sig.subject} underwent repeated adjustments.",
                        statement=f"Enterprise pricing for {sig.subject} underwent repeated adjustments ({trend_desc}) over the observed period.",
                        supporting_signal_ids=[sig.signal_id],
                        supporting_event_ids=event_ids,
                        interpretive_confidence_score=0.85,
                        interpretive_confidence_level=ConfidenceLevel.HIGH,
                        interpretive_rationale="Grounded in chronological pricing state transitions.",
                        status=InsightStatus.ASSESSED,
                    )
                )
                continue

            # C. Hiring Direction (Leadership Appointments)
            if self._is_hiring_subject(sig.subject, events):
                candidate_payloads.append(
                    RawInsightPayload(
                        insight_type=InsightType.HIRING_DIRECTION,
                        title=f"Organizational Focus: {sig.subject}",
                        summary=f"Sustained organizational additions observed in {sig.subject}.",
                        statement=f"Organizational leadership additions were observed in {sig.subject} across the monitored period.",
                        supporting_signal_ids=[sig.signal_id],
                        supporting_event_ids=event_ids,
                        interpretive_confidence_score=0.80,
                        interpretive_confidence_level=ConfidenceLevel.HIGH,
                        interpretive_rationale="Derived from executive hiring events and headcount signals.",
                        status=InsightStatus.ASSESSED,
                    )
                )
                continue

            # D. Competitive Movement & M&A (Acquisitions)
            if sig.signal_type == SignalType.EMERGING_ACTIVITY and self._is_acquisition_event(event_ids, events):
                candidate_payloads.append(
                    RawInsightPayload(
                        insight_type=InsightType.COMPETITIVE_MOVEMENT,
                        title=f"Competitive Movement: Acquisition of {sig.subject}",
                        summary=f"Acquisition activity completed regarding {sig.subject}.",
                        statement=f"Acquisition activity completed regarding {sig.subject}, expanding capability portfolio.",
                        supporting_signal_ids=[sig.signal_id],
                        supporting_event_ids=event_ids,
                        interpretive_confidence_score=0.90,
                        interpretive_confidence_level=ConfidenceLevel.VERY_HIGH,
                        interpretive_rationale="Supported by verified acquisition announcements.",
                        status=InsightStatus.ASSESSED,
                    )
                )
                continue

            # E. Product Strategy (Acceleration, Sustained Activity, Release Cadence)
            if sig.signal_type in (SignalType.ACCELERATION, SignalType.SUSTAINED_ACTIVITY, SignalType.REPEATED_CHANGE):
                candidate_payloads.append(
                    RawInsightPayload(
                        insight_type=InsightType.PRODUCT_STRATEGY,
                        title=f"Product Strategy Evolution for {sig.subject}",
                        summary=f"Product release cadence for {sig.subject} expanded during the observed period.",
                        statement=f"Product development cadence for {sig.subject} expanded with multiple releases over the observed timeframe.",
                        supporting_signal_ids=[sig.signal_id],
                        supporting_event_ids=event_ids,
                        interpretive_confidence_score=0.80,
                        interpretive_confidence_level=ConfidenceLevel.HIGH,
                        interpretive_rationale="Supported by chronological release cadence signals.",
                        status=InsightStatus.ASSESSED,
                    )
                )
                continue

        return candidate_payloads

    def _is_pricing_subject(self, subject: str, events: list[IntelligenceEvent]) -> bool:
        norm = subject.lower()
        if "price" in norm or "tier" in norm or "plan" in norm or "cost" in norm or "rate" in norm:
            return True
        return any(e.event_type == IntelligenceEventType.PRICING_CHANGE for e in events)

    def _is_hiring_subject(self, subject: str, events: list[IntelligenceEvent]) -> bool:
        norm = subject.lower()
        if "hire" in norm or "leadership" in norm or "headcount" in norm or "vp" in norm or "officer" in norm:
            return True
        return any(e.event_type == IntelligenceEventType.HIRING_OR_HEADCOUNT for e in events)

    def _is_acquisition_event(self, event_ids: list[str], events: list[IntelligenceEvent]) -> bool:
        ev_map = {e.event_id: e for e in events}
        return any(
            ev_map[eid].event_type == IntelligenceEventType.ACQUISITION
            for eid in event_ids
            if eid in ev_map
        )


class DeterministicStrategicAnalyzer(BaseStrategicAnalyzer):
    """Production component synthesizing verified, grounded strategic Insights."""

    def __init__(
        self,
        strategic_interpreter: BaseStrategicInterpreter | None = None,
        insight_validator: BaseInsightValidator | None = None,
    ) -> None:
        self.strategic_interpreter = strategic_interpreter or RuleBasedStrategicInterpreter()
        self.insight_validator = insight_validator or DeterministicInsightValidator()

    async def generate_insights(
        self,
        workspace_id: str,
        company_id: str,
        signals: list[InferredSignal],
        events: list[IntelligenceEvent],
        existing_insights: list[Insight] | None = None,
    ) -> list[Insight]:
        """Synthesize verified, grounded strategic Insights."""
        if not events and not signals:
            return []

        target_company = company_id.strip().lower()
        matched_events = [
            e for e in events if e.company_id.strip().lower() == target_company
        ]
        matched_signals = [
            s for s in signals if s.company_id.strip().lower() == target_company
        ]

        if not matched_events and not matched_signals:
            return []

        # 1. Interpret into candidate raw payloads
        candidate_payloads = await self.strategic_interpreter.interpret(
            workspace_id=workspace_id,
            company_id=company_id,
            signals=matched_signals,
            events=matched_events,
        )

        if not candidate_payloads:
            return []

        # 2. Validate and assemble Insights
        insights: list[Insight] = []
        event_map = {e.event_id: e for e in matched_events}

        for payload in candidate_payloads:
            is_valid, errors = self.insight_validator.validate(
                payload=payload,
                signals=matched_signals,
                events=matched_events,
            )
            if not is_valid:
                continue

            # Collect supporting events
            supporting_events = [
                event_map[eid] for eid in payload.supporting_event_ids if eid in event_map
            ]
            if not supporting_events:
                continue

            # Collect transitive evidence IDs
            all_evidence_ids: list[str] = []
            for ev in supporting_events:
                all_evidence_ids.extend([ref.evidence_id for ref in ev.evidence])

            # Compute factual confidence from supporting events
            ev_conf_scores = [ev.confidence.score for ev in supporting_events]
            min_ev_conf = min(ev_conf_scores)
            avg_ev_conf = sum(ev_conf_scores) / len(ev_conf_scores)
            factual_score = round(0.6 * min_ev_conf + 0.4 * avg_ev_conf, 2)
            factual_level = (
                ConfidenceLevel.VERY_HIGH
                if factual_score >= 0.90
                else ConfidenceLevel.HIGH
                if factual_score >= 0.80
                else ConfidenceLevel.MEDIUM
                if factual_score >= 0.65
                else ConfidenceLevel.LOW
            )
            factual_confidence = ConfidenceAssessment(
                score=factual_score,
                level=factual_level,
                rationale=f"Grounded across {len(supporting_events)} supporting events (min_conf={min_ev_conf:.2f}).",
            )

            # Interpretive confidence from candidate payload
            interpretive_confidence = InterpretiveConfidenceAssessment(
                score=payload.interpretive_confidence_score,
                level=payload.interpretive_confidence_level,
                rationale=payload.interpretive_rationale,
                factual_basis_summary=f"{len(supporting_events)} events, {len(payload.supporting_signal_ids)} signals",
            )

            # Significance remains initialized to UNASSESSED
            significance = SignificanceAssessment(
                status=SignificanceStatus.UNASSESSED,
                level=SignificanceLevel.UNKNOWN,
                score=None,
            )

            # Compute TemporalScope from supporting events
            timestamps = [
                ev.event_time.occurred_at or ev.event_time.acquired_at or ev.detected_at
                for ev in supporting_events
            ]
            start_date = min(timestamps) if timestamps else None
            end_date = max(timestamps) if timestamps else None
            temporal_scope = TemporalScope(
                start_date=start_date,
                end_date=end_date,
            )

            insight = Insight(
                workspace_id=workspace_id,
                company_id=company_id,
                insight_type=payload.insight_type,
                title=payload.title,
                summary=payload.summary,
                statement=payload.statement,
                supporting_signal_ids=payload.supporting_signal_ids,
                supporting_event_ids=payload.supporting_event_ids,
                supporting_evidence_ids=all_evidence_ids,
                factual_confidence=factual_confidence,
                interpretive_confidence=interpretive_confidence,
                significance=significance,
                temporal_scope=temporal_scope,
                status=payload.status,
                metadata=payload.metadata,
            )
            insights.append(insight)

        return insights
