"""Deterministic signal detector and strategies for Intelligence Analysis subsystem."""

from collections import defaultdict
from datetime import datetime, timezone
import re
from typing import Any

from competitive_intelligence.intelligence.models import (
    ConfidenceLevel,
    EventRelationship,
    EventRelationshipType,
    EvidenceRole,
    InferredSignal,
    IntelligenceEvent,
    IntelligenceEventType,
    SignalConfidenceAssessment,
    SignalStatus,
    SignalType,
)
from competitive_intelligence.intelligence.signals.comparators import (
    NumericStateComparator,
)
from competitive_intelligence.intelligence.signals.interfaces import (
    BaseEventRelationshipStrategy,
    BaseSignalConfidenceStrategy,
    BaseSignalDetector,
    BaseSignalGroupingStrategy,
    BaseStateComparator,
)
from competitive_intelligence.intelligence.signals.models import (
    EventCluster,
    SignalPolicy,
)


# ==============================================================================
# Default Deterministic Strategies
# ==============================================================================


class DeterministicEventRelationshipStrategy(BaseEventRelationshipStrategy):
    """Evaluates pairwise relationships between events based on subject, state, and contradiction markers."""

    def evaluate_relationship(
        self,
        event_a: IntelligenceEvent,
        event_b: IntelligenceEvent,
    ) -> EventRelationship:
        # Check company match
        if event_a.company_id.strip().lower() != event_b.company_id.strip().lower():
            return EventRelationship(
                source_event_id=event_a.event_id,
                target_event_id=event_b.event_id,
                relationship_type=EventRelationshipType.UNRELATED,
                confidence=1.0,
                rationale="Events belong to different companies.",
            )

        # Check explicit contradiction markers in evidence
        has_contradiction = any(
            ev.role == EvidenceRole.CONTRADICTING for ev in event_a.evidence + event_b.evidence
        )
        if has_contradiction and self._is_same_subject(event_a.subject, event_b.subject):
            return EventRelationship(
                source_event_id=event_a.event_id,
                target_event_id=event_b.event_id,
                relationship_type=EventRelationshipType.CONTRADICTS,
                confidence=0.95,
                rationale="Conflicting evidence or postponement reported for the same subject.",
            )

        # Check state predecessor / successor continuity
        if (
            event_a.new_state is not None
            and event_b.previous_state is not None
            and event_a.new_state == event_b.previous_state
        ):
            return EventRelationship(
                source_event_id=event_a.event_id,
                target_event_id=event_b.event_id,
                relationship_type=EventRelationshipType.PREDECESSOR_SUCCESSOR,
                confidence=1.0,
                rationale="Direct state transition continuity (event_b.previous_state == event_a.new_state).",
            )

        # Check subject match
        if self._is_same_subject(event_a.subject, event_b.subject):
            return EventRelationship(
                source_event_id=event_a.event_id,
                target_event_id=event_b.event_id,
                relationship_type=EventRelationshipType.SAME_SUBJECT,
                confidence=1.0,
                rationale="Events involve the same subject / product / topic.",
            )

        return EventRelationship(
            source_event_id=event_a.event_id,
            target_event_id=event_b.event_id,
            relationship_type=EventRelationshipType.UNRELATED,
            confidence=1.0,
            rationale="Events involve disjoint subjects.",
        )

    def _is_same_subject(self, sub_a: str, sub_b: str) -> bool:
        norm_a = re.sub(r"[^a-z0-9]", "", sub_a.lower().strip())
        norm_b = re.sub(r"[^a-z0-9]", "", sub_b.lower().strip())
        return bool(norm_a and norm_b and (norm_a == norm_b or norm_a in norm_b or norm_b in norm_a))


class DeterministicSignalGroupingStrategy(BaseSignalGroupingStrategy):
    """Clusters events deterministically by company and normalized subject."""

    def group_events(
        self,
        events: list[IntelligenceEvent],
    ) -> dict[str, list[IntelligenceEvent]]:
        clusters: dict[str, list[IntelligenceEvent]] = defaultdict(list)
        for event in events:
            norm_sub = re.sub(r"[^a-z0-9]", "", event.subject.lower().strip()) or "general"
            key = f"{event.company_id.strip().lower()}::{norm_sub}"
            clusters[key].append(event)
        return dict(clusters)


class DeterministicSignalConfidenceStrategy(BaseSignalConfidenceStrategy):
    """Baseline signal confidence strategy.

    Invariants:
    - Multiple low-confidence events DO NOT automatically produce high confidence.
    - Signal confidence is bounded by supporting event confidences.
    - Contradictions penalize confidence.
    """

    def compute_confidence(
        self,
        events: list[IntelligenceEvent],
        signal_type: SignalType,
        relationships: list[EventRelationship] | None = None,
    ) -> SignalConfidenceAssessment:
        if not events:
            return SignalConfidenceAssessment(score=0.0, level=ConfidenceLevel.LOW, rationale="No events")

        scores = [e.confidence.score for e in events]
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)

        # Baseline score: conservative blend bounded by min_score
        base_score = 0.6 * min_score + 0.4 * avg_score

        # Temporal bonus only if events have verified explicit dates
        has_explicit_dates = all(e.event_time.occurred_at is not None for e in events)
        if has_explicit_dates and len(events) >= 2:
            base_score = min(1.0, base_score + 0.05)

        # Contradiction penalty
        if signal_type == SignalType.CONTRADICTION:
            base_score = min(0.60, base_score)
            rationale = "Signal confidence capped due to contradictory or disputed evidence."
        else:
            rationale = (
                f"Synthesized across {len(events)} events (min_conf={min_score:.2f}, avg={avg_score:.2f})."
            )

        if base_score >= 0.90:
            level = ConfidenceLevel.VERY_HIGH
        elif base_score >= 0.80:
            level = ConfidenceLevel.HIGH
        elif base_score >= 0.65:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        return SignalConfidenceAssessment(
            score=round(base_score, 2),
            level=level,
            rationale=rationale,
            supporting_event_count=len(events),
        )


# ==============================================================================
# Deterministic Signal Detector Component
# ==============================================================================


class DeterministicSignalDetector(BaseSignalDetector):
    """Production component detecting InferredSignals from grounded IntelligenceEvents.

    Prioritizes mechanically defensible signals (REPEATED_CHANGE, REVERSAL, DIRECTIONAL_TREND, CONTRADICTION).
    Conservative baseline: insufficient evidence -> 0 signals.
    """

    def __init__(
        self,
        policy: SignalPolicy | None = None,
        state_comparator: BaseStateComparator | None = None,
        confidence_strategy: BaseSignalConfidenceStrategy | None = None,
        relationship_strategy: BaseEventRelationshipStrategy | None = None,
        grouping_strategy: BaseSignalGroupingStrategy | None = None,
    ) -> None:
        self.policy = policy or SignalPolicy()
        self.state_comparator = state_comparator or NumericStateComparator()
        self.confidence_strategy = confidence_strategy or DeterministicSignalConfidenceStrategy()
        self.relationship_strategy = (
            relationship_strategy or DeterministicEventRelationshipStrategy()
        )
        self.grouping_strategy = grouping_strategy or DeterministicSignalGroupingStrategy()

    async def detect_signals(
        self,
        workspace_id: str,
        company_id: str,
        events: list[IntelligenceEvent],
        existing_signals: list[InferredSignal] | None = None,
    ) -> list[InferredSignal]:
        """Synthesize high-level InferredSignals from grounded IntelligenceEvents."""
        if not events:
            return []

        # 1. Filter events matching target company
        target_company = company_id.strip().lower()
        company_events = [
            e for e in events if e.company_id.strip().lower() == target_company
        ]
        if not company_events:
            return []

        # 2. Sort chronologically
        sorted_events = self._sort_events_chronologically(company_events)

        # 3. Group by subject/cluster
        clusters = self.grouping_strategy.group_events(sorted_events)

        # 4. Evaluate each cluster against deterministic decision rules
        inferred_signals: list[InferredSignal] = []

        for cluster_key, cluster_events in clusters.items():
            if not cluster_events:
                continue

            # Evaluate pairwise relationships
            relationships: list[EventRelationship] = []
            for i in range(len(cluster_events) - 1):
                rel = self.relationship_strategy.evaluate_relationship(
                    cluster_events[i], cluster_events[i + 1]
                )
                relationships.append(rel)

            cluster = EventCluster(
                company_id=company_id,
                subject=cluster_events[0].subject,
                events=cluster_events,
                relationships=relationships,
            )

            # Synthesize signal for cluster
            signal = self._evaluate_cluster_signal(workspace_id, company_id, cluster)
            if signal is not None:
                inferred_signals.append(signal)

        return inferred_signals

    def _evaluate_cluster_signal(
        self,
        workspace_id: str,
        company_id: str,
        cluster: EventCluster,
    ) -> InferredSignal | None:
        events = cluster.events
        subject = cluster.subject
        event_count = len(events)

        # Rule 1: Single Event
        if event_count == 1:
            single_event = events[0]
            if single_event.event_type in self.policy.signal_worthy_single_event_types:
                conf = self.confidence_strategy.compute_confidence(
                    events, SignalType.EMERGING_ACTIVITY, cluster.relationships
                )
                return InferredSignal(
                    workspace_id=workspace_id,
                    company_id=company_id,
                    signal_type=SignalType.EMERGING_ACTIVITY,
                    subject=subject,
                    title=f"Emerging Notable Event: {single_event.title}",
                    summary=f"Isolated notable event observed for {subject}: {single_event.summary}",
                    supporting_event_ids=[single_event.event_id],
                    supporting_evidence_ids=[ev.evidence_id for ev in single_event.evidence],
                    first_observed_at=self._get_effective_time(single_event),
                    last_observed_at=self._get_effective_time(single_event),
                    confidence=conf,
                    status=SignalStatus.DETECTED,
                    sequence_id=single_event.sequence_id,
                )
            # Default single-event: 0 signals
            return None

        # Rule 2: Check for Contradictions
        has_contradiction_rel = any(
            r.relationship_type == EventRelationshipType.CONTRADICTS for r in cluster.relationships
        )
        has_contradiction_role = any(
            ev.role == EvidenceRole.CONTRADICTING for e in events for ev in e.evidence
        )
        if has_contradiction_rel or has_contradiction_role:
            conf = self.confidence_strategy.compute_confidence(
                events, SignalType.CONTRADICTION, cluster.relationships
            )
            return self._build_signal(
                workspace_id=workspace_id,
                company_id=company_id,
                signal_type=SignalType.CONTRADICTION,
                subject=subject,
                title=f"Contradiction / Dispute regarding {subject}",
                summary=f"Conflicting reports or disputed status observed regarding {subject}.",
                events=events,
                confidence=conf,
                status=SignalStatus.UNCERTAIN,
            )

        # Rule 3: Check for State Reversal (S_0 -> S_1 -> S_0)
        first_state = events[0].previous_state if events[0].previous_state is not None else events[0].new_state
        latest_state = events[-1].new_state
        if (
            event_count >= self.policy.min_events_for_reversal
            and first_state is not None
            and latest_state is not None
            and self.state_comparator.is_reversal(first_state, latest_state)
        ):
            conf = self.confidence_strategy.compute_confidence(
                events, SignalType.REVERSAL, cluster.relationships
            )
            return self._build_signal(
                workspace_id=workspace_id,
                company_id=company_id,
                signal_type=SignalType.REVERSAL,
                subject=subject,
                title=f"State Reversal on {subject}",
                summary=f"State progression for {subject} reverted to initial state ({first_state} -> {latest_state}).",
                events=events,
                confidence=conf,
                status=SignalStatus.RESOLVED,
            )

        # Rule 4: Check for Directional Trend (Monotonic progression via comparator)
        if event_count >= self.policy.min_events_for_trend:
            is_trend, trend_dir = self._check_monotonic_trend(events)
            if is_trend and trend_dir != 0:
                conf = self.confidence_strategy.compute_confidence(
                    events, SignalType.DIRECTIONAL_TREND, cluster.relationships
                )
                dir_desc = "upward / expanding" if trend_dir > 0 else "downward / reducing"
                return self._build_signal(
                    workspace_id=workspace_id,
                    company_id=company_id,
                    signal_type=SignalType.DIRECTIONAL_TREND,
                    subject=subject,
                    title=f"Directional Trend ({dir_desc}) on {subject}",
                    summary=f"Observed sequential {dir_desc} state progression across {event_count} events on {subject}.",
                    events=events,
                    confidence=conf,
                    status=SignalStatus.ACTIVE,
                )

        # Rule 5: Cadence Policies (Acceleration & Sustained Activity)
        if self.policy.enable_cadence_detection and event_count >= 2:
            window_days = self._calculate_window_days(events)
            if window_days is not None:
                # Acceleration check
                if (
                    event_count >= self.policy.acceleration_min_events
                    and window_days <= self.policy.acceleration_max_window_days
                ):
                    conf = self.confidence_strategy.compute_confidence(
                        events, SignalType.ACCELERATION, cluster.relationships
                    )
                    return self._build_signal(
                        workspace_id=workspace_id,
                        company_id=company_id,
                        signal_type=SignalType.ACCELERATION,
                        subject=subject,
                        title=f"Activity Acceleration on {subject}",
                        summary=f"Rapid cadence of {event_count} events observed within {window_days:.0f} days.",
                        events=events,
                        confidence=conf,
                        status=SignalStatus.ACTIVE,
                    )
                # Sustained Activity check
                if (
                    event_count >= self.policy.sustained_min_events
                    and window_days >= self.policy.sustained_min_window_days
                ):
                    conf = self.confidence_strategy.compute_confidence(
                        events, SignalType.SUSTAINED_ACTIVITY, cluster.relationships
                    )
                    return self._build_signal(
                        workspace_id=workspace_id,
                        company_id=company_id,
                        signal_type=SignalType.SUSTAINED_ACTIVITY,
                        subject=subject,
                        title=f"Sustained Activity on {subject}",
                        summary=f"Sustained operational cadence across {event_count} events over {window_days:.0f} days.",
                        events=events,
                        confidence=conf,
                        status=SignalStatus.ACTIVE,
                    )

        # Rule 6: Repeated Change (Generic multi-event change on subject)
        has_multiple_changes = any(
            e.event_type in (
                IntelligenceEventType.PRICING_CHANGE,
                IntelligenceEventType.PRODUCT_CHANGE,
                IntelligenceEventType.RELEASE,
                IntelligenceEventType.HIRING_OR_HEADCOUNT,
            )
            for e in events
        )
        if event_count >= 2 and has_multiple_changes:
            conf = self.confidence_strategy.compute_confidence(
                events, SignalType.REPEATED_CHANGE, cluster.relationships
            )
            return self._build_signal(
                workspace_id=workspace_id,
                company_id=company_id,
                signal_type=SignalType.REPEATED_CHANGE,
                subject=subject,
                title=f"Repeated Activity on {subject}",
                summary=f"Multiple chronological events ({event_count} total) observed on {subject}.",
                events=events,
                confidence=conf,
                status=SignalStatus.ACTIVE,
            )

        # Conservative fallback: No Signal
        return None

    def _check_monotonic_trend(self, events: list[IntelligenceEvent]) -> tuple[bool, int]:
        """Check if events exhibit a strict monotonic progression via the configured state comparator."""
        directions: list[int] = []
        for i in range(len(events) - 1):
            s_curr = events[i].new_state
            s_next = events[i + 1].new_state
            if s_curr is None or s_next is None:
                return False, 0
            dir_res = self.state_comparator.compare_direction(s_curr, s_next)
            if dir_res is None or dir_res == 0:
                return False, 0
            directions.append(dir_res)

        if not directions:
            return False, 0

        first_dir = directions[0]
        if all(d == first_dir for d in directions):
            return True, first_dir
        return False, 0

    def _build_signal(
        self,
        workspace_id: str,
        company_id: str,
        signal_type: SignalType,
        subject: str,
        title: str,
        summary: str,
        events: list[IntelligenceEvent],
        confidence: SignalConfidenceAssessment,
        status: SignalStatus,
    ) -> InferredSignal:
        first_time = self._get_effective_time(events[0])
        last_time = self._get_effective_time(events[-1])
        all_evidence_ids = [ev.evidence_id for e in events for ev in e.evidence]

        # Use sequence_id if any event is associated with one
        seq_id = next((e.sequence_id for e in events if e.sequence_id is not None), None)

        return InferredSignal(
            workspace_id=workspace_id,
            company_id=company_id,
            signal_type=signal_type,
            subject=subject,
            title=title,
            summary=summary,
            supporting_event_ids=[e.event_id for e in events],
            supporting_evidence_ids=all_evidence_ids,
            first_observed_at=first_time,
            last_observed_at=last_time,
            confidence=confidence,
            status=status,
            sequence_id=seq_id,
        )

    def _sort_events_chronologically(
        self, events: list[IntelligenceEvent]
    ) -> list[IntelligenceEvent]:
        """Sort events by effective chronological timestamp."""
        return sorted(events, key=lambda e: self._get_effective_time(e))

    def _get_effective_time(self, event: IntelligenceEvent) -> datetime:
        """Get best available timestamp: occurred_at -> acquired_at -> detected_at."""
        if event.event_time.occurred_at is not None:
            return event.event_time.occurred_at
        if event.event_time.acquired_at is not None:
            return event.event_time.acquired_at
        return event.detected_at

    def _calculate_window_days(self, events: list[IntelligenceEvent]) -> float | None:
        """Calculate elapsed days between first and last event in cluster."""
        if len(events) < 2:
            return None
        t_first = self._get_effective_time(events[0])
        t_last = self._get_effective_time(events[-1])
        return abs((t_last - t_first).total_seconds()) / 86400.0
