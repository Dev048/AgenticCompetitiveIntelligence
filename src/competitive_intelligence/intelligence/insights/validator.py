"""Validator component for structural grounding and provenance verification in strategic insights."""

from typing import Any

from competitive_intelligence.intelligence.insights.interfaces import BaseInsightValidator
from competitive_intelligence.intelligence.insights.models import RawInsightPayload
from competitive_intelligence.intelligence.models import (
    InferredSignal,
    IntelligenceEvent,
)


class DeterministicInsightValidator(BaseInsightValidator):
    """Production validator verifying structural grounding, ID existence, and provenance completeness.

    Enforces the core invariant: No evidence/signals -> No Insight.
    Does not rely on brittle keyword blacklists; performs structural ID and reference checks.
    """

    def validate(
        self,
        payload: Any,
        signals: list[InferredSignal],
        events: list[IntelligenceEvent],
    ) -> tuple[bool, list[str]]:
        """Validate candidate payload: returns (is_valid, validation_errors)."""
        errors: list[str] = []

        if not isinstance(payload, RawInsightPayload):
            return False, ["Payload is not an instance of RawInsightPayload."]

        # 1. Non-empty string checks
        if not payload.title or not payload.title.strip():
            errors.append("Insight title cannot be empty.")
        if not payload.summary or not payload.summary.strip():
            errors.append("Insight summary cannot be empty.")
        if not payload.statement or not payload.statement.strip():
            errors.append("Insight statement cannot be empty.")

        # 2. Structural Signal ID Verification
        known_signal_ids = {s.signal_id for s in signals}
        for sig_id in payload.supporting_signal_ids:
            if sig_id not in known_signal_ids:
                errors.append(f"Referenced signal_id '{sig_id}' does not exist in the candidate signals pool.")

        # 3. Structural Event ID Verification
        known_event_ids = {e.event_id for e in events}
        # Also include event IDs from referenced signals
        for sig in signals:
            known_event_ids.update(sig.supporting_event_ids)

        if not payload.supporting_event_ids:
            errors.append("Insight must reference at least one supporting event ID.")
        else:
            for ev_id in payload.supporting_event_ids:
                if ev_id not in known_event_ids:
                    errors.append(f"Referenced event_id '{ev_id}' does not exist in the candidate events pool.")

        is_valid = len(errors) == 0
        return is_valid, errors
