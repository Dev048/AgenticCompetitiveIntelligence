"""Deterministic event sequence tracker for maintaining structural chronological state chains."""

from datetime import datetime, timezone

from competitive_intelligence.intelligence.interfaces import BaseEventSequenceTracker
from competitive_intelligence.intelligence.models import (
    EventSequence,
    IntelligenceEvent,
)


class DeterministicEventSequenceTracker(BaseEventSequenceTracker):
    """Production implementation of BaseEventSequenceTracker.

    Maintains a strictly structural, linear chain of state transitions on a specific subject.
    Does not synthesize higher-level business signals (delegated to SignalDetector).
    """

    def append_event(
        self,
        sequence: EventSequence | None,
        event: IntelligenceEvent,
    ) -> EventSequence:
        """Associate an event with an EventSequence, updating the sequence's state chain."""
        now = datetime.now(timezone.utc)

        if sequence is None:
            # Initialize a new EventSequence
            initial_state = event.previous_state if event.previous_state is not None else event.new_state
            return EventSequence(
                company_id=event.company_id,
                subject=event.subject,
                event_ids=[event.event_id],
                initial_state=initial_state,
                current_state=event.new_state,
                created_at=now,
                updated_at=now,
            )

        # Update existing sequence
        updated_event_ids = list(sequence.event_ids)
        if event.event_id not in updated_event_ids:
            updated_event_ids.append(event.event_id)

        # Update initial_state if not already established
        initial_state = sequence.initial_state
        if initial_state is None and event.previous_state is not None:
            initial_state = event.previous_state

        return EventSequence(
            sequence_id=sequence.sequence_id,
            company_id=sequence.company_id,
            subject=sequence.subject,
            event_ids=updated_event_ids,
            initial_state=initial_state,
            current_state=event.new_state if event.new_state is not None else sequence.current_state,
            created_at=sequence.created_at,
            updated_at=now,
        )
