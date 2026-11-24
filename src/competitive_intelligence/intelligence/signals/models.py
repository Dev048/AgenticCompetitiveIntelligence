"""Configuration policies and container models for Signal Detection."""

from typing import Any

from pydantic import BaseModel, Field

from competitive_intelligence.intelligence.models import (
    EventRelationship,
    IntelligenceEvent,
    IntelligenceEventType,
)


class SignalPolicy(BaseModel):
    """Configuration policy governing deterministic signal creation."""

    # 1. Single-event signal policy: default empty (1 event -> 0 signals)
    # Only explicitly declared types are permitted to form single-event signals
    signal_worthy_single_event_types: set[IntelligenceEventType] = Field(
        default_factory=lambda: {
            IntelligenceEventType.ACQUISITION,
            IntelligenceEventType.STRATEGIC_ANNOUNCEMENT,
        }
    )

    # 2. Cadence policies (conservative by default)
    enable_cadence_detection: bool = True
    acceleration_min_events: int = 3
    acceleration_max_window_days: int = 30
    sustained_min_events: int = 3
    sustained_min_window_days: int = 60

    # 3. Minimum event count for generalized multi-event signals
    min_events_for_trend: int = 2
    min_events_for_reversal: int = 2


class EventCluster(BaseModel):
    """Container holding grouped events and evaluated relationships for a single subject."""

    company_id: str
    subject: str
    events: list[IntelligenceEvent]
    relationships: list[EventRelationship] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
