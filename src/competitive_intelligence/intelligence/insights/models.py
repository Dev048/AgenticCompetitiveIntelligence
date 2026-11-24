"""Models and intermediate payloads for Strategic Analysis and Insight Generation."""

from typing import Any

from pydantic import BaseModel, Field

from competitive_intelligence.intelligence.models import (
    ConfidenceLevel,
    InsightStatus,
    InsightType,
    TemporalScope,
)


class RawInsightPayload(BaseModel):
    """Intermediate candidate insight payload produced by a strategic interpreter."""

    insight_type: InsightType
    title: str
    summary: str
    statement: str
    supporting_signal_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(min_length=1)
    interpretive_confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    interpretive_confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    interpretive_rationale: str | None = None
    temporal_scope: TemporalScope | None = None
    status: InsightStatus = InsightStatus.DETECTED
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategicAnalysisPolicy(BaseModel):
    """Configuration policy governing deterministic strategic analysis and insight generation."""

    min_supporting_events: int = 1
    require_supporting_signals: bool = False
    enable_contradiction_insights: bool = True
    default_confidence_threshold: float = 0.50
