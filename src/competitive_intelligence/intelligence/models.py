"""Domain models for the Intelligence Analysis subsystem.

Defines observable competitive intelligence events (IntelligenceEvent), grounded
multi-source evidence references (EvidenceReference), temporal abstractions (EventTime),
confidence and significance models, and historical sequence associations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import urllib.parse
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# Enumerations
# ==============================================================================


class IntelligenceEventType(str, Enum):
    """Extensible taxonomy of observable competitive intelligence events."""

    PRICING_CHANGE = "pricing_change"
    PRODUCT_LAUNCH = "product_launch"
    PRODUCT_CHANGE = "product_change"
    PRODUCT_DEPRECATION = "product_deprecation"
    RELEASE = "release"
    HIRING_OR_HEADCOUNT = "hiring_or_headcount"
    ACQUISITION = "acquisition"
    PARTNERSHIP = "partnership"
    FUNDING_OR_FINANCIAL = "funding_or_financial"
    STRATEGIC_ANNOUNCEMENT = "strategic_announcement"
    INFRASTRUCTURE_OR_TECH = "infrastructure_or_tech"
    SECURITY_OR_COMPLIANCE = "security_or_compliance"
    MARKET_ACTIVITY = "market_activity"
    EXTERNAL_OPINION = "external_opinion"
    OTHER = "other"


class EventCategory(str, Enum):
    """Provisional categorization of detected factual events for initial scaffolding."""

    PRICING_CHANGE = "pricing_change"
    PRODUCT_LAUNCH = "product_launch"
    FEATURE_UPDATE = "feature_update"
    LEADERSHIP_CHANGE = "leadership_change"
    HIRING_SURGE = "hiring_surge"
    PARTNERSHIP = "partnership"
    REPOSITORY_ACTIVITY = "repository_activity"
    COMMUNITY_BUZZ = "community_buzz"
    PRESS_RELEASE = "press_release"


class TimePrecision(str, Enum):
    """Precision level of real-world event occurrence."""

    EXACT_TIMESTAMP = "exact_timestamp"
    DATE_ONLY = "date_only"
    MONTH_YEAR = "month_year"
    QUARTER_YEAR = "quarter_year"
    YEAR_ONLY = "year_only"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Qualitative certainty grading for evidence extraction."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class SignificanceLevel(str, Enum):
    """Qualitative strategic significance grading."""

    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SignificanceStatus(str, Enum):
    """Evaluation status of strategic significance."""

    UNASSESSED = "unassessed"
    PRELIMINARY = "preliminary"
    ASSESSED = "assessed"


class SourcePerspective(str, Enum):
    """Classification of source authority and point of view."""

    FIRST_PARTY = "first_party"
    THIRD_PARTY_NEWS = "third_party_news"
    COMMUNITY_DISCUSSION = "community_discussion"
    MARKET_ANALYSIS = "market_analysis"
    EXTERNAL_OPINION = "external_opinion"
    UNKNOWN = "unknown"


class EvidenceRole(str, Enum):
    """Role of an individual evidence snippet in supporting an event."""

    PRIMARY_DIRECT = "primary_direct"
    CONFIRMATORY = "confirmatory"
    CONTEXTUAL = "contextual"
    CONTRADICTING = "contradicting"


class SignalType(str, Enum):
    """Extensible pattern taxonomy representing observable multi-event behavior."""

    REPEATED_CHANGE = "repeated_change"
    DIRECTIONAL_TREND = "directional_trend"
    REVERSAL = "reversal"
    ESCALATION = "escalation"
    ACCELERATION = "acceleration"
    EMERGING_ACTIVITY = "emerging_activity"
    SUSTAINED_ACTIVITY = "sustained_activity"
    DECLINE = "decline"
    CONTRADICTION = "contradiction"
    CORRELATED_EVENTS = "correlated_events"
    OTHER = "other"

    # Provisional / Legacy placeholders for backward compatibility
    ENTERPRISE_EXPANSION = "enterprise_expansion"
    DOWN_MARKET_EXPANSION = "down_market_expansion"
    TECH_STACK_PIVOT = "tech_stack_pivot"
    TALENT_ACQUISITION_DRIVE = "talent_acquisition_drive"
    AGGRESSIVE_DISCOUNTING = "aggressive_discounting"
    PRODUCT_SUNSET = "product_sunset"
    SECURITY_OR_COMPLIANCE_FOCUS = "security_or_compliance_focus"


class SignalStatus(str, Enum):
    """Lifecycle status of an inferred signal."""

    DETECTED = "detected"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RESOLVED = "resolved"
    UNCERTAIN = "uncertain"


class EventRelationshipType(str, Enum):
    """Deterministic relationship between two IntelligenceEvents."""

    SAME_SUBJECT = "same_subject"
    PREDECESSOR_SUCCESSOR = "predecessor_successor"
    CONTRADICTS = "contradicts"
    REINFORCES = "reinforces"
    RELATED = "related"
    UNRELATED = "unrelated"


class EventRelationship(BaseModel):
    """Explicit recorded relationship between a pair of events."""

    source_event_id: str
    target_event_id: str
    relationship_type: EventRelationshipType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str | None = None


class SignalConfidenceAssessment(BaseModel):
    """Confidence assessment for an inferred signal."""

    score: float = Field(default=0.8, ge=0.0, le=1.0)
    level: ConfidenceLevel = ConfidenceLevel.HIGH
    rationale: str | None = None
    supporting_event_count: int = 1


class SeverityLevel(str, Enum):
    """Downstream placeholder: Urgency / business impact of an event or alert."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ==============================================================================
# Supporting Domain Models
# ==============================================================================


class EventTime(BaseModel):
    """Structured temporal representation separating real-world occurrence from system acquisition."""

    occurred_at: datetime | None = None
    precision: TimePrecision = TimePrecision.UNKNOWN
    approximate_period: str | None = None
    acquired_at: datetime
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("occurred_at", "acquired_at", "detected_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime | None) -> datetime | None:
        """Ensure all timestamps are timezone-aware if present."""
        if value is not None and value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


class ConfidenceAssessment(BaseModel):
    """Extraction and evidence certainty evaluation."""

    score: float = Field(ge=0.0, le=1.0)
    level: ConfidenceLevel
    rationale: str | None = None
    is_corroborated: bool = False


class SignificanceAssessment(BaseModel):
    """Strategic business impact placeholder model (evaluated in later phases)."""

    status: SignificanceStatus = SignificanceStatus.UNASSESSED
    level: SignificanceLevel = SignificanceLevel.UNKNOWN
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None
    evaluated_at: datetime | None = None

    @field_validator("evaluated_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime | None) -> datetime | None:
        """Ensure evaluated_at is timezone-aware if present."""
        if value is not None and value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


class EvidenceReference(BaseModel):
    """Reference to a specific evidence snippet with end-to-end provenance."""

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    document_id: str
    version_id: str
    chunk_id: str | None = None
    source_url: str
    source_category: str
    source_perspective: SourcePerspective = SourcePerspective.UNKNOWN
    extracted_text: str
    acquired_at: datetime
    retrieval_score: float | None = None
    role: EvidenceRole = EvidenceRole.PRIMARY_DIRECT
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "evidence_id", "source_id", "document_id", "version_id", "source_category", "extracted_text"
    )
    @classmethod
    def validate_non_empty_identifiers(cls, value: str) -> str:
        """Ensure required identifiers and text are non-empty strings."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        """Ensure source_url is a valid HTTP/HTTPS URL with host/domain."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_url cannot be empty or whitespace-only")
        parsed = urllib.parse.urlparse(cleaned)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid URL '{cleaned}'. Expected HTTP/HTTPS URL with domain.")
        return cleaned

    @field_validator("acquired_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure acquired_at timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


# ==============================================================================
# Core Intelligence Domain Models
# ==============================================================================


class IntelligenceEvent(BaseModel):
    """An observable, evidence-backed competitive intelligence event."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str
    company_id: str
    event_type: IntelligenceEventType
    title: str
    summary: str
    subject: str

    # State transitions (optional for non-state-change events like announcements/launches)
    previous_state: dict[str, Any] | str | None = None
    new_state: dict[str, Any] | str | None = None

    # Temporal, confidence, and significance models
    event_time: EventTime
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: ConfidenceAssessment
    significance: SignificanceAssessment = Field(default_factory=SignificanceAssessment)

    # Multi-source grounded evidence (must contain at least one evidence item)
    evidence: list[EvidenceReference] = Field(min_length=1)
    primary_perspective: SourcePerspective = SourcePerspective.UNKNOWN

    # Optional historical sequence associations
    sequence_id: str | None = None
    preceding_event_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "workspace_id", "company_id", "title", "summary", "subject")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        """Ensure core fields are non-empty strings."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("detected_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure detected_at is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


# ==============================================================================
# Provisional Scaffolding Models (Maintained for Backward Compatibility)
# ==============================================================================


class RawEvidence(BaseModel):
    """Discrete factual snippet retrieved from a DocumentChunk with complete provenance."""

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str
    source_id: str
    source_url: str
    company_id: str
    extracted_text: str
    timestamp: datetime
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("timestamp")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


class DetectedEvent(BaseModel):
    """A verified factual occurrence extracted from document diffs or monitoring checks."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str
    category: EventCategory
    title: str
    summary: str
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evidence_ids: list[str] = Field(default_factory=list)
    diff_details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("detected_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure detected_at is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


class EventSequence(BaseModel):
    """Optional chronological association linking related state changes over time."""

    sequence_id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str
    subject: str
    event_ids: list[str] = Field(default_factory=list)
    initial_state: Any | None = None
    current_state: Any | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("sequence_id", "company_id", "subject")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        """Ensure sequence identifiers are non-empty strings."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()


# ==============================================================================
# Downstream Subsystem Placeholders
# ==============================================================================


class InferredSignal(BaseModel):
    """Higher-level analytical pattern derived from one or more grounded IntelligenceEvents."""

    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = "default_workspace"
    company_id: str = "default_company"
    company_ids: list[str] = Field(default_factory=list)
    signal_type: SignalType
    subject: str = "general"
    title: str = "Inferred Signal"
    summary: str

    # Direct Transitive Provenance
    supporting_event_ids: list[str] = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)

    # Temporal Bounds (never fabricated)
    first_observed_at: datetime | None = None
    last_observed_at: datetime | None = None
    inferred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Confidence and Lifecycle
    confidence: SignalConfidenceAssessment = Field(default_factory=SignalConfidenceAssessment)
    status: SignalStatus = SignalStatus.DETECTED

    # Sequence Association
    sequence_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def reconcile_company_and_confidence_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            # If company_id provided but not company_ids
            if "company_id" in values and values["company_id"]:
                if "company_ids" not in values or not values["company_ids"]:
                    values["company_ids"] = [values["company_id"]]
            # If company_ids provided but not company_id
            elif "company_ids" in values and values["company_ids"]:
                values["company_id"] = values["company_ids"][0]
            # If confidence passed as float/int
            if "confidence" in values and isinstance(values["confidence"], (int, float)):
                conf_val = float(values["confidence"])
                lvl = (
                    ConfidenceLevel.HIGH
                    if conf_val >= 0.8
                    else ConfidenceLevel.MEDIUM
                    if conf_val >= 0.65
                    else ConfidenceLevel.LOW
                )
                values["confidence"] = SignalConfidenceAssessment(score=conf_val, level=lvl)
        return values


class InsightType(str, Enum):
    """Extensible taxonomy of strategic competitive insights."""

    COMPETITIVE_MOVEMENT = "competitive_movement"
    PRODUCT_STRATEGY = "product_strategy"
    PRICING_STRATEGY = "pricing_strategy"
    MARKET_EXPANSION = "market_expansion"
    TECHNOLOGY_DIRECTION = "technology_direction"
    HIRING_DIRECTION = "hiring_direction"
    PARTNERSHIP_STRATEGY = "partnership_strategy"
    RISK_SIGNAL = "risk_signal"
    OPPORTUNITY_SIGNAL = "opportunity_signal"
    UNCERTAINTY = "uncertainty"
    OTHER = "other"


class InsightStatus(str, Enum):
    """Lifecycle status of a strategic insight."""

    DETECTED = "detected"
    PRELIMINARY = "preliminary"
    ASSESSED = "assessed"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    UNCERTAIN = "uncertain"


class InterpretiveConfidenceAssessment(BaseModel):
    """Confidence assessment in the strategic interpretation itself (separate from factual confidence)."""

    score: float = Field(default=0.8, ge=0.0, le=1.0)
    level: ConfidenceLevel = ConfidenceLevel.HIGH
    rationale: str | None = None
    factual_basis_summary: str | None = None


class TemporalScope(BaseModel):
    """Temporal window covered by a strategic insight (never fabricated)."""

    start_date: datetime | None = None
    end_date: datetime | None = None
    period_description: str | None = None


class Insight(BaseModel):
    """Bounded strategic interpretation supported by observed signals, events, and evidence."""

    insight_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = "default_workspace"
    company_id: str = "default_company"
    insight_type: InsightType = InsightType.OTHER
    title: str = "Strategic Insight"
    summary: str
    statement: str

    # Transitive Provenance (Audit Trail)
    supporting_signal_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)

    # Multi-Dimensional Confidence & Significance
    factual_confidence: ConfidenceAssessment
    interpretive_confidence: InterpretiveConfidenceAssessment = Field(
        default_factory=InterpretiveConfidenceAssessment
    )
    significance: SignificanceAssessment = Field(default_factory=SignificanceAssessment)

    # Temporal Scope & Timestamps
    temporal_scope: TemporalScope = Field(default_factory=TemporalScope)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Lifecycle & Lineage
    status: InsightStatus = InsightStatus.DETECTED
    supersedes_insight_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedInsight(BaseModel):
    """Legacy / provisional placeholder model for actionable intelligence findings."""

    insight_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str
    title: str
    narrative: str
    key_implications: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    supporting_signal_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Alert(BaseModel):
    """Notification triggered by significant events or high-impact insights."""

    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str
    severity: SeverityLevel
    title: str
    summary: str
    insight_id: str | None = None
    event_id: str | None = None
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_read: bool = False


class IntelligenceReport(BaseModel):
    """Formatted executive report presenting synthesized intelligence."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str
    title: str
    content_markdown: str
    tracked_company_ids: list[str] = Field(default_factory=list)
    generated_insight_ids: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
