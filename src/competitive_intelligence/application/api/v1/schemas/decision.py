"""API schemas for Decision Contexts, Options, Recommendations, and Provenance."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

from competitive_intelligence.decision_support.models import (
    DecisionConstraint,
    DecisionDomain,
    DecisionObjective,
    DecisionUrgency,
    EvaluatedAlternative,
    RecommendationConfidenceAssessment,
    RiskTolerance,
    TimeHorizon,
)


class DecisionContextCreateRequest(BaseModel):
    """Payload to create a decision context."""

    title: str = Field(min_length=1, max_length=255)
    decision_domain: DecisionDomain = DecisionDomain.PRICING
    competitor_ids: list[str] = Field(default_factory=list)
    business_objectives: list[DecisionObjective] = Field(min_length=1)
    constraints: list[DecisionConstraint] = Field(default_factory=list)
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM
    current_state_summary: str | None = None
    deadline: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionContextResponse(BaseModel):
    """Decision context details DTO."""

    decision_id: str
    workspace_id: str
    organization_id: str
    title: str
    decision_domain: str
    competitor_ids: list[str]
    business_objectives: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    risk_tolerance: str
    time_horizon: str
    current_state_summary: str | None = None
    deadline: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionOptionResponse(BaseModel):
    """Decision option response DTO."""

    option_id: str
    decision_id: str
    title: str
    description: str
    action_type: str
    expected_effects: list[dict[str, Any]]
    required_resources: list[str]
    estimated_cost_usd: float | None = None
    risks: list[dict[str, Any]]
    tradeoffs: list[str]
    assumptions: list[str]
    supporting_insight_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    """Actionable recommendation response DTO."""

    recommendation_id: str
    decision_id: str
    workspace_id: str
    selected_option_id: str | None = None
    urgency: str
    status: str
    title: str
    summary: str
    rationale: str
    evaluated_alternatives: list[dict[str, Any]]
    expected_benefits: list[str]
    tradeoffs: list[str]
    critical_assumptions: list[str]
    unresolved_uncertainties: list[str]
    confidence: dict[str, Any]
    supporting_insight_ids: list[str]
    supporting_event_ids: list[str]
    generated_at: datetime


class ProvenanceRecommendationLayer(BaseModel):
    recommendation_id: str
    title: str
    summary: str | None = None
    rationale: str | None = None
    status: str
    urgency: str | None = None


class ProvenanceContextLayer(BaseModel):
    decision_id: str
    title: str
    decision_domain: str | None = None


class ProvenanceInsightLayer(BaseModel):
    insight_id: str
    insight_type: str | None = None
    title: str
    statement: str | None = None


class ProvenanceSignalLayer(BaseModel):
    signal_id: str
    signal_type: str | None = None
    title: str
    summary: str | None = None


class ProvenanceEventLayer(BaseModel):
    event_id: str
    event_type: str | None = None
    title: str
    summary: str | None = None
    detected_at: str | None = None


class ProvenanceEvidenceLayer(BaseModel):
    evidence_id: str
    quote: str | None = None
    evidence_role: str | None = None
    retrieval_score: float | None = None
    chunk_id: str | None = None
    version_id: str
    document_id: str
    source_id: str
    source_url: str | None = None


class ProvenanceChunkLayer(BaseModel):
    chunk_id: str
    version_id: str
    chunk_index: int
    total_chunks: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    content_hash: str | None = None
    preview: str | None = None


class ProvenanceVersionLayer(BaseModel):
    version_id: str
    document_id: str
    content_hash: str
    acquired_at: str | None = None


class ProvenanceDocumentLayer(BaseModel):
    document_id: str
    source_id: str | None = None
    canonical_url: str | None = None


class ProvenanceSourceLayer(BaseModel):
    source_id: str
    display_name: str | None = None
    category: str | None = None
    url: str | None = None


class ProvenanceLayers(BaseModel):
    """Per-layer details, top (recommendation) to bottom (monitored source)."""

    recommendation: ProvenanceRecommendationLayer | None = None
    decision_context: ProvenanceContextLayer | None = None
    insights: list[ProvenanceInsightLayer] = Field(default_factory=list)
    signals: list[ProvenanceSignalLayer] = Field(default_factory=list)
    events: list[ProvenanceEventLayer] = Field(default_factory=list)
    evidence: list[ProvenanceEvidenceLayer] = Field(default_factory=list)
    chunks: list[ProvenanceChunkLayer] = Field(default_factory=list)
    versions: list[ProvenanceVersionLayer] = Field(default_factory=list)
    documents: list[ProvenanceDocumentLayer] = Field(default_factory=list)
    sources: list[ProvenanceSourceLayer] = Field(default_factory=list)


class ProvenanceTreeResponse(BaseModel):
    """Complete 10-layer reconstructed provenance tree DTO."""

    recommendation_id: str
    decision_id: str
    selected_option: dict[str, Any] | None = None
    insight_count: int = 0
    signal_count: int = 0
    event_count: int = 0
    evidence_count: int = 0
    chunk_count: int = 0
    version_count: int = 0
    document_count: int = 0
    source_count: int = 0
    source_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    version_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    signal_ids: list[str] = Field(default_factory=list)
    insight_ids: list[str] = Field(default_factory=list)
    layers: ProvenanceLayers = Field(default_factory=ProvenanceLayers)


class RecommendationStatusUpdateRequest(BaseModel):
    """Payload to record human decision review for a proposed recommendation."""

    status: str = Field(description="Target recommendation status ('accepted' or 'rejected')")
    review_notes: str | None = Field(default=None, description="Optional reviewer justification notes")
