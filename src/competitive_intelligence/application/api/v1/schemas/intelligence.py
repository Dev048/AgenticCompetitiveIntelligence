"""API schemas for Intelligence Events, Signals, and Insights."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class EvidenceReferenceResponse(BaseModel):
    """Evidence excerpt backing an intelligence event."""

    evidence_id: str
    source_id: str
    document_id: str
    version_id: str
    chunk_id: str | None = None
    source_url: str
    source_category: str
    source_perspective: str
    evidence_role: str
    extracted_text: str
    retrieval_score: float | None = None
    acquired_at: datetime


class IntelligenceEventResponse(BaseModel):
    """Intelligence event response DTO."""

    event_id: str
    workspace_id: str
    company_id: str
    event_type: str
    title: str
    summary: str
    subject: str
    previous_state: Any = None
    new_state: Any = None
    event_time: dict[str, Any]
    detected_at: datetime
    confidence: dict[str, Any]
    significance: dict[str, Any]
    primary_perspective: str
    sequence_id: str | None = None
    supporting_evidence: list[EvidenceReferenceResponse] = Field(default_factory=list)
    created_at: datetime


class InferredSignalResponse(BaseModel):
    """Inferred signal response DTO."""

    signal_id: str
    workspace_id: str
    company_id: str
    signal_type: str
    subject: str
    title: str
    summary: str
    first_observed_at: datetime | None = None
    last_observed_at: datetime | None = None
    inferred_at: datetime
    confidence: dict[str, Any]
    status: str
    supporting_event_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class InsightResponse(BaseModel):
    """Strategic insight response DTO."""

    insight_id: str
    workspace_id: str
    company_id: str
    insight_type: str
    title: str
    summary: str
    statement: str
    factual_confidence: dict[str, Any]
    interpretive_confidence: dict[str, Any]
    significance: dict[str, Any]
    temporal_scope: dict[str, Any]
    generated_at: datetime
    status: str
    supporting_signal_ids: list[str] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)
    created_at: datetime
