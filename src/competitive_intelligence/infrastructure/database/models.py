"""SQLAlchemy 2.0 Declarative ORM models for durable PostgreSQL persistence.

Defines the canonical schema of 25 relational tables representing application
boundaries, source/knowledge lineage, grounded intelligence graphs, decision support,
and workflow execution state.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

# Dialect-agnostic JSON column that renders as native JSONB on PostgreSQL
JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""

    pass


# ==============================================================================
# 1. Application Identity & Context (3 tables)
# ==============================================================================


class UserOrm(Base):
    """User account entity."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    workspaces: Mapped[list["WorkspaceOrm"]] = relationship(
        "WorkspaceOrm", back_populates="owner", cascade="all, delete-orphan"
    )


class WorkspaceOrm(Base):
    """Multi-tenant workspace boundary entity."""

    __tablename__ = "workspaces"

    workspace_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    owner_user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.user_id", ondelete="RESTRICT"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    owner: Mapped["UserOrm"] = relationship("UserOrm", back_populates="workspaces")
    competitors: Mapped[list["TrackedCompetitorOrm"]] = relationship(
        "TrackedCompetitorOrm", back_populates="workspace", cascade="all, delete-orphan"
    )
    sources: Mapped[list["SourceOrm"]] = relationship(
        "SourceOrm", back_populates="workspace", cascade="all, delete-orphan"
    )
    documents: Mapped[list["DocumentOrm"]] = relationship(
        "DocumentOrm", back_populates="workspace", cascade="all, delete-orphan"
    )


class TrackedCompetitorOrm(Base):
    """Competitor entity configured for active monitoring."""

    __tablename__ = "tracked_competitors"

    competitor_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("workspace_id", "domain", name="uq_tracked_competitors_workspace_domain"),
    )

    workspace: Mapped["WorkspaceOrm"] = relationship("WorkspaceOrm", back_populates="competitors")
    sources: Mapped[list["SourceOrm"]] = relationship(
        "SourceOrm", back_populates="competitor", cascade="all, delete-orphan"
    )
    documents: Mapped[list["DocumentOrm"]] = relationship(
        "DocumentOrm", back_populates="competitor", cascade="all, delete-orphan"
    )


# ==============================================================================
# 2. Source and Knowledge Lineage (4 tables)
# ==============================================================================


class SourceOrm(Base):
    """Independently monitored source metadata and configuration."""

    __tablename__ = "sources"

    source_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    company_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tracked_competitors.competitor_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Classification
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    platform: Mapped[str] = mapped_column(String(50), nullable=False, default="web")
    content_type: Mapped[str] = mapped_column(String(50), nullable=False, default="text/html")

    # Location
    url_or_identifier: Mapped[str] = mapped_column(Text, nullable=False)
    access_pattern: Mapped[str] = mapped_column(String(50), nullable=False, default="public_web")
    requires_javascript: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Acquisition Config
    acquisition_method: Mapped[str] = mapped_column(String(50), nullable=False, default="crawl4ai")
    tool_chain: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE, nullable=True)
    requires_authentication: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    rate_limit_constraints: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE, nullable=False, default=dict
    )
    last_fetch_status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="never_fetched"
    )
    consecutive_failure_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Monitoring Policy
    monitoring_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    frequency_seconds: Mapped[int] = mapped_column(Integer, default=86400, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    volatility_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Reliability
    credibility_score: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    source_tier: Mapped[str] = mapped_column(String(50), nullable=False, default="primary")
    historical_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Governance
    terms_of_service_status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="unknown"
    )
    robots_txt_respected: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    data_sensitivity: Mapped[str] = mapped_column(String(50), nullable=False, default="public")

    # Lifecycle
    lifecycle_status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
    replaced_by_source_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("sources.source_id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "company_id",
            "url_or_identifier",
            name="uq_sources_workspace_company_url",
        ),
    )

    workspace: Mapped["WorkspaceOrm"] = relationship("WorkspaceOrm", back_populates="sources")
    competitor: Mapped["TrackedCompetitorOrm"] = relationship(
        "TrackedCompetitorOrm", back_populates="sources"
    )
    documents: Mapped[list["DocumentOrm"]] = relationship(
        "DocumentOrm", back_populates="source", cascade="all, delete-orphan"
    )
    lease: Mapped["SourceMonitoringLeaseOrm | None"] = relationship(
        "SourceMonitoringLeaseOrm", back_populates="source", uselist=False, cascade="all, delete-orphan"
    )


class SourceMonitoringLeaseOrm(Base):
    """PostgreSQL-backed durable lease preventing concurrent monitoring of the same source."""

    __tablename__ = "source_monitoring_leases"

    source_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sources.source_id", ondelete="CASCADE"),
        primary_key=True,
    )
    lease_owner: Mapped[str] = mapped_column(String(255), nullable=False)
    lease_acquired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    lease_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    source: Mapped["SourceOrm"] = relationship("SourceOrm", back_populates="lease")


class DocumentOrm(Base):
    """Canonical document identity tracked across versions."""

    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sources.source_id", ondelete="CASCADE"), nullable=False, index=True
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    company_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tracked_competitors.competitor_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("source_id", "canonical_url", name="uq_documents_source_canonical_url"),
    )

    source: Mapped["SourceOrm"] = relationship("SourceOrm", back_populates="documents")
    workspace: Mapped["WorkspaceOrm"] = relationship("WorkspaceOrm", back_populates="documents")
    competitor: Mapped["TrackedCompetitorOrm"] = relationship(
        "TrackedCompetitorOrm", back_populates="documents"
    )
    versions: Mapped[list["DocumentVersionOrm"]] = relationship(
        "DocumentVersionOrm", back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["DocumentChunkOrm"]] = relationship(
        "DocumentChunkOrm", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentVersionOrm(Base):
    """Immutable content snapshot of a document at a specific timestamp."""

    __tablename__ = "document_versions"

    version_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_TYPE, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("document_id", "content_hash", name="uq_document_versions_doc_hash"),
    )

    document: Mapped["DocumentOrm"] = relationship("DocumentOrm", back_populates="versions")
    chunks: Mapped[list["DocumentChunkOrm"]] = relationship(
        "DocumentChunkOrm", back_populates="version", cascade="all, delete-orphan"
    )


class DocumentChunkOrm(Base):
    """Searchable chunk unit representing structured textual knowledge in PostgreSQL."""

    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("document_versions.version_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_chunk_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("document_chunks.chunk_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    chunk_level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    language: Mapped[str] = mapped_column(String(20), nullable=False, default="en")
    custom_tags: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    __table_args__ = (
        Index("ix_document_chunks_doc_ver_idx", "document_id", "version_id", "chunk_index"),
    )

    document: Mapped["DocumentOrm"] = relationship("DocumentOrm", back_populates="chunks")
    version: Mapped["DocumentVersionOrm"] = relationship(
        "DocumentVersionOrm", back_populates="chunks"
    )
    parent_chunk: Mapped["DocumentChunkOrm | None"] = relationship(
        "DocumentChunkOrm", remote_side=[chunk_id], backref="child_chunks"
    )


# ==============================================================================
# 3. Intelligence & Grounded Provenance (8 tables)
# ==============================================================================


class EvidenceReferenceOrm(Base):
    """Grounded factual evidence excerpt with end-to-end lineage."""

    __tablename__ = "evidence_references"

    evidence_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sources.source_id", ondelete="RESTRICT"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.document_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("document_versions.version_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("document_chunks.chunk_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_category: Mapped[str] = mapped_column(String(50), nullable=False)
    source_perspective: Mapped[str] = mapped_column(String(50), nullable=False, default="unknown")
    evidence_role: Mapped[str] = mapped_column(
        String(50), nullable=False, default="primary_direct"
    )
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_TYPE, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class IntelligenceEventOrm(Base):
    """Observable factual competitive intelligence event."""

    __tablename__ = "intelligence_events"

    event_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    company_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tracked_competitors.competitor_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    previous_state: Mapped[dict[str, Any] | str | None] = mapped_column(JSON_TYPE, nullable=True)
    new_state: Mapped[dict[str, Any] | str | None] = mapped_column(JSON_TYPE, nullable=True)
    event_time: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    confidence: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    significance: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    primary_perspective: Mapped[str] = mapped_column(String(50), nullable=False, default="unknown")
    sequence_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    preceding_event_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("intelligence_events.event_id", ondelete="SET NULL"), nullable=True
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_TYPE, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    evidence_links: Mapped[list["EventEvidenceLinkOrm"]] = relationship(
        "EventEvidenceLinkOrm", cascade="all, delete-orphan"
    )


class EventEvidenceLinkOrm(Base):
    """Explicit association table between IntelligenceEvents and EvidenceReferences."""

    __tablename__ = "event_evidence_links"

    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("intelligence_events.event_id", ondelete="CASCADE"),
        primary_key=True,
    )
    evidence_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("evidence_references.evidence_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class InferredSignalOrm(Base):
    """Observable multi-event trend or pattern."""

    __tablename__ = "inferred_signals"

    signal_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    company_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tracked_competitors.competitor_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    signal_type: Mapped[str] = mapped_column(String(50), nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    first_observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    inferred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    confidence: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="detected")
    sequence_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_TYPE, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    event_links: Mapped[list["SignalEventLinkOrm"]] = relationship(
        "SignalEventLinkOrm", cascade="all, delete-orphan"
    )


class SignalEventLinkOrm(Base):
    """Explicit association table between InferredSignals and IntelligenceEvents."""

    __tablename__ = "signal_event_links"

    signal_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inferred_signals.signal_id", ondelete="CASCADE"),
        primary_key=True,
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("intelligence_events.event_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class InsightOrm(Base):
    """Bounded strategic interpretation with full evidence grounding."""

    __tablename__ = "insights"

    insight_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    company_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tracked_competitors.competitor_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    insight_type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    factual_confidence: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    interpretive_confidence: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    significance: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    temporal_scope: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="detected")
    supersedes_insight_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("insights.insight_id", ondelete="SET NULL"), nullable=True
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_TYPE, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    signal_links: Mapped[list["InsightSignalLinkOrm"]] = relationship(
        "InsightSignalLinkOrm", cascade="all, delete-orphan"
    )
    event_links: Mapped[list["InsightEventLinkOrm"]] = relationship(
        "InsightEventLinkOrm", cascade="all, delete-orphan"
    )


class InsightSignalLinkOrm(Base):
    """Explicit association table between Insights and InferredSignals."""

    __tablename__ = "insight_signal_links"

    insight_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("insights.insight_id", ondelete="CASCADE"), primary_key=True
    )
    signal_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inferred_signals.signal_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class InsightEventLinkOrm(Base):
    """Explicit association table between Insights and IntelligenceEvents."""

    __tablename__ = "insight_event_links"

    insight_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("insights.insight_id", ondelete="CASCADE"), primary_key=True
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("intelligence_events.event_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


# ==============================================================================
# 4. Decision Support & Recommendations (8 tables)
# ==============================================================================


class DecisionContextOrm(Base):
    """User-supplied strategic context for candidate decision evaluation."""

    __tablename__ = "decision_contexts"

    decision_context_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    organization_id: Mapped[str] = mapped_column(
        String(255), nullable=False, default="default_org"
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    decision_domain: Mapped[str] = mapped_column(
        String(50), nullable=False, default="general_strategy"
    )
    business_objectives: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON_TYPE, nullable=False, default=list
    )
    constraints: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON_TYPE, nullable=False, default=list
    )
    risk_tolerance: Mapped[str] = mapped_column(String(50), nullable=False, default="moderate")
    time_horizon: Mapped[str] = mapped_column(String(50), nullable=False, default="medium_term")
    deadline: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    current_state_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_TYPE, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    competitor_links: Mapped[list["DecisionContextCompetitorOrm"]] = relationship(
        "DecisionContextCompetitorOrm", cascade="all, delete-orphan"
    )
    options: Mapped[list["DecisionOptionOrm"]] = relationship(
        "DecisionOptionOrm", back_populates="context", cascade="all, delete-orphan"
    )
    recommendations: Mapped[list["RecommendationOrm"]] = relationship(
        "RecommendationOrm", back_populates="context", cascade="all, delete-orphan"
    )


class DecisionContextCompetitorOrm(Base):
    """Explicit association table between DecisionContexts and TrackedCompetitors."""

    __tablename__ = "decision_context_competitors"

    decision_context_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("decision_contexts.decision_context_id", ondelete="CASCADE"),
        primary_key=True,
    )
    competitor_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tracked_competitors.competitor_id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class DecisionOptionOrm(Base):
    """Candidate action option generated and evaluated against decision context."""

    __tablename__ = "decision_options"

    option_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    decision_context_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("decision_contexts.decision_context_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    expected_effects: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON_TYPE, nullable=False, default=list
    )
    required_resources: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    risks: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    tradeoffs: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    assumptions: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    context: Mapped["DecisionContextOrm"] = relationship(
        "DecisionContextOrm", back_populates="options"
    )
    insight_links: Mapped[list["DecisionOptionInsightLinkOrm"]] = relationship(
        "DecisionOptionInsightLinkOrm", cascade="all, delete-orphan"
    )
    event_links: Mapped[list["DecisionOptionEventLinkOrm"]] = relationship(
        "DecisionOptionEventLinkOrm", cascade="all, delete-orphan"
    )


class DecisionOptionInsightLinkOrm(Base):
    """Explicit association table between DecisionOptions and supporting Insights."""

    __tablename__ = "decision_option_insight_links"

    option_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("decision_options.option_id", ondelete="CASCADE"),
        primary_key=True,
    )
    insight_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("insights.insight_id", ondelete="RESTRICT"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class DecisionOptionEventLinkOrm(Base):
    """Explicit association table between DecisionOptions and supporting IntelligenceEvents."""

    __tablename__ = "decision_option_event_links"

    option_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("decision_options.option_id", ondelete="CASCADE"),
        primary_key=True,
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("intelligence_events.event_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class RecommendationOrm(Base):
    """Actionable prescriptive proposal preferred under supplied objectives and constraints."""

    __tablename__ = "recommendations"

    recommendation_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    decision_context_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("decision_contexts.decision_context_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    selected_option_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("decision_options.option_id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    urgency: Mapped[str] = mapped_column(String(50), nullable=False, default="recommended")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="proposed")
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evaluated_alternatives: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON_TYPE, nullable=False, default=list
    )
    expected_benefits: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    tradeoffs: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    critical_assumptions: Mapped[list[str]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    unresolved_uncertainties: Mapped[list[str]] = mapped_column(
        JSON_TYPE, nullable=False, default=list
    )
    confidence: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    supersedes_recommendation_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("recommendations.recommendation_id", ondelete="SET NULL"),
        nullable=True,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_TYPE, nullable=False, default=dict
    )
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    context: Mapped["DecisionContextOrm"] = relationship(
        "DecisionContextOrm", back_populates="recommendations"
    )
    insight_links: Mapped[list["RecommendationInsightLinkOrm"]] = relationship(
        "RecommendationInsightLinkOrm", cascade="all, delete-orphan"
    )
    event_links: Mapped[list["RecommendationEventLinkOrm"]] = relationship(
        "RecommendationEventLinkOrm", cascade="all, delete-orphan"
    )


class RecommendationInsightLinkOrm(Base):
    """Explicit association table between Recommendations and supporting Insights."""

    __tablename__ = "recommendation_insight_links"

    recommendation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("recommendations.recommendation_id", ondelete="CASCADE"),
        primary_key=True,
    )
    insight_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("insights.insight_id", ondelete="RESTRICT"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class RecommendationEventLinkOrm(Base):
    """Explicit association table between Recommendations and supporting IntelligenceEvents."""

    __tablename__ = "recommendation_event_links"

    recommendation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("recommendations.recommendation_id", ondelete="CASCADE"),
        primary_key=True,
    )
    event_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("intelligence_events.event_id", ondelete="RESTRICT"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


# ==============================================================================
# 5. Workflow Execution & Durability (2 tables)
# ==============================================================================


class WorkflowRunOrm(Base):
    """Durable execution state record for operational workflow runs."""

    __tablename__ = "workflow_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    workflow_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default="competitive_intelligence_pipeline"
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    company_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tracked_competitors.competitor_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trigger_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True)
    current_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    target_stop_stage: Mapped[str] = mapped_column(
        String(50), nullable=False, default="insight_generation"
    )
    produced_outputs: Mapped[dict[str, Any]] = mapped_column(
        JSON_TYPE, nullable=False, default=dict
    )
    errors: Mapped[list[dict[str, Any]]] = mapped_column(JSON_TYPE, nullable=False, default=list)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_checkpointed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    correlation_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON_TYPE, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "idempotency_key",
            name="uq_workflow_runs_workspace_idempotency_key",
        ),
    )

    stage_records: Mapped[list["StageExecutionRecordOrm"]] = relationship(
        "StageExecutionRecordOrm", back_populates="run", cascade="all, delete-orphan"
    )


class StageExecutionRecordOrm(Base):
    """Audit record capturing the execution attempt of an individual pipeline stage."""

    __tablename__ = "stage_execution_records"

    stage_execution_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workflow_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    input_summary: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False, default=dict)
    output_ids: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False, default=dict)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "run_id", "stage", "attempt_count", name="uq_stage_exec_run_stage_attempt"
        ),
    )

    run: Mapped["WorkflowRunOrm"] = relationship("WorkflowRunOrm", back_populates="stage_records")
