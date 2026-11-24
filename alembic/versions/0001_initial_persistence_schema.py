"""Initial PostgreSQL persistence schema with 25 relational tables

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-09-12 16:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. users
    op.create_table(
        "users",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # 2. workspaces
    op.create_table(
        "workspaces",
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("owner_user_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_user_id"], ["users.user_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("workspace_id"),
    )
    op.create_index("ix_workspaces_owner_user_id", "workspaces", ["owner_user_id"])

    # 3. tracked_competitors
    op.create_table(
        "tracked_competitors",
        sa.Column("competitor_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("competitor_id"),
        sa.UniqueConstraint("workspace_id", "domain", name="uq_tracked_competitors_workspace_domain"),
    )
    op.create_index("ix_tracked_competitors_workspace_id", "tracked_competitors", ["workspace_id"])

    # 4. sources
    op.create_table(
        "sources",
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.String(length=36), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("platform", sa.String(length=50), nullable=False, server_default="web"),
        sa.Column("content_type", sa.String(length=50), nullable=False, server_default="text/html"),
        sa.Column("url_or_identifier", sa.Text(), nullable=False),
        sa.Column("access_pattern", sa.String(length=50), nullable=False, server_default="public_web"),
        sa.Column("requires_javascript", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("acquisition_method", sa.String(length=50), nullable=False, server_default="crawl4ai"),
        sa.Column("tool_chain", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("requires_authentication", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("rate_limit_constraints", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("last_fetch_status", sa.String(length=50), nullable=False, server_default="never_fetched"),
        sa.Column("consecutive_failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("monitoring_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("frequency_seconds", sa.Integer(), nullable=False, server_default="86400"),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("volatility_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("credibility_score", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("source_tier", sa.String(length=50), nullable=False, server_default="primary"),
        sa.Column("historical_accuracy", sa.Float(), nullable=True),
        sa.Column("terms_of_service_status", sa.String(length=50), nullable=False, server_default="unknown"),
        sa.Column("robots_txt_respected", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("data_sensitivity", sa.String(length=50), nullable=False, server_default="public"),
        sa.Column("lifecycle_status", sa.String(length=50), nullable=False, server_default="active"),
        sa.Column("replaced_by_source_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["tracked_competitors.competitor_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["replaced_by_source_id"], ["sources.source_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("source_id"),
        sa.UniqueConstraint("workspace_id", "company_id", "url_or_identifier", name="uq_sources_workspace_company_url"),
    )
    op.create_index("ix_sources_workspace_id", "sources", ["workspace_id"])
    op.create_index("ix_sources_company_id", "sources", ["company_id"])

    # 5. documents
    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.String(length=36), nullable=False),
        sa.Column("canonical_url", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["sources.source_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["tracked_competitors.competitor_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_id"),
        sa.UniqueConstraint("source_id", "canonical_url", name="uq_documents_source_canonical_url"),
    )
    op.create_index("ix_documents_source_id", "documents", ["source_id"])
    op.create_index("ix_documents_workspace_id", "documents", ["workspace_id"])
    op.create_index("ix_documents_company_id", "documents", ["company_id"])

    # 6. document_versions
    op.create_table(
        "document_versions",
        sa.Column("version_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("version_id"),
        sa.UniqueConstraint("document_id", "content_hash", name="uq_document_versions_doc_hash"),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_versions_content_hash", "document_versions", ["content_hash"])

    # 7. document_chunks
    op.create_table(
        "document_chunks",
        sa.Column("chunk_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("version_id", sa.String(length=36), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("parent_chunk_id", sa.String(length=36), nullable=True),
        sa.Column("chunk_level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("start_char", sa.Integer(), nullable=True),
        sa.Column("end_char", sa.Integer(), nullable=True),
        sa.Column("total_chunks", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("language", sa.String(length=20), nullable=False, server_default="en"),
        sa.Column("custom_tags", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.version_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["parent_chunk_id"], ["document_chunks.chunk_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("ix_document_chunks_version_id", "document_chunks", ["version_id"])
    op.create_index("ix_document_chunks_parent_chunk_id", "document_chunks", ["parent_chunk_id"])
    op.create_index("ix_document_chunks_doc_ver_idx", "document_chunks", ["document_id", "version_id", "chunk_index"])

    # 8. evidence_references
    op.create_table(
        "evidence_references",
        sa.Column("evidence_id", sa.String(length=36), nullable=False),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("version_id", sa.String(length=36), nullable=False),
        sa.Column("chunk_id", sa.String(length=36), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_category", sa.String(length=50), nullable=False),
        sa.Column("source_perspective", sa.String(length=50), nullable=False, server_default="unknown"),
        sa.Column("evidence_role", sa.String(length=50), nullable=False, server_default="primary_direct"),
        sa.Column("extracted_text", sa.Text(), nullable=False),
        sa.Column("retrieval_score", sa.Float(), nullable=True),
        sa.Column("acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["sources.source_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.document_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.version_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["chunk_id"], ["document_chunks.chunk_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("evidence_id"),
    )
    op.create_index("ix_evidence_references_source_id", "evidence_references", ["source_id"])
    op.create_index("ix_evidence_references_document_id", "evidence_references", ["document_id"])
    op.create_index("ix_evidence_references_version_id", "evidence_references", ["version_id"])
    op.create_index("ix_evidence_references_chunk_id", "evidence_references", ["chunk_id"])

    # 9. intelligence_events
    op.create_table(
        "intelligence_events",
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.String(length=36), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("previous_state", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("new_state", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("event_time", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("significance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("primary_perspective", sa.String(length=50), nullable=False, server_default="unknown"),
        sa.Column("sequence_id", sa.String(length=36), nullable=True),
        sa.Column("preceding_event_id", sa.String(length=36), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["tracked_competitors.competitor_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["preceding_event_id"], ["intelligence_events.event_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("ix_intelligence_events_workspace_id", "intelligence_events", ["workspace_id"])
    op.create_index("ix_intelligence_events_company_id", "intelligence_events", ["company_id"])
    op.create_index("ix_intelligence_events_detected_at", "intelligence_events", ["detected_at"])
    op.create_index("ix_intelligence_events_sequence_id", "intelligence_events", ["sequence_id"])

    # 10. event_evidence_links
    op.create_table(
        "event_evidence_links",
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("evidence_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["intelligence_events.event_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["evidence_id"], ["evidence_references.evidence_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("event_id", "evidence_id"),
    )

    # 11. inferred_signals
    op.create_table(
        "inferred_signals",
        sa.Column("signal_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.String(length=36), nullable=False),
        sa.Column("signal_type", sa.String(length=50), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("first_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("inferred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="detected"),
        sa.Column("sequence_id", sa.String(length=36), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["tracked_competitors.competitor_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("signal_id"),
    )
    op.create_index("ix_inferred_signals_workspace_id", "inferred_signals", ["workspace_id"])
    op.create_index("ix_inferred_signals_company_id", "inferred_signals", ["company_id"])
    op.create_index("ix_inferred_signals_inferred_at", "inferred_signals", ["inferred_at"])

    # 12. signal_event_links
    op.create_table(
        "signal_event_links",
        sa.Column("signal_id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["signal_id"], ["inferred_signals.signal_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["intelligence_events.event_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("signal_id", "event_id"),
    )

    # 13. insights
    op.create_table(
        "insights",
        sa.Column("insight_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.String(length=36), nullable=False),
        sa.Column("insight_type", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("factual_confidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("interpretive_confidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("significance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("temporal_scope", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="detected"),
        sa.Column("supersedes_insight_id", sa.String(length=36), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["tracked_competitors.competitor_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["supersedes_insight_id"], ["insights.insight_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("insight_id"),
    )
    op.create_index("ix_insights_workspace_id", "insights", ["workspace_id"])
    op.create_index("ix_insights_company_id", "insights", ["company_id"])
    op.create_index("ix_insights_generated_at", "insights", ["generated_at"])

    # 14. insight_signal_links
    op.create_table(
        "insight_signal_links",
        sa.Column("insight_id", sa.String(length=36), nullable=False),
        sa.Column("signal_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["insight_id"], ["insights.insight_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["signal_id"], ["inferred_signals.signal_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("insight_id", "signal_id"),
    )

    # 15. insight_event_links
    op.create_table(
        "insight_event_links",
        sa.Column("insight_id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["insight_id"], ["insights.insight_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["intelligence_events.event_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("insight_id", "event_id"),
    )

    # 16. decision_contexts
    op.create_table(
        "decision_contexts",
        sa.Column("decision_context_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("organization_id", sa.String(length=255), nullable=False, server_default="default_org"),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("decision_domain", sa.String(length=50), nullable=False, server_default="general_strategy"),
        sa.Column("business_objectives", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("constraints", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("risk_tolerance", sa.String(length=50), nullable=False, server_default="moderate"),
        sa.Column("time_horizon", sa.String(length=50), nullable=False, server_default="medium_term"),
        sa.Column("deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column("current_state_summary", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("decision_context_id"),
    )
    op.create_index("ix_decision_contexts_workspace_id", "decision_contexts", ["workspace_id"])

    # 17. decision_context_competitors
    op.create_table(
        "decision_context_competitors",
        sa.Column("decision_context_id", sa.String(length=36), nullable=False),
        sa.Column("competitor_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["decision_context_id"], ["decision_contexts.decision_context_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["competitor_id"], ["tracked_competitors.competitor_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("decision_context_id", "competitor_id"),
    )

    # 18. decision_options
    op.create_table(
        "decision_options",
        sa.Column("option_id", sa.String(length=36), nullable=False),
        sa.Column("decision_context_id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("action_type", sa.String(length=50), nullable=False),
        sa.Column("expected_effects", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("required_resources", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=True),
        sa.Column("risks", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("tradeoffs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("assumptions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["decision_context_id"], ["decision_contexts.decision_context_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("option_id"),
    )
    op.create_index("ix_decision_options_decision_context_id", "decision_options", ["decision_context_id"])

    # 19. decision_option_insight_links
    op.create_table(
        "decision_option_insight_links",
        sa.Column("option_id", sa.String(length=36), nullable=False),
        sa.Column("insight_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["option_id"], ["decision_options.option_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["insight_id"], ["insights.insight_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("option_id", "insight_id"),
    )

    # 20. decision_option_event_links
    op.create_table(
        "decision_option_event_links",
        sa.Column("option_id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["option_id"], ["decision_options.option_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["intelligence_events.event_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("option_id", "event_id"),
    )

    # 21. recommendations
    op.create_table(
        "recommendations",
        sa.Column("recommendation_id", sa.String(length=36), nullable=False),
        sa.Column("decision_context_id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("selected_option_id", sa.String(length=36), nullable=True),
        sa.Column("urgency", sa.String(length=50), nullable=False, server_default="recommended"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="proposed"),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("evaluated_alternatives", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("expected_benefits", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("tradeoffs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("critical_assumptions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("unresolved_uncertainties", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("confidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("supersedes_recommendation_id", sa.String(length=36), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["decision_context_id"], ["decision_contexts.decision_context_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["selected_option_id"], ["decision_options.option_id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["supersedes_recommendation_id"], ["recommendations.recommendation_id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("recommendation_id"),
    )
    op.create_index("ix_recommendations_decision_context_id", "recommendations", ["decision_context_id"])
    op.create_index("ix_recommendations_workspace_id", "recommendations", ["workspace_id"])
    op.create_index("ix_recommendations_selected_option_id", "recommendations", ["selected_option_id"])
    op.create_index("ix_recommendations_generated_at", "recommendations", ["generated_at"])

    # 22. recommendation_insight_links
    op.create_table(
        "recommendation_insight_links",
        sa.Column("recommendation_id", sa.String(length=36), nullable=False),
        sa.Column("insight_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["recommendation_id"], ["recommendations.recommendation_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["insight_id"], ["insights.insight_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("recommendation_id", "insight_id"),
    )

    # 23. recommendation_event_links
    op.create_table(
        "recommendation_event_links",
        sa.Column("recommendation_id", sa.String(length=36), nullable=False),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["recommendation_id"], ["recommendations.recommendation_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["event_id"], ["intelligence_events.event_id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("recommendation_id", "event_id"),
    )

    # 24. workflow_runs
    op.create_table(
        "workflow_runs",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("workflow_id", sa.String(length=100), nullable=False, server_default="competitive_intelligence_pipeline"),
        sa.Column("workspace_id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.String(length=36), nullable=False),
        sa.Column("trigger_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("current_stage", sa.String(length=50), nullable=True),
        sa.Column("target_stop_stage", sa.String(length=50), nullable=False, server_default="insight_generation"),
        sa.Column("produced_outputs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("errors", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_checkpointed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("correlation_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["tracked_competitors.competitor_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index("ix_workflow_runs_workspace_id", "workflow_runs", ["workspace_id"])
    op.create_index("ix_workflow_runs_company_id", "workflow_runs", ["company_id"])
    op.create_index("ix_workflow_runs_status", "workflow_runs", ["status"])
    op.create_index("ix_workflow_runs_started_at", "workflow_runs", ["started_at"])
    op.create_index("ix_workflow_runs_correlation_id", "workflow_runs", ["correlation_id"])
    op.create_index("ix_workflow_runs_idempotency_key", "workflow_runs", ["idempotency_key"])

    # 25. stage_execution_records
    op.create_table(
        "stage_execution_records",
        sa.Column("stage_execution_id", sa.String(length=36), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("stage", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("input_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("output_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["workflow_runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("stage_execution_id"),
        sa.UniqueConstraint("run_id", "stage", "attempt_count", name="uq_stage_exec_run_stage_attempt"),
    )
    op.create_index("ix_stage_execution_records_run_id", "stage_execution_records", ["run_id"])
    op.create_index("ix_stage_execution_records_idempotency_key", "stage_execution_records", ["idempotency_key"])


def downgrade() -> None:
    op.drop_table("stage_execution_records")
    op.drop_table("workflow_runs")
    op.drop_table("recommendation_event_links")
    op.drop_table("recommendation_insight_links")
    op.drop_table("recommendations")
    op.drop_table("decision_option_event_links")
    op.drop_table("decision_option_insight_links")
    op.drop_table("decision_options")
    op.drop_table("decision_context_competitors")
    op.drop_table("decision_contexts")
    op.drop_table("insight_event_links")
    op.drop_table("insight_signal_links")
    op.drop_table("insights")
    op.drop_table("signal_event_links")
    op.drop_table("inferred_signals")
    op.drop_table("event_evidence_links")
    op.drop_table("intelligence_events")
    op.drop_table("evidence_references")
    op.drop_table("document_chunks")
    op.drop_table("document_versions")
    op.drop_table("documents")
    op.drop_table("sources")
    op.drop_table("tracked_competitors")
    op.drop_table("workspaces")
    op.drop_table("users")
