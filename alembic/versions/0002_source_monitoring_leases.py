"""Add source_monitoring_leases table for durable continuous monitoring

Revision ID: 0002_source_monitoring_leases
Revises: 0001_initial_schema
Create Date: 2026-09-12 17:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0002_source_monitoring_leases"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "source_monitoring_leases",
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("lease_owner", sa.String(length=255), nullable=False),
        sa.Column("lease_acquired_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["sources.source_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("source_id"),
    )
    op.create_index(
        "ix_source_monitoring_leases_lease_expires_at",
        "source_monitoring_leases",
        ["lease_expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_source_monitoring_leases_lease_expires_at",
        table_name="source_monitoring_leases",
    )
    op.drop_table("source_monitoring_leases")
