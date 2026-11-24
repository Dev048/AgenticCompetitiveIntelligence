"""Add unique constraint on workflow_runs (workspace_id, idempotency_key)

Revision ID: 0003_workflow_run_idempotency
Revises: 0002_source_monitoring_leases
Create Date: 2026-09-12 17:58:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0003_workflow_run_idempotency"
down_revision: Union[str, None] = "0002_source_monitoring_leases"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_workflow_runs_workspace_idempotency_key",
        "workflow_runs",
        ["workspace_id", "idempotency_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_workflow_runs_workspace_idempotency_key",
        "workflow_runs",
        type_="unique",
    )
