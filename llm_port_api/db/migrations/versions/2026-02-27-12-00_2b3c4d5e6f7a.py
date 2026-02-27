"""Add pii_config JSON column to tenant_llm_policy.

Revision ID: 2b3c4d5e6f7a
Revises: 1a2b3c4d5e6f
Create Date: 2026-02-27 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "2b3c4d5e6f7a"
down_revision = "1a2b3c4d5e6f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add pii_config to tenant_llm_policy."""
    op.add_column(
        "tenant_llm_policy",
        sa.Column(
            "pii_config",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove pii_config from tenant_llm_policy."""
    op.drop_column("tenant_llm_policy", "pii_config")
