"""Add chat attachments table and content_parts_json column.

Revision ID: 4d5e6f7a8b9c
Revises: 3c4d5e6f7a8b
Create Date: 2026-03-08 13:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "4d5e6f7a8b9c"
down_revision = "3c4d5e6f7a8b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply schema changes."""

    # --- Enums (drop orphans from prior failed runs) ---
    op.execute("DROP TYPE IF EXISTS chat_extraction_status")
    op.execute("DROP TYPE IF EXISTS chat_attachment_scope")

    # --- chat_attachment table ---
    op.create_table(
        "chat_attachment",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_session.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_project.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_message.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("content_type", sa.String(128), nullable=False),
        sa.Column("size_bytes", sa.Integer, nullable=False),
        sa.Column("storage_key", sa.String(1024), nullable=False),
        sa.Column("extracted_text", sa.Text, nullable=True),
        sa.Column(
            "extraction_status",
            sa.Enum(
                "pending", "completed", "failed", "skipped",
                name="chat_extraction_status",
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "scope",
            sa.Enum(
                "session", "project",
                name="chat_attachment_scope",
            ),
            nullable=False,
            server_default="session",
        ),
        sa.Column("page_count", sa.Integer, nullable=True),
        sa.Column("truncated", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_chat_attachment_session", "chat_attachment", ["session_id"])
    op.create_index("ix_chat_attachment_project", "chat_attachment", ["project_id"])
    op.create_index("ix_chat_attachment_message", "chat_attachment", ["message_id"])
    op.create_index(
        "ix_chat_attachment_tenant_user", "chat_attachment", ["tenant_id", "user_id"],
    )

    # --- Add content_parts_json to chat_message ---
    op.add_column(
        "chat_message",
        sa.Column("content_parts_json", postgresql.JSON, nullable=True),
    )


def downgrade() -> None:
    """Reverse schema changes."""
    op.drop_column("chat_message", "content_parts_json")

    op.drop_index("ix_chat_attachment_tenant_user", table_name="chat_attachment")
    op.drop_index("ix_chat_attachment_message", table_name="chat_attachment")
    op.drop_index("ix_chat_attachment_project", table_name="chat_attachment")
    op.drop_index("ix_chat_attachment_session", table_name="chat_attachment")
    op.drop_table("chat_attachment")

    op.execute("DROP TYPE IF EXISTS chat_attachment_scope")
    op.execute("DROP TYPE IF EXISTS chat_extraction_status")
