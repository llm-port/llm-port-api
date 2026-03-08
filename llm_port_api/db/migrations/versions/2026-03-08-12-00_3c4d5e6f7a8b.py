"""Add chat projects, sessions, messages, summaries and memory facts.

Revision ID: 3c4d5e6f7a8b
Revises: 2b3c4d5e6f7a
Create Date: 2026-03-08 12:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "3c4d5e6f7a8b"
down_revision = "2b3c4d5e6f7a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Apply schema changes."""

    # --- Enums (drop orphans from prior failed runs) ---
    op.execute("DROP TYPE IF EXISTS chat_session_status")
    op.execute("DROP TYPE IF EXISTS memory_fact_scope")
    op.execute("DROP TYPE IF EXISTS memory_fact_status")

    # --- chat_project ---
    op.create_table(
        "chat_project",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("system_instructions", sa.Text(), nullable=True),
        sa.Column("model_alias", sa.String(length=256), nullable=True),
        sa.Column(
            "metadata_json",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_project_tenant_user",
        "chat_project",
        ["tenant_id", "user_id"],
    )

    # --- chat_session ---
    op.create_table(
        "chat_session",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "active", "archived", "deleted",
                name="chat_session_status",
            ),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "metadata_json",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["chat_project.id"], ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_session_tenant_user_status",
        "chat_session",
        ["tenant_id", "user_id", "status"],
    )

    # --- chat_message ---
    op.create_table(
        "chat_message",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "tool_call_json",
            postgresql.JSON(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("model_alias", sa.String(length=256), nullable=True),
        sa.Column(
            "provider_instance_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("token_estimate", sa.Integer(), nullable=True),
        sa.Column("trace_id", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["chat_session.id"], ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_message_session_id", "chat_message", ["session_id"],
    )
    op.create_index(
        "ix_chat_message_created_at", "chat_message", ["created_at"],
    )

    # --- session_summary ---
    op.create_table(
        "session_summary",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column(
            "last_message_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "token_estimate",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["chat_session.id"], ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_session_summary_session_id", "session_summary", ["session_id"],
    )

    # --- memory_fact ---
    op.create_table(
        "memory_fact",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column(
            "scope",
            sa.Enum(
                "session", "project", "user",
                name="memory_fact_scope",
            ),
            nullable=False,
            server_default="session",
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "project_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("key", sa.String(length=256), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "confidence",
            sa.Float(),
            nullable=False,
            server_default="1.0",
        ),
        sa.Column(
            "source_message_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "candidate", "active", "expired",
                name="memory_fact_status",
            ),
            nullable=False,
            server_default="candidate",
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"], ["chat_session.id"], ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["chat_project.id"], ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_memory_fact_scope_lookup",
        "memory_fact",
        ["tenant_id", "user_id", "scope", "status"],
    )

    # --- DB-level row-count guard for chat_project (defense-in-depth) ---
    # Source: https://stackoverflow.com/a/1743742 by Aleksander Kmetec
    # License: CC BY-SA 2.5 — Retrieved 2026-03-08
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_chat_project_limit()
        RETURNS trigger AS $$
        DECLARE
            max_count INTEGER := 3;
            current_count INTEGER := 0;
            must_check BOOLEAN := false;
        BEGIN
            IF TG_OP = 'INSERT' THEN
                must_check := true;
            END IF;

            IF TG_OP = 'UPDATE' THEN
                IF (NEW.tenant_id != OLD.tenant_id OR NEW.user_id != OLD.user_id) THEN
                    must_check := true;
                END IF;
            END IF;

            IF must_check THEN
                LOCK TABLE chat_project IN EXCLUSIVE MODE;

                SELECT INTO current_count COUNT(*)
                FROM chat_project
                WHERE tenant_id = NEW.tenant_id AND user_id = NEW.user_id;

                IF current_count >= max_count THEN
                    RAISE EXCEPTION
                        'Project limit reached. '
                        'Upgrade to LLM.port Enterprise for unlimited projects.';
                END IF;
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_enforce_chat_project_limit
            BEFORE INSERT OR UPDATE ON chat_project
            FOR EACH ROW EXECUTE PROCEDURE enforce_chat_project_limit();
        """
    )


def downgrade() -> None:
    """Rollback schema changes."""
    op.execute("DROP TRIGGER IF EXISTS trg_enforce_chat_project_limit ON chat_project")
    op.execute("DROP FUNCTION IF EXISTS enforce_chat_project_limit()")

    op.drop_index("ix_memory_fact_scope_lookup", table_name="memory_fact")
    op.drop_table("memory_fact")
    op.drop_index("ix_session_summary_session_id", table_name="session_summary")
    op.drop_table("session_summary")
    op.drop_index("ix_chat_message_created_at", table_name="chat_message")
    op.drop_index("ix_chat_message_session_id", table_name="chat_message")
    op.drop_table("chat_message")
    op.drop_index(
        "ix_chat_session_tenant_user_status", table_name="chat_session",
    )
    op.drop_table("chat_session")
    op.drop_index("ix_chat_project_tenant_user", table_name="chat_project")
    op.drop_table("chat_project")

    op.execute("DROP TYPE IF EXISTS memory_fact_status")
    op.execute("DROP TYPE IF EXISTS memory_fact_scope")
    op.execute("DROP TYPE IF EXISTS chat_session_status")
