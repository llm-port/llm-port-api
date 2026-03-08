"""Convert sensitive chat columns to encrypted-compatible types.

Revision ID: 5e6f7a8b9c0d
Revises: 4d5e6f7a8b9c
Create Date: 2026-03-08 14:00:00.000000

Changes:
- content_parts_json: json → text (ciphertext is not valid JSON)
- tool_call_json: json → text
- memory_fact.key: varchar(256) → text (ciphertext exceeds 256 chars)

Note: The ``content``, ``summary_text``, ``value``, ``extracted_text``,
and ``system_instructions`` columns are already ``text`` — no DDL
change needed.  The EncryptedText TypeDecorator handles encryption
transparently at the ORM layer.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "5e6f7a8b9c0d"
down_revision = "4d5e6f7a8b9c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # content_parts_json: json → text
    op.alter_column(
        "chat_message",
        "content_parts_json",
        type_=sa.Text(),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using="content_parts_json::text",
    )

    # tool_call_json: json → text
    op.alter_column(
        "chat_message",
        "tool_call_json",
        type_=sa.Text(),
        existing_type=sa.JSON(),
        existing_nullable=True,
        postgresql_using="tool_call_json::text",
    )

    # memory_fact.key: varchar(256) → text
    op.alter_column(
        "memory_fact",
        "key",
        type_=sa.Text(),
        existing_type=sa.String(256),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "memory_fact",
        "key",
        type_=sa.String(256),
        existing_type=sa.Text(),
        existing_nullable=False,
    )

    op.alter_column(
        "chat_message",
        "tool_call_json",
        type_=sa.JSON(),
        existing_type=sa.Text(),
        existing_nullable=True,
        postgresql_using="tool_call_json::json",
    )

    op.alter_column(
        "chat_message",
        "content_parts_json",
        type_=sa.JSON(),
        existing_type=sa.Text(),
        existing_nullable=True,
        postgresql_using="content_parts_json::json",
    )
