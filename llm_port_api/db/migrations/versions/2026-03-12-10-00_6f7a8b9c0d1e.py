"""Add LiteLLM columns to llm_provider_instance and extend gateway_provider_type enum.

Revision ID: 6f7a8b9c0d1e
Revises: 5e6f7a8b9c0d
Create Date: 2026-03-12 10:00:00.000000

Changes:
- Extend gateway_provider_type enum with remote_anthropic, remote_google,
  remote_bedrock, remote_azure, remote_mistral, remote_groq, remote_deepseek,
  remote_cohere, remote_custom.
- Add api_key_encrypted (text, nullable) to llm_provider_instance.
- Add litellm_provider (varchar(64), nullable) to llm_provider_instance.
- Add litellm_model (varchar(256), nullable) to llm_provider_instance.
- Add extra_params (json, nullable) to llm_provider_instance.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "6f7a8b9c0d1e"
down_revision = "5e6f7a8b9c0d"
branch_labels = None
depends_on = None

_NEW_ENUM_VALUES = [
    "remote_anthropic",
    "remote_google",
    "remote_bedrock",
    "remote_azure",
    "remote_mistral",
    "remote_groq",
    "remote_deepseek",
    "remote_cohere",
    "remote_custom",
]


def upgrade() -> None:
    # 1. Extend the gateway_provider_type enum with new remote values.
    # Note: DDL statements cannot use bind parameters with asyncpg,
    # so we interpolate directly. Values are hardcoded constants above.
    for val in _NEW_ENUM_VALUES:
        op.execute(f"ALTER TYPE gateway_provider_type ADD VALUE IF NOT EXISTS '{val}'")

    # 2. Add new columns to llm_provider_instance.
    op.add_column(
        "llm_provider_instance",
        sa.Column("api_key_encrypted", sa.Text(), nullable=True),
    )
    op.add_column(
        "llm_provider_instance",
        sa.Column("litellm_provider", sa.String(64), nullable=True),
    )
    op.add_column(
        "llm_provider_instance",
        sa.Column("litellm_model", sa.String(256), nullable=True),
    )
    op.add_column(
        "llm_provider_instance",
        sa.Column("extra_params", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("llm_provider_instance", "extra_params")
    op.drop_column("llm_provider_instance", "litellm_model")
    op.drop_column("llm_provider_instance", "litellm_provider")
    op.drop_column("llm_provider_instance", "api_key_encrypted")
    # Note: PostgreSQL does not support removing values from an enum type.
    # The new enum values will remain after downgrade.
