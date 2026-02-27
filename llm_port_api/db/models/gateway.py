import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from llm_port_api.db.base import Base


class ProviderType(enum.StrEnum):
    """Gateway provider instance type."""

    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    TGI = "tgi"
    OLLAMA = "ollama"
    REMOTE_OPENAI = "remote_openai"


class ProviderHealthStatus(enum.StrEnum):
    """Health of a provider instance."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class PrivacyMode(enum.StrEnum):
    """Tracing payload policy for tenant data."""

    FULL = "full"
    REDACTED = "redacted"
    METADATA_ONLY = "metadata_only"


class LLMModelAlias(Base):
    """Logical model aliases exposed on /v1/models."""

    __tablename__ = "llm_model_alias"

    alias: Mapped[str] = mapped_column(String(256), primary_key=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_parameters: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True,
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LLMProviderInstance(Base):
    """Concrete upstream instance that can serve aliases."""

    __tablename__ = "llm_provider_instance"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    type: Mapped[ProviderType] = mapped_column(
        SAEnum(
            ProviderType,
            name="gateway_provider_type",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
    )
    base_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    max_concurrency: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    capabilities: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    health_status: Mapped[ProviderHealthStatus] = mapped_column(
        SAEnum(
            ProviderHealthStatus,
            name="gateway_provider_health_status",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=ProviderHealthStatus.UNKNOWN,
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LLMPoolMembership(Base):
    """Mapping of alias -> provider instances with optional weight override."""

    __tablename__ = "llm_pool_membership"

    model_alias: Mapped[str] = mapped_column(
        String(256),
        ForeignKey("llm_model_alias.alias", ondelete="CASCADE"),
        primary_key=True,
    )
    provider_instance_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("llm_provider_instance.id", ondelete="CASCADE"),
        primary_key=True,
    )
    weight_override: Mapped[float | None] = mapped_column(Float, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class TenantLLMPolicy(Base):
    """Tenant policy controls model/provider access, limits and PII."""

    __tablename__ = "tenant_llm_policy"

    tenant_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    privacy_mode: Mapped[PrivacyMode] = mapped_column(
        SAEnum(
            PrivacyMode,
            name="gateway_privacy_mode",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=PrivacyMode.METADATA_ONLY,
    )
    allowed_model_aliases: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    allowed_provider_types: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True,
    )
    rpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tpm_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ----- PII policy (nullable = feature disabled) -----
    pii_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        doc=(
            "JSON PII policy: "
            "{telemetry: {enabled, mode, store_raw}, "
            " egress: {enabled_for_cloud, enabled_for_local, mode, fail_action}, "
            " presidio: {language, threshold, entities}}"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class LLMGatewayRequestLog(Base):
    """Audit log for routed gateway requests."""

    __tablename__ = "llm_gateway_request_log"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    request_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_alias: Mapped[str | None] = mapped_column(String(256), nullable=True)
    provider_instance_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    endpoint: Mapped[str] = mapped_column(String(128), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
