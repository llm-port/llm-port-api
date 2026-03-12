import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
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
from llm_port_api.db.crypto import EncryptedJSON, EncryptedText


class ProviderType(enum.StrEnum):
    """Gateway provider instance type."""

    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    TGI = "tgi"
    OLLAMA = "ollama"
    REMOTE_OPENAI = "remote_openai"
    REMOTE_ANTHROPIC = "remote_anthropic"
    REMOTE_GOOGLE = "remote_google"
    REMOTE_BEDROCK = "remote_bedrock"
    REMOTE_AZURE = "remote_azure"
    REMOTE_MISTRAL = "remote_mistral"
    REMOTE_GROQ = "remote_groq"
    REMOTE_DEEPSEEK = "remote_deepseek"
    REMOTE_COHERE = "remote_cohere"
    REMOTE_CUSTOM = "remote_custom"


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


class SessionStatus(enum.StrEnum):
    """Chat session lifecycle status."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemoryFactScope(enum.StrEnum):
    """Scope of a durable memory fact."""

    SESSION = "session"
    PROJECT = "project"
    USER = "user"


class MemoryFactStatus(enum.StrEnum):
    """Lifecycle status of a memory fact."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    EXPIRED = "expired"


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
    api_key_encrypted: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        doc="Fernet-encrypted provider API key.",
    )
    litellm_provider: Mapped[str | None] = mapped_column(
        String(64), nullable=True,
        doc="LiteLLM provider prefix (e.g. 'anthropic', 'vertex_ai').",
    )
    litellm_model: Mapped[str | None] = mapped_column(
        String(256), nullable=True,
        doc="LiteLLM model identifier (e.g. 'claude-sonnet-4-20250514').",
    )
    extra_params: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True,
        doc="Provider-specific params (region, project_id, api_version, custom headers).",
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


# ---------------------------------------------------------------------------
# Chat: Projects, Sessions, Messages
# ---------------------------------------------------------------------------


class ChatProject(Base):
    """A project groups related sessions and scopes memory."""

    __tablename__ = "chat_project"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_instructions: Mapped[str | None] = mapped_column(
        EncryptedText("chat-content"), nullable=True,
    )
    model_alias: Mapped[str | None] = mapped_column(String(256), nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_chat_project_tenant_user", "tenant_id", "user_id"),
    )


class ChatSession(Base):
    """A single chat conversation thread."""

    __tablename__ = "chat_session"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chat_project.id", ondelete="CASCADE"),
        nullable=True,
    )
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[SessionStatus] = mapped_column(
        SAEnum(
            SessionStatus,
            name="chat_session_status",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=SessionStatus.ACTIVE,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_chat_session_tenant_user_status", "tenant_id", "user_id", "status"),
    )


class ChatMessage(Base):
    """An individual message within a chat session."""

    __tablename__ = "chat_message"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chat_session.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(EncryptedText("chat-content"), nullable=False)
    content_parts_json: Mapped[list[Any] | None] = mapped_column(
        EncryptedJSON("chat-content"), nullable=True,
    )
    tool_call_json: Mapped[dict[str, Any] | None] = mapped_column(
        EncryptedJSON("chat-content"), nullable=True,
    )
    model_alias: Mapped[str | None] = mapped_column(String(256), nullable=True)
    provider_instance_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True,
    )
    token_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), index=True,
    )


# ---------------------------------------------------------------------------
# Summaries & Memory Facts
# ---------------------------------------------------------------------------


class SessionSummary(Base):
    """Rolling summary of a session's conversation."""

    __tablename__ = "session_summary"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chat_session.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    summary_text: Mapped[str] = mapped_column(
        EncryptedText("chat-content"), nullable=False,
    )
    last_message_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), nullable=False,
    )
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )


class MemoryFact(Base):
    """A durable memory fact extracted from or manually added to a session."""

    __tablename__ = "memory_fact"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    scope: Mapped[MemoryFactScope] = mapped_column(
        SAEnum(
            MemoryFactScope,
            name="memory_fact_scope",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=MemoryFactScope.SESSION,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chat_session.id", ondelete="SET NULL"),
        nullable=True,
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chat_project.id", ondelete="SET NULL"),
        nullable=True,
    )
    key: Mapped[str] = mapped_column(EncryptedText("chat-memory"), nullable=False)
    value: Mapped[str] = mapped_column(EncryptedText("chat-memory"), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    source_message_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True,
    )
    status: Mapped[MemoryFactStatus] = mapped_column(
        SAEnum(
            MemoryFactStatus,
            name="memory_fact_status",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=MemoryFactStatus.CANDIDATE,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now(),
    )

    __table_args__ = (
        Index(
            "ix_memory_fact_scope_lookup",
            "tenant_id", "user_id", "scope", "status",
        ),
    )


# ---------------------------------------------------------------------------
# Chat Attachments
# ---------------------------------------------------------------------------


class AttachmentScope(enum.StrEnum):
    """Scope of a chat attachment."""

    SESSION = "session"
    PROJECT = "project"


class ExtractionStatus(enum.StrEnum):
    """Text extraction status for an attachment."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ChatAttachment(Base):
    """A file attached to a chat session or project."""

    __tablename__ = "chat_attachment"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chat_session.id", ondelete="CASCADE"),
        nullable=True,
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chat_project.id", ondelete="CASCADE"),
        nullable=True,
    )
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chat_message.id", ondelete="SET NULL"),
        nullable=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    extracted_text: Mapped[str | None] = mapped_column(
        EncryptedText("chat-attachments"), nullable=True,
    )
    extraction_status: Mapped[ExtractionStatus] = mapped_column(
        SAEnum(
            ExtractionStatus,
            name="chat_extraction_status",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=ExtractionStatus.PENDING,
    )
    scope: Mapped[AttachmentScope] = mapped_column(
        SAEnum(
            AttachmentScope,
            name="chat_attachment_scope",
            create_type=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=False,
        default=AttachmentScope.SESSION,
    )
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_chat_attachment_session", "session_id"),
        Index("ix_chat_attachment_project", "project_id"),
        Index("ix_chat_attachment_message", "message_id"),
        Index("ix_chat_attachment_tenant_user", "tenant_id", "user_id"),
    )
