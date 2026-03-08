"""Pydantic schemas for projects, sessions, messages, summaries and memory facts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ── Projects ──────────────────────────────────────────────────────


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = None
    system_instructions: str | None = None
    model_alias: str | None = None
    metadata_json: dict[str, Any] | None = None


class ProjectUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = None
    system_instructions: str | None = None
    model_alias: str | None = None
    metadata_json: dict[str, Any] | None = None


class ProjectDTO(BaseModel):
    id: uuid.UUID
    tenant_id: str
    user_id: str
    name: str
    description: str | None
    system_instructions: str | None
    model_alias: str | None
    metadata_json: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Sessions ──────────────────────────────────────────────────────


class SessionCreateRequest(BaseModel):
    project_id: uuid.UUID | None = None
    title: str | None = Field(default=None, max_length=512)
    metadata_json: dict[str, Any] | None = None


class SessionUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=512)
    status: str | None = None
    metadata_json: dict[str, Any] | None = None


class SessionDTO(BaseModel):
    id: uuid.UUID
    tenant_id: str
    user_id: str
    project_id: uuid.UUID | None
    title: str | None
    status: str
    metadata_json: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Messages ──────────────────────────────────────────────────────


class MessageDTO(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    content_parts: list[dict[str, Any]] | None = None
    tool_call_json: dict[str, Any] | None
    model_alias: str | None
    provider_instance_id: uuid.UUID | None
    token_estimate: int | None
    trace_id: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_model(cls, msg: Any) -> "MessageDTO":
        """Create a DTO, mapping ``content_parts_json`` to ``content_parts``."""
        return cls(
            id=msg.id,
            session_id=msg.session_id,
            role=msg.role,
            content=msg.content,
            content_parts=msg.content_parts_json,
            tool_call_json=msg.tool_call_json,
            model_alias=msg.model_alias,
            provider_instance_id=msg.provider_instance_id,
            token_estimate=msg.token_estimate,
            trace_id=msg.trace_id,
            created_at=msg.created_at,
        )


# ── Summaries ─────────────────────────────────────────────────────


class SummaryDTO(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    summary_text: str
    last_message_id: uuid.UUID
    token_estimate: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Memory Facts ──────────────────────────────────────────────────


class MemoryFactCreateRequest(BaseModel):
    scope: str = Field(default="user", pattern=r"^(session|project|user)$")
    session_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    key: str = Field(min_length=1, max_length=256)
    value: str = Field(min_length=1)
    confidence: float = 1.0


class MemoryFactUpdateRequest(BaseModel):
    value: str | None = None
    confidence: float | None = None
    status: str | None = Field(default=None, pattern=r"^(candidate|active|expired)$")


class MemoryFactDTO(BaseModel):
    id: uuid.UUID
    tenant_id: str
    user_id: str
    scope: str
    session_id: uuid.UUID | None
    project_id: uuid.UUID | None
    key: str
    value: str
    confidence: float
    source_message_id: uuid.UUID | None
    status: str
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ── Attachments ───────────────────────────────────────────────────


class AttachmentDTO(BaseModel):
    id: uuid.UUID
    tenant_id: str
    user_id: str
    session_id: uuid.UUID | None
    project_id: uuid.UUID | None
    message_id: uuid.UUID | None
    filename: str
    content_type: str
    size_bytes: int
    extraction_status: str
    scope: str
    page_count: int | None
    truncated: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AttachmentUploadResponse(BaseModel):
    attachment: AttachmentDTO
    extracted_text_length: int
    token_estimate: int


class AttachmentStatsDTO(BaseModel):
    total_count: int
    total_bytes: int
