from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionRequest(BaseModel):
    """Minimal core contract for chat completions."""

    model: str = Field(min_length=1)
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    session_id: str | None = None

    model_config = ConfigDict(extra="allow")


class EmbeddingsRequest(BaseModel):
    """Minimal core contract for embeddings endpoint."""

    model: str = Field(min_length=1)
    input: Any

    model_config = ConfigDict(extra="allow")


class ModelObject(BaseModel):
    """OpenAI model object shape."""

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ListModelsResponse(BaseModel):
    """OpenAI list models response shape."""

    object: str = "list"
    data: list[ModelObject]
