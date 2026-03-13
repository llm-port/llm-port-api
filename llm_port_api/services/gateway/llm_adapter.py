"""LiteLLM adapter — unified LLM completion / embedding interface.

Wraps ``litellm.acompletion`` and ``litellm.aembedding`` to provide a
provider-agnostic calling layer.  The gateway service delegates all
upstream calls through this adapter instead of the raw HTTP proxy.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import litellm

from llm_port_api.db.crypto import decrypt_value
from llm_port_api.db.models.gateway import ProviderType

logger = logging.getLogger(__name__)

# ── Provider type → LiteLLM provider prefix mapping ──────────────────
_PROVIDER_PREFIX: dict[str, str] = {
    ProviderType.REMOTE_OPENAI: "openai",
    ProviderType.REMOTE_ANTHROPIC: "anthropic",
    ProviderType.REMOTE_GOOGLE: "gemini",
    ProviderType.REMOTE_BEDROCK: "bedrock",
    ProviderType.REMOTE_AZURE: "azure",
    ProviderType.REMOTE_MISTRAL: "mistral",
    ProviderType.REMOTE_GROQ: "groq",
    ProviderType.REMOTE_DEEPSEEK: "deepseek",
    ProviderType.REMOTE_COHERE: "cohere",
    ProviderType.REMOTE_CUSTOM: "openai",
    # Local inference engines — all speak OpenAI protocol
    ProviderType.VLLM: "openai",
    ProviderType.LLAMACPP: "openai",
    ProviderType.TGI: "openai",
    ProviderType.OLLAMA: "ollama",
}


@dataclass(slots=True, frozen=True)
class CompletionResult:
    """Unified non-streaming completion result."""

    status_code: int
    payload: dict[str, Any]


def _build_litellm_model_name(
    *,
    provider_type: ProviderType,
    litellm_provider: str | None,
    litellm_model: str | None,
    requested_model: str,
) -> str:
    """Build the ``model`` string that LiteLLM expects.

    LiteLLM uses a ``provider/model`` naming convention.  If the user
    has configured an explicit ``litellm_provider`` and ``litellm_model``
    we use those.  Otherwise we derive them from the ``ProviderType``
    and the model alias sent in the request.
    """
    prefix = litellm_provider or _PROVIDER_PREFIX.get(provider_type, "openai")
    model = litellm_model or requested_model

    # For local engines the model name alone suffices (LiteLLM resolves
    # via api_base).  For remote cloud providers we need provider/model.
    if provider_type in (
        ProviderType.VLLM,
        ProviderType.LLAMACPP,
        ProviderType.TGI,
    ):
        return model

    if provider_type == ProviderType.OLLAMA:
        return f"ollama/{model}"

    # Remote providers — use explicit prefix/model
    return f"{prefix}/{model}"


def _resolve_api_key(encrypted_key: str | None) -> str | None:
    """Decrypt the Fernet-encrypted API key stored in the DB."""
    if not encrypted_key:
        return None
    try:
        return decrypt_value(encrypted_key, purpose="provider-api-key")
    except Exception:
        logger.warning("Failed to decrypt provider API key; sending without auth")
        return None


class LLMAdapter:
    """Provider-agnostic adapter backed by LiteLLM."""

    async def completion(
        self,
        *,
        provider_type: ProviderType,
        base_url: str | None,
        api_key_encrypted: str | None,
        litellm_provider: str | None,
        litellm_model: str | None,
        extra_params: dict[str, Any] | None,
        payload: dict[str, Any],
        stream: bool = False,
    ) -> CompletionResult | AsyncIterator[Any]:
        """Run a chat completion (streaming or non-streaming).

        Returns ``CompletionResult`` for non-streaming, or an async
        iterator of ``ModelResponse`` chunks for streaming.
        """
        model_name = _build_litellm_model_name(
            provider_type=provider_type,
            litellm_provider=litellm_provider,
            litellm_model=litellm_model,
            requested_model=payload.get("model", ""),
        )
        api_key = _resolve_api_key(api_key_encrypted)

        # Build kwargs for litellm.acompletion
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": payload.get("messages", []),
            "stream": stream,
        }
        if base_url and not base_url.startswith("litellm://"):
            kwargs["api_base"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

        # Pass through supported OpenAI params
        for key in (
            "temperature", "top_p", "max_tokens", "stop",
            "presence_penalty", "frequency_penalty", "logit_bias",
            "user", "tools", "tool_choice", "response_format",
            "seed", "n",
        ):
            if key in payload:
                kwargs[key] = payload[key]

        # Merge extra_params (custom headers, api_version, etc.)
        if extra_params:
            extra_headers = extra_params.pop("extra_headers", None)
            if extra_headers and isinstance(extra_headers, dict):
                kwargs["extra_headers"] = extra_headers
            # Remaining params go directly to litellm
            kwargs.update(extra_params)

        if stream:
            # Request token usage in the final streaming chunk (OpenAI-compatible).
            kwargs["stream_options"] = {"include_usage": True}
            return self._stream_completion(**kwargs)

        return await self._non_stream_completion(**kwargs)

    async def _non_stream_completion(self, **kwargs: Any) -> CompletionResult:
        """Execute a non-streaming completion."""
        try:
            response = await litellm.acompletion(**kwargs)
            # LiteLLM returns a ModelResponse — convert to dict
            payload = response.model_dump()  # type: ignore[union-attr]
            return CompletionResult(status_code=200, payload=payload)
        except litellm.exceptions.AuthenticationError as exc:
            return CompletionResult(
                status_code=401,
                payload=_error_payload("authentication_error", str(exc)),
            )
        except litellm.exceptions.RateLimitError as exc:
            return CompletionResult(
                status_code=429,
                payload=_error_payload("rate_limit_error", str(exc)),
            )
        except litellm.exceptions.BadRequestError as exc:
            return CompletionResult(
                status_code=400,
                payload=_error_payload("invalid_request_error", str(exc)),
            )
        except Exception as exc:
            logger.exception("LiteLLM completion failed")
            return CompletionResult(
                status_code=502,
                payload=_error_payload("server_error", str(exc)),
            )

    async def _stream_completion(self, **kwargs: Any) -> AsyncIterator[bytes]:
        """Execute a streaming completion, yielding SSE-encoded bytes."""
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            # Pre-stream failure — yield the error as an SSE event so the
            # client receives a structured error instead of a broken stream.
            logger.exception("LiteLLM streaming failed (pre-stream)")
            error_data = _error_payload("server_error", str(exc))
            yield f"data: {json.dumps(error_data)}\n\n".encode()
            yield b"data: [DONE]\n\n"
            return

        try:
            async for chunk in response:  # type: ignore[union-attr]
                data = chunk.model_dump()  # type: ignore[union-attr]
                yield f"data: {json.dumps(data)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("LiteLLM streaming failed (mid-stream)")
            error_data = _error_payload("server_error", str(exc))
            yield f"data: {json.dumps(error_data)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    async def embedding(
        self,
        *,
        provider_type: ProviderType,
        base_url: str | None,
        api_key_encrypted: str | None,
        litellm_provider: str | None,
        litellm_model: str | None,
        extra_params: dict[str, Any] | None,
        payload: dict[str, Any],
    ) -> CompletionResult:
        """Run an embedding request."""
        model_name = _build_litellm_model_name(
            provider_type=provider_type,
            litellm_provider=litellm_provider,
            litellm_model=litellm_model,
            requested_model=payload.get("model", ""),
        )
        api_key = _resolve_api_key(api_key_encrypted)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "input": payload.get("input", ""),
        }
        if base_url and not base_url.startswith("litellm://"):
            kwargs["api_base"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        if extra_params:
            kwargs.update(extra_params)

        try:
            response = await litellm.aembedding(**kwargs)
            return CompletionResult(
                status_code=200,
                payload=response.model_dump(),  # type: ignore[union-attr]
            )
        except Exception as exc:
            logger.exception("LiteLLM embedding failed")
            return CompletionResult(
                status_code=502,
                payload=_error_payload("server_error", str(exc)),
            )


def _error_payload(error_type: str, message: str) -> dict[str, Any]:
    return {
        "error": {
            "type": error_type,
            "message": message,
            "param": None,
            "code": error_type,
        },
    }
