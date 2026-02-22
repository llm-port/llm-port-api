from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from llm_port_api.db.models.gateway import PrivacyMode

try:  # pragma: no cover - import guard for optional runtime integration.
    from langfuse import Langfuse
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment]


@dataclass(slots=True, frozen=True)
class GatewayTraceContext:
    """Request-scoped Langfuse trace context."""

    trace_id: str | None
    observation: Any | None
    endpoint: str
    privacy_mode: PrivacyMode


class GatewayObservability:
    """Langfuse adapter for gateway request tracing."""

    def __init__(
        self,
        *,
        enabled: bool,
        host: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
        tracing_enabled: bool = True,
        release: str | None = None,
        debug: bool = False,
        client: Any | None = None,
    ) -> None:
        self.enabled = enabled
        if not enabled:
            self._client: Any | None = None
            return
        if client is not None:
            self._client = client
            return
        if not host or not public_key or not secret_key:
            msg = "Langfuse requires host, public key, and secret key when enabled."
            raise ValueError(msg)
        if Langfuse is None:
            msg = "langfuse package is not installed but integration is enabled."
            raise ValueError(msg)
        self._client = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
            tracing_enabled=tracing_enabled,
            release=release,
            debug=debug,
        )

    def start_request_trace(
        self,
        *,
        request_id: str,
        tenant_id: str,
        user_id: str,
        endpoint: str,
        model_alias: str,
        payload: dict[str, Any],
        privacy_mode: PrivacyMode | None,
        stream: bool,
    ) -> GatewayTraceContext:
        """Create one gateway trace and root observation."""
        mode = privacy_mode or PrivacyMode.METADATA_ONLY
        if not self.enabled or self._client is None:
            return GatewayTraceContext(
                trace_id=None, observation=None, endpoint=endpoint, privacy_mode=mode,
            )
        trace_id = self._safe_create_trace_id(seed=request_id) or request_id
        metadata = {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "endpoint": endpoint,
            "model_alias": model_alias,
            "stream": stream,
        }
        sanitized_input = self._sanitize_input(
            endpoint=endpoint, payload=payload, mode=mode,
        )
        observation = None
        try:
            observation = self._client.start_observation(
                name=self._observation_name(endpoint),
                as_type="generation",
                trace_context={"trace_id": trace_id},
                input=sanitized_input,
                metadata=metadata,
                model=model_alias,
            )
        except Exception:
            observation = None
        return GatewayTraceContext(
            trace_id=trace_id,
            observation=observation,
            endpoint=endpoint,
            privacy_mode=mode,
        )

    def record_success(
        self,
        context: GatewayTraceContext,
        *,
        status_code: int,
        latency_ms: int,
        ttft_ms: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        provider_instance_id: str | None,
        output_payload: dict[str, Any] | None,
    ) -> None:
        """Record successful request completion details."""
        self._finalize(
            context=context,
            status_code=status_code,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            provider_instance_id=provider_instance_id,
            error_code=None,
            output_payload=output_payload,
        )

    def record_failure(
        self,
        context: GatewayTraceContext,
        *,
        status_code: int,
        latency_ms: int,
        provider_instance_id: str | None,
        error_code: str | None,
        error_message: str,
    ) -> None:
        """Record failed request completion details."""
        self._finalize(
            context=context,
            status_code=status_code,
            latency_ms=latency_ms,
            ttft_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            provider_instance_id=provider_instance_id,
            error_code=error_code,
            output_payload={"error_message": error_message},
        )

    def finalize_stream(
        self,
        context: GatewayTraceContext,
        *,
        status_code: int,
        latency_ms: int,
        ttft_ms: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        provider_instance_id: str | None,
        error_code: str | None,
    ) -> None:
        """Finalize streaming observation with usage/timing/error state."""
        self._finalize(
            context=context,
            status_code=status_code,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            provider_instance_id=provider_instance_id,
            error_code=error_code,
            output_payload=None,
        )

    def flush(self) -> None:
        """Flush Langfuse event queue."""
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            return

    def shutdown(self) -> None:
        """Flush and shutdown client transport."""
        if self._client is None:
            return
        self.flush()
        shutdown = getattr(self._client, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                return

    @staticmethod
    def _observation_name(endpoint: str) -> str:
        return endpoint.removeprefix("/v1/").replace("/", "_")

    def _finalize(
        self,
        *,
        context: GatewayTraceContext,
        status_code: int,
        latency_ms: int,
        ttft_ms: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        provider_instance_id: str | None,
        error_code: str | None,
        output_payload: dict[str, Any] | None,
    ) -> None:
        if not self.enabled or self._client is None:
            return
        metadata: dict[str, Any] = {
            "status_code": status_code,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "provider_instance_id": provider_instance_id,
            "error_code": error_code,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        observation = context.observation
        if observation is not None:
            with contextlib.suppress(Exception):
                observation.update(
                    output=self._sanitize_output(
                        endpoint=context.endpoint,
                        payload=output_payload,
                        mode=context.privacy_mode,
                    ),
                    metadata=metadata,
                )
            end_fn = getattr(observation, "end", None)
            if callable(end_fn):
                with contextlib.suppress(Exception):
                    end_fn()
            return
        if context.trace_id is None:
            return
        try:
            self._client.create_event(
                trace_id=context.trace_id,
                name=f"{self._observation_name(context.endpoint)}_result",
                metadata=metadata,
            )
        except Exception:
            return

    def _safe_create_trace_id(self, *, seed: str) -> str | None:
        if self._client is None:
            return None
        create_trace_id = getattr(self._client, "create_trace_id", None)
        if not callable(create_trace_id):
            return None
        try:
            return str(create_trace_id(seed=seed))
        except Exception:
            return None

    def _sanitize_input(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        mode: PrivacyMode,
    ) -> dict[str, Any]:
        if mode == PrivacyMode.METADATA_ONLY:
            return self._metadata_only_input(endpoint=endpoint, payload=payload)
        if endpoint == "/v1/chat/completions":
            return {"messages": self._sanitize_messages(payload.get("messages"), mode)}
        if endpoint == "/v1/embeddings":
            return {"input": self._sanitize_embeddings_input(payload.get("input"), mode)}
        return self._metadata_only_input(endpoint=endpoint, payload=payload)

    def _sanitize_output(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any] | None,
        mode: PrivacyMode,
    ) -> dict[str, Any] | None:
        if payload is None:
            return None
        if mode == PrivacyMode.METADATA_ONLY:
            return {"output_present": True}
        if endpoint == "/v1/chat/completions":
            return self._sanitize_chat_output(payload, mode)
        if endpoint == "/v1/embeddings":
            return self._sanitize_embeddings_output(payload, mode)
        return {"output_present": True}

    @staticmethod
    def _metadata_only_input(
        *,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if endpoint == "/v1/chat/completions":
            messages = payload.get("messages")
            count = len(messages) if isinstance(messages, list) else 0
            return {"messages_count": count}
        if endpoint == "/v1/embeddings":
            input_value = payload.get("input")
            return {"input_summary": GatewayObservability._input_summary(input_value)}
        return {"keys": sorted(payload.keys())}

    @staticmethod
    def _sanitize_messages(messages: Any, mode: PrivacyMode) -> list[dict[str, Any]]:
        if not isinstance(messages, list):
            return []
        sanitized: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", ""))
            content = message.get("content")
            if mode == PrivacyMode.FULL:
                sanitized.append({"role": role, "content": content})
            else:
                sanitized.append(
                    {
                        "role": role,
                        "content": "[REDACTED]",
                        "content_length": len(str(content)) if content is not None else 0,
                    },
                )
        return sanitized

    @staticmethod
    def _sanitize_embeddings_input(value: Any, mode: PrivacyMode) -> Any:
        if mode == PrivacyMode.FULL:
            return value
        summary = GatewayObservability._input_summary(value)
        return {"redacted": True, "summary": summary}

    @staticmethod
    def _sanitize_chat_output(
        payload: dict[str, Any], mode: PrivacyMode,
    ) -> dict[str, Any]:
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return {"choices": []}
        sanitized_choices: list[dict[str, Any]] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            message = choice.get("message")
            if not isinstance(message, dict):
                sanitized_choices.append({"finish_reason": finish_reason})
                continue
            content = message.get("content")
            if mode == PrivacyMode.FULL:
                sanitized_choices.append(
                    {
                        "finish_reason": finish_reason,
                        "message": {
                            "role": message.get("role"),
                            "content": content,
                        },
                    },
                )
            else:
                sanitized_choices.append(
                    {
                        "finish_reason": finish_reason,
                        "message": {
                            "role": message.get("role"),
                            "content": "[REDACTED]",
                            "content_length": len(str(content))
                            if content is not None
                            else 0,
                        },
                    },
                )
        return {"choices": sanitized_choices}

    @staticmethod
    def _sanitize_embeddings_output(payload: dict[str, Any], mode: PrivacyMode) -> dict[str, Any]:
        del mode
        data = payload.get("data")
        data_count = len(data) if isinstance(data, list) else 0
        return {"object": payload.get("object"), "data_count": data_count}

    @staticmethod
    def _input_summary(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            return {"type": "string", "length": len(value)}
        if isinstance(value, list):
            return {"type": "list", "length": len(value)}
        if isinstance(value, dict):
            return {"type": "object", "keys": sorted(value.keys())}
        if value is None:
            return {"type": "null"}
        return {"type": type(value).__name__}
