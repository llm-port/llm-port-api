from typing import Any

import pytest

from llm_port_api.db.models.gateway import PrivacyMode
from llm_port_api.services.gateway.observability import GatewayObservability


class _DummyObservation:
    def __init__(self) -> None:
        self.updated: dict[str, Any] | None = None
        self.ended = False

    def update(self, **kwargs: Any) -> None:
        self.updated = kwargs

    def end(self) -> None:
        self.ended = True


class _DummyClient:
    def __init__(self) -> None:
        self.start_calls: list[dict[str, Any]] = []
        self.event_calls: list[dict[str, Any]] = []
        self.flushed = False

    def create_trace_id(self, *, seed: str) -> str:
        return f"trace-{seed}"

    def start_observation(self, **kwargs: Any) -> _DummyObservation:
        self.start_calls.append(kwargs)
        return _DummyObservation()

    def create_event(self, **kwargs: Any) -> None:
        self.event_calls.append(kwargs)

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.flushed = True


def test_observability_disabled_is_noop() -> None:
    obs = GatewayObservability(enabled=False)
    ctx = obs.start_request_trace(
        request_id="req-1",
        tenant_id="t-1",
        user_id="u-1",
        endpoint="/v1/chat/completions",
        model_alias="qwen3-32b",
        payload={"messages": [{"role": "user", "content": "hello"}]},
        privacy_mode=PrivacyMode.METADATA_ONLY,
        stream=False,
    )
    assert ctx.trace_id is None
    obs.flush()
    obs.shutdown()


def test_observability_enabled_requires_credentials() -> None:
    with pytest.raises(ValueError):
        GatewayObservability(
            enabled=True,
            host="http://langfuse-web:3000",
            public_key=None,
            secret_key=None,
        )


def test_privacy_mode_redacted_sanitizes_input() -> None:
    dummy = _DummyClient()
    obs = GatewayObservability(enabled=True, client=dummy)
    obs.start_request_trace(
        request_id="req-1",
        tenant_id="tenant-a",
        user_id="user-a",
        endpoint="/v1/chat/completions",
        model_alias="qwen3-32b",
        payload={"messages": [{"role": "user", "content": "secret message"}]},
        privacy_mode=PrivacyMode.REDACTED,
        stream=False,
    )
    assert len(dummy.start_calls) == 1
    sent_input = dummy.start_calls[0]["input"]
    assert isinstance(sent_input, dict)
    messages = sent_input["messages"]
    assert isinstance(messages, list)
    assert messages[0]["content"] == "[REDACTED]"


def test_privacy_mode_metadata_only_embeddings_sanitizes_input() -> None:
    dummy = _DummyClient()
    obs = GatewayObservability(enabled=True, client=dummy)
    obs.start_request_trace(
        request_id="req-2",
        tenant_id="tenant-a",
        user_id="user-a",
        endpoint="/v1/embeddings",
        model_alias="text-embedding-3-small",
        payload={"input": "some text"},
        privacy_mode=PrivacyMode.METADATA_ONLY,
        stream=False,
    )
    sent_input = dummy.start_calls[0]["input"]
    assert sent_input == {"input_summary": {"type": "string", "length": 9}}
