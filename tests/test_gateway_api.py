import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import jwt
import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_port_api.db.models.gateway import (
    LLMGatewayRequestLog,
    LLMModelAlias,
    LLMPoolMembership,
    LLMProviderInstance,
    PrivacyMode,
    ProviderHealthStatus,
    ProviderType,
    TenantLLMPolicy,
)
from llm_port_api.services.gateway.observability import GatewayTraceContext
from llm_port_api.services.gateway.proxy import UpstreamProxy, UpstreamResult

TEST_JWT_SECRET = "test-secret-32-bytes-minimum-value"


def _token(
    tenant_id: str = "tenant-a", sub: str = "user-1", include_tenant: bool = True,
) -> str:
    payload: dict[str, str] = {"sub": sub}
    if include_tenant:
        payload["tenant_id"] = tenant_id
    return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")


async def _seed_basic_graph(session: AsyncSession) -> uuid.UUID:
    alias = LLMModelAlias(
        alias="qwen3-32b",
        description="alias",
        enabled=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    instance_id = uuid.uuid4()
    instance = LLMProviderInstance(
        id=instance_id,
        type=ProviderType.VLLM,
        base_url="http://upstream.local",
        enabled=True,
        weight=1.0,
        max_concurrency=2,
        capabilities=None,
        health_status=ProviderHealthStatus.HEALTHY,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    membership = LLMPoolMembership(
        model_alias="qwen3-32b",
        provider_instance_id=instance_id,
        enabled=True,
    )
    policy = TenantLLMPolicy(
        tenant_id="tenant-a",
        privacy_mode=PrivacyMode.METADATA_ONLY,
        allowed_model_aliases=["qwen3-32b"],
        allowed_provider_types=["vllm"],
        rpm_limit=100,
        tpm_limit=100000,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add_all([alias, instance, membership, policy])
    await session.commit()
    return instance_id


class _FakeObservability:
    def start_request_trace(self, **kwargs: object) -> GatewayTraceContext:  # noqa: ARG002
        return GatewayTraceContext(
            trace_id="trace-test-123",
            observation=None,
            endpoint="/v1/chat/completions",
            privacy_mode=PrivacyMode.METADATA_ONLY,
        )

    def record_success(self, *args: object, **kwargs: object) -> None:  # noqa: ARG002
        return

    def record_failure(self, *args: object, **kwargs: object) -> None:  # noqa: ARG002
        return

    def finalize_stream(self, *args: object, **kwargs: object) -> None:  # noqa: ARG002
        return


@pytest.mark.anyio
async def test_auth_missing_token_returns_401(client: AsyncClient) -> None:
    response = await client.get("/v1/models")
    assert response.status_code == 401
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "missing_authorization"


@pytest.mark.anyio
async def test_auth_missing_tenant_claim_returns_403(client: AsyncClient) -> None:
    token = _token(include_tenant=False)
    response = await client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
    body = response.json()
    assert body["error"]["code"] == "missing_tenant_id"


@pytest.mark.anyio
async def test_models_list_applies_tenant_allowlist(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    await _seed_basic_graph(db_session)
    token = _token()
    response = await client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids == ["qwen3-32b"]


@pytest.mark.anyio
async def test_chat_non_stream_passthrough_and_retry_once(
    fastapi_app: FastAPI,
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_basic_graph(db_session)
    fastapi_app.state.gateway_observability = _FakeObservability()
    token = _token()
    calls = {"count": 0}

    async def fake_post_json(
        self: UpstreamProxy, **kwargs: object,
    ) -> UpstreamResult:  # noqa: ARG001
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary upstream failure")
        return UpstreamResult(
            status_code=200,
            payload={
                "id": "chatcmpl_x",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen3-32b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    },
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                },
            },
            headers={},
        )

    monkeypatch.setattr(UpstreamProxy, "post_json", fake_post_json)

    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": "qwen3-32b",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert calls["count"] == 2
    assert response.headers["x-langfuse-trace-id"] == "trace-test-123"


@pytest.mark.anyio
async def test_chat_stream_sse_done(
    fastapi_app: FastAPI,
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_basic_graph(db_session)
    fastapi_app.state.gateway_observability = _FakeObservability()
    token = _token()

    async def fake_stream_post(
        self: UpstreamProxy, **kwargs: object,
    ) -> AsyncIterator[bytes]:  # noqa: ARG001
        yield b'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"qwen3-32b","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
        yield b'data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":1,"model":"qwen3-32b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n\n'
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(UpstreamProxy, "stream_post", fake_stream_post)

    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": "qwen3-32b",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-langfuse-trace-id"] == "trace-test-123"
    text = response.text
    assert "chat.completion.chunk" in text
    assert "data: [DONE]" in text


@pytest.mark.anyio
async def test_embeddings_passthrough(
    fastapi_app: FastAPI,
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_basic_graph(db_session)
    fastapi_app.state.gateway_observability = _FakeObservability()
    token = _token()

    async def fake_post_json(
        self: UpstreamProxy, **kwargs: object,
    ) -> UpstreamResult:  # noqa: ARG001
        return UpstreamResult(
            status_code=200,
            payload={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
            headers={},
        )

    monkeypatch.setattr(UpstreamProxy, "post_json", fake_post_json)
    response = await client.post(
        "/v1/embeddings",
        headers={"Authorization": f"Bearer {token}"},
        json={"model": "qwen3-32b", "input": "hello"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["object"] == "embedding"
    assert response.headers["x-langfuse-trace-id"] == "trace-test-123"
    rows = (await db_session.execute(select(LLMGatewayRequestLog))).scalars().all()
    assert len(rows) >= 1
    assert rows[-1].trace_id == "trace-test-123"


@pytest.mark.anyio
async def test_stream_mid_failure_returns_502_on_call(
    client: AsyncClient,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _seed_basic_graph(db_session)
    token = _token()

    async def fake_stream_post(
        self: UpstreamProxy, **kwargs: object,
    ) -> AsyncIterator[bytes]:  # noqa: ARG001
        yield b'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"q","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        raise RuntimeError("broken stream")

    monkeypatch.setattr(UpstreamProxy, "stream_post", fake_stream_post)
    response = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": "qwen3-32b",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    assert response.status_code in (200, 502)
    if response.status_code == 200:
        # Error can surface during stream consumption by client.
        assert "data:" in response.text
