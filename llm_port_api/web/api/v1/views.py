from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette import status

from llm_port_api.db.dao.gateway_dao import GatewayDAO
from llm_port_api.services.gateway.audit import AuditService
from llm_port_api.services.gateway.auth import AuthContext, get_auth_context
from llm_port_api.services.gateway.errors import GatewayError, error_response
from llm_port_api.services.gateway.lease import LeaseManager
from llm_port_api.services.gateway.observability import GatewayObservability
from llm_port_api.services.gateway.pii_client import PIIClient
from llm_port_api.services.gateway.proxy import UpstreamProxy
from llm_port_api.services.gateway.ratelimit import RateLimiter
from llm_port_api.services.gateway.routing import RouterService
from llm_port_api.services.gateway.schemas import (
    ChatCompletionRequest,
    EmbeddingsRequest,
)
from llm_port_api.services.gateway.service import GatewayService
from llm_port_api.services.registry import service_registry
from llm_port_api.settings import settings

router = APIRouter()


@router.get("/health")
def public_health_check() -> None:
    """Public health endpoint for L7 probes."""


@router.get("/v1/services")
async def list_services(request: Request) -> JSONResponse:
    """Return the manifest of optional service modules.

    The frontend uses this to discover which features (PII, Auth, RAG, ...)
    are available and healthy so it can show/hide UI sections accordingly.
    """
    registry = request.app.state.service_registry
    # Run async health checks against enabled services
    await registry.check_health(request.app.state.http_client)
    return JSONResponse(status_code=200, content=registry.to_dict())


def get_gateway_service(
    request: Request,
    dao: GatewayDAO = Depends(),
) -> GatewayService:
    """Build gateway service with request-scoped dependencies."""
    redis_pool = request.app.state.redis_pool
    lease_manager = LeaseManager(redis_pool, ttl_sec=settings.lease_ttl_sec)
    router_service = RouterService(
        dao=dao,
        redis_pool=redis_pool,
        lease_manager=lease_manager,
    )
    proxy = UpstreamProxy(client=request.app.state.http_client)
    limiter = RateLimiter(redis_pool)
    audit = AuditService(dao)
    observability: GatewayObservability = request.app.state.gateway_observability

    # PII client (optional - only when PII module is enabled in registry)
    pii_client: PIIClient | None = None
    pii_url = service_registry.get_url("pii")
    if pii_url:
        pii_client = PIIClient(
            base_url=pii_url,
            http_client=request.app.state.http_client,
        )

    return GatewayService(
        dao=dao,
        router=router_service,
        proxy=proxy,
        limiter=limiter,
        audit=audit,
        observability=observability,
        pii_client=pii_client,
    )


@router.get("/v1/models")
async def list_models(
    auth: AuthContext = Depends(get_auth_context),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    """List model aliases available for tenant."""
    try:
        payload = await service.list_models(auth)
        return JSONResponse(status_code=200, content=payload)
    except GatewayError as exc:
        return error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_type=exc.error_type,
            param=exc.param,
            code=exc.code,
        )


@router.post("/v1/embeddings")
async def create_embeddings(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    service: GatewayService = Depends(get_gateway_service),
) -> JSONResponse:
    """Proxy embeddings request through shared pipeline."""
    try:
        payload = await _get_json_payload(request)
        EmbeddingsRequest.model_validate(payload)
        request_id = _request_id(request)
        routed = await service.route_non_stream(
            auth=auth,
            endpoint="/v1/embeddings",
            payload=payload,
            request_id=request_id,
        )
        response = JSONResponse(status_code=routed.status_code, content=routed.payload)
        response.headers["x-request-id"] = request_id
        response.headers["x-provider-instance-id"] = routed.provider_instance_id
        if routed.trace_id:
            response.headers["x-langfuse-trace-id"] = routed.trace_id
        return response
    except GatewayError as exc:
        return error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_type=exc.error_type,
            param=exc.param,
            code=exc.code,
        )


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completions(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    service: GatewayService = Depends(get_gateway_service),
) -> Response:
    """Proxy chat completions with stream and non-stream support."""
    try:
        payload = await _get_json_payload(request)
        parsed = ChatCompletionRequest.model_validate(payload)
        request_id = _request_id(request)

        if parsed.stream:
            streamed = await service.route_stream_chat(
                auth=auth,
                payload=payload,
                request_id=request_id,
            )
            stream_response = StreamingResponse(
                streamed.stream,
                media_type="text/event-stream",
            )
            stream_response.headers["x-request-id"] = request_id
            stream_response.headers["x-provider-instance-id"] = (
                streamed.provider_instance_id
            )
            stream_response.headers["cache-control"] = "no-cache"
            if streamed.trace_id:
                stream_response.headers["x-langfuse-trace-id"] = streamed.trace_id
            return stream_response

        non_stream = await service.route_non_stream(
            auth=auth,
            endpoint="/v1/chat/completions",
            payload=payload,
            request_id=request_id,
        )
        json_response = JSONResponse(
            status_code=non_stream.status_code,
            content=non_stream.payload,
        )
        json_response.headers["x-request-id"] = request_id
        json_response.headers["x-provider-instance-id"] = (
            non_stream.provider_instance_id
        )
        if non_stream.trace_id:
            json_response.headers["x-langfuse-trace-id"] = non_stream.trace_id
        return json_response
    except GatewayError as exc:
        return error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_type=exc.error_type,
            param=exc.param,
            code=exc.code,
        )


def _request_id(request: Request) -> str:
    header_val = request.headers.get("x-request-id", "").strip()
    return header_val or str(uuid.uuid4())


async def _get_json_payload(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > settings.request_max_body_bytes:
                raise GatewayError(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    message="Request body exceeds size limit.",
                    code="request_too_large",
                )
        except ValueError:
            pass
    body = await request.body()
    if len(body) > settings.request_max_body_bytes:
        raise GatewayError(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            message="Request body exceeds size limit.",
            code="request_too_large",
        )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GatewayError(
            status_code=400,
            message="Request body must be valid JSON.",
            code="invalid_json",
        ) from exc
    if not isinstance(payload, dict):
        raise GatewayError(
            status_code=400,
            message="Request body must be a JSON object.",
            code="invalid_json_body",
        )
    return payload
