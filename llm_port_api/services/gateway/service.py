from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from llm_port_api.db.dao.gateway_dao import GatewayDAO
from llm_port_api.db.models.gateway import ProviderType
from llm_port_api.services.gateway.audit import AuditService
from llm_port_api.services.gateway.auth import AuthContext
from llm_port_api.services.gateway.errors import GatewayError
from llm_port_api.services.gateway.observability import (
    GatewayObservability,
)
from llm_port_api.services.gateway.pii_client import PIIClient
from llm_port_api.services.gateway.pii_policy import PIIPolicy, parse_pii_policy
from llm_port_api.services.gateway.proxy import UpstreamProxy, UpstreamResult
from llm_port_api.services.gateway.ratelimit import RateLimiter
from llm_port_api.services.gateway.routing import RouterService
from llm_port_api.services.gateway.stream import StreamStats, wrap_sse_stream
from llm_port_api.services.gateway.usage import (
    estimate_input_tokens,
    usage_from_payload,
)
from llm_port_api.settings import settings

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class GatewayResponse:
    """Structured non-streaming gateway output."""

    status_code: int
    payload: dict[str, Any]
    provider_instance_id: str
    latency_ms: int
    trace_id: str | None = None


@dataclass(slots=True, frozen=True)
class StreamingGatewayResponse:
    """Structured streaming gateway output."""

    stream: AsyncIterator[bytes]
    provider_instance_id: str
    latency_ms: int
    stats: StreamStats
    trace_id: str | None = None


class GatewayService:
    """Core shared pipeline for chat + embeddings + models."""

    def __init__(
        self,
        *,
        dao: GatewayDAO,
        router: RouterService,
        proxy: UpstreamProxy,
        limiter: RateLimiter,
        audit: AuditService,
        observability: GatewayObservability,
        pii_client: PIIClient | None = None,
    ) -> None:
        self.dao = dao
        self.router = router
        self.proxy = proxy
        self.limiter = limiter
        self.audit = audit
        self.observability = observability
        self.pii_client = pii_client

    async def list_models(self, auth: AuthContext) -> dict[str, Any]:
        aliases = await self.dao.list_enabled_aliases_for_tenant(auth.tenant_id)
        return {
            "object": "list",
            "data": [
                {
                    "id": alias.alias,
                    "object": "model",
                    "created": int(alias.created_at.timestamp()),
                    "owned_by": "llm-port",
                }
                for alias in aliases
            ],
        }

    async def route_non_stream(
        self,
        *,
        auth: AuthContext,
        endpoint: str,
        payload: dict[str, Any],
        request_id: str,
    ) -> GatewayResponse:
        started = time.perf_counter()
        model_alias = _require_model(payload)
        policy = await self.dao.get_tenant_policy(auth.tenant_id)

        await _check_limits(
            limiter=self.limiter,
            tenant_id=auth.tenant_id,
            payload=payload,
            rpm_limit=policy.rpm_limit if policy else None,
            tpm_limit=policy.tpm_limit if policy else None,
        )

        candidates = await self.router.resolve_alias(
            alias=model_alias, tenant_id=auth.tenant_id,
        )
        decision = await self.router.pick_and_lease(
            candidates=candidates, request_id=request_id,
        )

        # --- PII policy resolution ---
        pii_policy = parse_pii_policy(
            policy.pii_config if policy else None,
        )
        egress_payload = payload
        token_mapping: dict[str, str] | None = None
        is_cloud = decision.candidate.provider_type == ProviderType.REMOTE_OPENAI

        if pii_policy and self.pii_client:
            egress_payload, token_mapping = await self._apply_egress_pii(
                payload=payload,
                pii_policy=pii_policy,
                is_cloud=is_cloud,
                request_id=request_id,
            )

        # Payload used for observability (may be further sanitized)
        obs_payload = egress_payload
        if pii_policy and self.pii_client and pii_policy.telemetry.enabled:
            obs_payload = await self._apply_telemetry_pii(
                payload=payload,
                pii_policy=pii_policy,
                request_id=request_id,
            )

        result: UpstreamResult | None = None
        error_code: str | None = None
        status_code = 500
        usage_prompt = None
        usage_completion = None
        usage_total = None
        trace_context = self.observability.start_request_trace(
            request_id=request_id,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            endpoint=endpoint,
            model_alias=model_alias,
            payload=obs_payload,
            privacy_mode=policy.privacy_mode if policy else None,
            stream=False,
        )
        try:
            for attempt in range(settings.retry_pre_first_token + 1):
                try:
                    result = await self.proxy.post_json(
                        base_url=decision.candidate.base_url,
                        path=endpoint,
                        payload=egress_payload,
                    )
                    status_code = result.status_code
                    break
                except Exception as exc:
                    if attempt >= settings.retry_pre_first_token:
                        raise GatewayError(
                            status_code=502,
                            message=f"Upstream request failed: {exc}",
                            error_type="server_error",
                            code="upstream_request_failed",
                        ) from exc
            if result is None:
                raise GatewayError(
                    status_code=502,
                    message="Upstream returned no response.",
                    error_type="server_error",
                    code="upstream_request_failed",
                )
            usage = usage_from_payload(result.payload)
            usage_prompt = usage.prompt_tokens
            usage_completion = usage.completion_tokens
            usage_total = usage.total_tokens
            latency_ms = int((time.perf_counter() - started) * 1000)

            # Detokenize response when tokenize mode was used
            response_payload = result.payload
            if token_mapping and self.pii_client:
                try:
                    response_payload = await self.pii_client.detokenize(
                        payload=result.payload,
                        token_mapping=token_mapping,
                    )
                except Exception:
                    logger.warning(
                        "PII detokenize failed for %s; returning raw response",
                        request_id,
                    )

            self.observability.record_success(
                trace_context,
                status_code=result.status_code,
                latency_ms=latency_ms,
                ttft_ms=None,
                prompt_tokens=usage_prompt,
                completion_tokens=usage_completion,
                total_tokens=usage_total,
                provider_instance_id=str(decision.candidate.instance_id),
                output_payload=result.payload,
            )
            return GatewayResponse(
                status_code=result.status_code,
                payload=response_payload,
                provider_instance_id=str(decision.candidate.instance_id),
                latency_ms=latency_ms,
                trace_id=trace_context.trace_id,
            )
        except GatewayError as exc:
            error_code = exc.code
            status_code = exc.status_code
            self.observability.record_failure(
                trace_context,
                status_code=exc.status_code,
                latency_ms=int((time.perf_counter() - started) * 1000),
                provider_instance_id=str(decision.candidate.instance_id),
                error_code=exc.code,
                error_message=exc.message,
            )
            raise
        finally:
            await self.router.release(decision)
            await self.audit.log(
                request_id=request_id,
                trace_id=trace_context.trace_id,
                tenant_id=auth.tenant_id,
                user_id=auth.user_id,
                model_alias=model_alias,
                provider_instance_id=str(decision.candidate.instance_id),
                endpoint=endpoint,
                status_code=status_code,
                latency_ms=int((time.perf_counter() - started) * 1000),
                ttft_ms=None,
                prompt_tokens=usage_prompt,
                completion_tokens=usage_completion,
                total_tokens=usage_total,
                error_code=error_code,
            )

    async def route_stream_chat(
        self,
        *,
        auth: AuthContext,
        payload: dict[str, Any],
        request_id: str,
    ) -> StreamingGatewayResponse:
        started = time.perf_counter()
        endpoint = "/v1/chat/completions"
        model_alias = _require_model(payload)
        policy = await self.dao.get_tenant_policy(auth.tenant_id)
        await _check_limits(
            limiter=self.limiter,
            tenant_id=auth.tenant_id,
            payload=payload,
            rpm_limit=policy.rpm_limit if policy else None,
            tpm_limit=policy.tpm_limit if policy else None,
        )

        candidates = await self.router.resolve_alias(
            alias=model_alias, tenant_id=auth.tenant_id,
        )
        decision = await self.router.pick_and_lease(
            candidates=candidates, request_id=request_id,
        )

        # --- PII policy resolution (stream) ---
        pii_policy = parse_pii_policy(
            policy.pii_config if policy else None,
        )
        egress_payload = payload
        is_cloud = decision.candidate.provider_type == ProviderType.REMOTE_OPENAI

        if pii_policy and self.pii_client:
            # For streaming, only sanitize the INPUT payload (egress).
            # SSE chunk-level PII detection is a non-goal per spec.
            egress_payload, _ = await self._apply_egress_pii(
                payload=payload,
                pii_policy=pii_policy,
                is_cloud=is_cloud,
                request_id=request_id,
            )

        obs_payload = egress_payload
        if pii_policy and self.pii_client and pii_policy.telemetry.enabled:
            obs_payload = await self._apply_telemetry_pii(
                payload=payload,
                pii_policy=pii_policy,
                request_id=request_id,
            )

        trace_context = self.observability.start_request_trace(
            request_id=request_id,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            endpoint=endpoint,
            model_alias=model_alias,
            payload=obs_payload,
            privacy_mode=policy.privacy_mode if policy else None,
            stream=True,
        )
        raw_stream = self.proxy.stream_post(
            base_url=decision.candidate.base_url,
            path=endpoint,
            payload=egress_payload,
        )
        wrapped_stream, stats = await wrap_sse_stream(raw_stream)

        async def _stream_with_finalize() -> AsyncIterator[bytes]:
            status_code = 200
            error_code: str | None = None
            try:
                async for chunk in wrapped_stream:
                    yield chunk
            except Exception as exc:
                status_code = 502
                error_code = "upstream_stream_failed"
                del exc
                # Response has already started; terminate stream gracefully.
                yield b"data: [DONE]\n\n"
            finally:
                await self.router.release(decision)
                await self.audit.log(
                    request_id=request_id,
                    trace_id=trace_context.trace_id,
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    model_alias=model_alias,
                    provider_instance_id=str(decision.candidate.instance_id),
                    endpoint=endpoint,
                    status_code=status_code,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    ttft_ms=stats.ttft_ms,
                    prompt_tokens=stats.usage.prompt_tokens,
                    completion_tokens=stats.usage.completion_tokens,
                    total_tokens=stats.usage.total_tokens,
                    error_code=error_code,
                )
                self.observability.finalize_stream(
                    trace_context,
                    status_code=status_code,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    ttft_ms=stats.ttft_ms,
                    prompt_tokens=stats.usage.prompt_tokens,
                    completion_tokens=stats.usage.completion_tokens,
                    total_tokens=stats.usage.total_tokens,
                    provider_instance_id=str(decision.candidate.instance_id),
                    error_code=error_code,
                )

        return StreamingGatewayResponse(
            stream=_stream_with_finalize(),
            provider_instance_id=str(decision.candidate.instance_id),
            latency_ms=int((time.perf_counter() - started) * 1000),
            stats=stats,
            trace_id=trace_context.trace_id,
        )

    # ------------------------------------------------------------------
    # PII helpers
    # ------------------------------------------------------------------

    async def _apply_egress_pii(
        self,
        *,
        payload: dict[str, Any],
        pii_policy: PIIPolicy,
        is_cloud: bool,
        request_id: str,
    ) -> tuple[dict[str, Any], dict[str, str] | None]:
        """Sanitize *payload* before sending to upstream provider.

        Returns ``(sanitized_payload, token_mapping | None)``.
        If PII scanning is not applicable (local provider, policy disabled),
        the original *payload* is returned unchanged.
        """
        assert self.pii_client is not None  # noqa: S101

        should_scan = (
            (is_cloud and pii_policy.egress.enabled_for_cloud)
            or (not is_cloud and pii_policy.egress.enabled_for_local)
        )
        if not should_scan:
            return payload, None

        try:
            result = await self.pii_client.sanitize(
                payload=payload,
                policy=pii_policy,
                mode=pii_policy.egress.mode,
            )
        except Exception:
            # Honour fail_action
            if pii_policy.egress.fail_action == "block":
                raise GatewayError(
                    status_code=502,
                    message="PII service unavailable and fail_action=block.",
                    error_type="server_error",
                    code="pii_service_unavailable",
                )
            logger.warning(
                "PII egress scan failed for %s; fail_action=%s, allowing through",
                request_id,
                pii_policy.egress.fail_action,
            )
            return payload, None

        if result.pii_detected and pii_policy.egress.fail_action == "block":
            # In redact mode we already replaced PII; "block" means
            # we should reject the request when PII is found.
            if pii_policy.egress.mode == "redact":
                # Still send the redacted payload (PII is removed).
                pass

        return result.sanitized_payload, result.token_mapping

    async def _apply_telemetry_pii(
        self,
        *,
        payload: dict[str, Any],
        pii_policy: PIIPolicy,
        request_id: str,
    ) -> dict[str, Any]:
        """Produce a PII-clean version of *payload* for observability.

        If the telemetry mode is ``metrics_only`` we return a minimal
        stub so that Langfuse still gets token counts but no text.
        """
        assert self.pii_client is not None  # noqa: S101

        if pii_policy.telemetry.mode == "metrics_only":
            # Strip all text content; keep only model + metadata
            return {"model": payload.get("model"), "_pii_mode": "metrics_only"}

        try:
            result = await self.pii_client.sanitize(
                payload=payload,
                policy=pii_policy,
                mode="redact",  # always redact for telemetry
            )
            return result.sanitized_payload
        except Exception:
            logger.warning(
                "PII telemetry scan failed for %s; falling back to metadata-only",
                request_id,
            )
            return {"model": payload.get("model"), "_pii_mode": "fallback"}


def _require_model(payload: dict[str, Any]) -> str:
    model = str(payload.get("model", "")).strip()
    if not model:
        raise GatewayError(
            status_code=400,
            message="Request must include a non-empty model.",
            code="missing_model",
            param="model",
        )
    return model


async def _check_limits(
    *,
    limiter: RateLimiter,
    tenant_id: str,
    payload: dict[str, Any],
    rpm_limit: int | None,
    tpm_limit: int | None,
) -> None:
    rpm = await limiter.check_rpm(tenant_id=tenant_id, limit=rpm_limit)
    if rpm and not rpm.allowed:
        raise GatewayError(
            status_code=429,
            message="Rate limit exceeded (RPM).",
            code="rate_limit_rpm",
        )
    estimated_tokens = estimate_input_tokens(
        payload.get("input") or payload.get("messages"),
    )
    tpm = await limiter.check_tpm(
        tenant_id=tenant_id, tokens=estimated_tokens, limit=tpm_limit,
    )
    if tpm and not tpm.allowed:
        raise GatewayError(
            status_code=429,
            message="Rate limit exceeded (TPM).",
            code="rate_limit_tpm",
        )
