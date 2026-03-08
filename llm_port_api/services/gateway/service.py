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
from llm_port_api.services.gateway.rag_lite_client import RagLiteClient
from llm_port_api.services.gateway.ratelimit import RateLimiter
from llm_port_api.services.gateway.routing import RouterService, RoutingDecision
from llm_port_api.services.gateway.stream import StreamStats, wrap_sse_stream
from llm_port_api.services.gateway.usage import (
    estimate_input_tokens,
    usage_from_payload,
)
from llm_port_api.settings import settings

logger = logging.getLogger(__name__)


class _PIIFallbackToLocalRequested(Exception):
    """Signal that cloud egress should be rerouted to a local provider."""


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
        rag_lite_client: RagLiteClient | None = None,
    ) -> None:
        self.dao = dao
        self.router = router
        self.proxy = proxy
        self.limiter = limiter
        self.audit = audit
        self.observability = observability
        self.pii_client = pii_client
        self.rag_lite_client = rag_lite_client

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
        decision: RoutingDecision | None = await self.router.pick_and_lease(
            candidates=candidates, request_id=request_id,
        )

        result: UpstreamResult | None = None
        fallback_outcome = "not_used"
        error_code: str | None = None
        status_code = 500
        usage_prompt = None
        usage_completion = None
        usage_total = None
        trace_context = None
        try:
            # RAG Lite context injection (before PII so context is scanned too)
            payload = await self._inject_rag_context(payload, auth)

            pii_policy = _resolve_pii_policy(policy)
            egress_payload = payload
            token_mapping: dict[str, str] | None = None

            if pii_policy and self.pii_client and decision is not None:
                try:
                    egress_payload, token_mapping = await self._apply_egress_pii(
                        payload=payload,
                        pii_policy=pii_policy,
                        is_cloud=self._is_cloud_provider(decision),
                        request_id=request_id,
                    )
                except _PIIFallbackToLocalRequested:
                    fallback_outcome = "fallback_to_local_attempted"
                    released_decision = decision
                    decision = None
                    try:
                        decision = await self._fallback_to_local_candidate(
                            current_decision=released_decision,
                            candidates=candidates,
                            request_id=request_id,
                        )
                    except GatewayError:
                        fallback_outcome = "fallback_to_local_failed"
                        raise
                    fallback_outcome = "fallback_to_local_succeeded"
                    egress_payload, token_mapping = await self._apply_egress_pii(
                        payload=payload,
                        pii_policy=pii_policy,
                        is_cloud=False,
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
                stream=False,
                routing_metadata={"pii_fallback_outcome": fallback_outcome},
            )

            for attempt in range(settings.retry_pre_first_token + 1):
                try:
                    result = await self.proxy.post_json(
                        base_url=decision.candidate.base_url,  # type: ignore[union-attr]
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

            if trace_context is not None:
                self.observability.record_success(
                    trace_context,
                    status_code=result.status_code,
                    latency_ms=latency_ms,
                    ttft_ms=None,
                    prompt_tokens=usage_prompt,
                    completion_tokens=usage_completion,
                    total_tokens=usage_total,
                    provider_instance_id=(
                        str(decision.candidate.instance_id) if decision is not None else None
                    ),
                    output_payload=result.payload,
                )
            return GatewayResponse(
                status_code=result.status_code,
                payload=response_payload,
                provider_instance_id=str(decision.candidate.instance_id),  # type: ignore[union-attr]
                latency_ms=latency_ms,
                trace_id=trace_context.trace_id if trace_context is not None else None,
            )
        except GatewayError as exc:
            error_code = exc.code
            status_code = exc.status_code
            if trace_context is None:
                trace_context = self.observability.start_request_trace(
                    request_id=request_id,
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    endpoint=endpoint,
                    model_alias=model_alias,
                    payload={"model": payload.get("model"), "_pii_mode": "pre_upstream_error"},
                    privacy_mode=policy.privacy_mode if policy else None,
                    stream=False,
                    routing_metadata={"pii_fallback_outcome": fallback_outcome},
                )
            self.observability.record_failure(
                trace_context,
                status_code=exc.status_code,
                latency_ms=int((time.perf_counter() - started) * 1000),
                provider_instance_id=(
                    str(decision.candidate.instance_id) if decision is not None else None
                ),
                error_code=exc.code,
                error_message=exc.message,
            )
            raise
        finally:
            if decision is not None:
                await self.router.release(decision)
            await self.audit.log(
                request_id=request_id,
                trace_id=trace_context.trace_id if trace_context is not None else None,
                tenant_id=auth.tenant_id,
                user_id=auth.user_id,
                model_alias=model_alias,
                provider_instance_id=(
                    str(decision.candidate.instance_id) if decision is not None else None
                ),
                endpoint=endpoint,
                status_code=status_code,
                latency_ms=int((time.perf_counter() - started) * 1000),
                ttft_ms=None,
                prompt_tokens=usage_prompt,
                completion_tokens=usage_completion,
                total_tokens=usage_total,
                error_code=error_code or (
                    "pii_fallback_to_local_succeeded"
                    if fallback_outcome == "fallback_to_local_succeeded"
                    else None
                ),
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
        decision: RoutingDecision | None = await self.router.pick_and_lease(
            candidates=candidates, request_id=request_id,
        )

        fallback_outcome = "not_used"
        trace_context = None
        stream_started = False
        stats: StreamStats | None = None
        pre_stream_status_code = 500
        pre_stream_error_code: str | None = None
        try:
            # RAG Lite context injection (before PII so context is scanned too)
            payload = await self._inject_rag_context(payload, auth)

            pii_policy = _resolve_pii_policy(policy)
            egress_payload = payload

            if pii_policy and self.pii_client and decision is not None:
                try:
                    # For streaming, sanitize only request egress payload.
                    egress_payload, _ = await self._apply_egress_pii(
                        payload=payload,
                        pii_policy=pii_policy,
                        is_cloud=self._is_cloud_provider(decision),
                        request_id=request_id,
                    )
                except _PIIFallbackToLocalRequested:
                    fallback_outcome = "fallback_to_local_attempted"
                    released_decision = decision
                    decision = None
                    try:
                        decision = await self._fallback_to_local_candidate(
                            current_decision=released_decision,
                            candidates=candidates,
                            request_id=request_id,
                        )
                    except GatewayError:
                        fallback_outcome = "fallback_to_local_failed"
                        raise
                    fallback_outcome = "fallback_to_local_succeeded"
                    egress_payload, _ = await self._apply_egress_pii(
                        payload=payload,
                        pii_policy=pii_policy,
                        is_cloud=False,
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
                routing_metadata={"pii_fallback_outcome": fallback_outcome},
            )
            raw_stream = self.proxy.stream_post(
                base_url=decision.candidate.base_url,  # type: ignore[union-attr]
                path=endpoint,
                payload=egress_payload,
            )
            wrapped_stream, stats = await wrap_sse_stream(raw_stream)
            stream_started = True

            async def _stream_with_finalize() -> AsyncIterator[bytes]:
                stream_status_code = 200
                stream_error_code: str | None = None
                try:
                    async for chunk in wrapped_stream:
                        yield chunk
                except Exception as exc:
                    stream_status_code = 502
                    stream_error_code = "upstream_stream_failed"
                    del exc
                    # Response has already started; terminate stream gracefully.
                    yield b"data: [DONE]\n\n"
                finally:
                    final_error_code = stream_error_code or (
                        "pii_fallback_to_local_succeeded"
                        if fallback_outcome == "fallback_to_local_succeeded"
                        else None
                    )
                    if decision is not None:
                        await self.router.release(decision)
                    await self.audit.log(
                        request_id=request_id,
                        trace_id=trace_context.trace_id if trace_context is not None else None,
                        tenant_id=auth.tenant_id,
                        user_id=auth.user_id,
                        model_alias=model_alias,
                        provider_instance_id=(
                            str(decision.candidate.instance_id) if decision is not None else None
                        ),
                        endpoint=endpoint,
                        status_code=stream_status_code,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        ttft_ms=stats.ttft_ms if stats is not None else None,
                        prompt_tokens=stats.usage.prompt_tokens if stats is not None else None,
                        completion_tokens=stats.usage.completion_tokens if stats is not None else None,
                        total_tokens=stats.usage.total_tokens if stats is not None else None,
                        error_code=final_error_code,
                    )
                    if trace_context is not None:
                        self.observability.finalize_stream(
                            trace_context,
                            status_code=stream_status_code,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            ttft_ms=stats.ttft_ms if stats is not None else None,
                            prompt_tokens=stats.usage.prompt_tokens if stats is not None else None,
                            completion_tokens=stats.usage.completion_tokens if stats is not None else None,
                            total_tokens=stats.usage.total_tokens if stats is not None else None,
                            provider_instance_id=(
                                str(decision.candidate.instance_id) if decision is not None else None
                            ),
                            error_code=final_error_code,
                        )

            return StreamingGatewayResponse(
                stream=_stream_with_finalize(),
                provider_instance_id=str(decision.candidate.instance_id),  # type: ignore[union-attr]
                latency_ms=int((time.perf_counter() - started) * 1000),
                stats=stats,
                trace_id=trace_context.trace_id if trace_context is not None else None,
            )
        except GatewayError as exc:
            pre_stream_status_code = exc.status_code
            pre_stream_error_code = exc.code
            if trace_context is None:
                trace_context = self.observability.start_request_trace(
                    request_id=request_id,
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    endpoint=endpoint,
                    model_alias=model_alias,
                    payload={"model": payload.get("model"), "_pii_mode": "pre_upstream_error"},
                    privacy_mode=policy.privacy_mode if policy else None,
                    stream=True,
                    routing_metadata={"pii_fallback_outcome": fallback_outcome},
                )
            self.observability.record_failure(
                trace_context,
                status_code=exc.status_code,
                latency_ms=int((time.perf_counter() - started) * 1000),
                provider_instance_id=(
                    str(decision.candidate.instance_id) if decision is not None else None
                ),
                error_code=exc.code,
                error_message=exc.message,
            )
            raise
        finally:
            if not stream_started:
                if decision is not None:
                    await self.router.release(decision)
                await self.audit.log(
                    request_id=request_id,
                    trace_id=trace_context.trace_id if trace_context is not None else None,
                    tenant_id=auth.tenant_id,
                    user_id=auth.user_id,
                    model_alias=model_alias,
                    provider_instance_id=(
                        str(decision.candidate.instance_id) if decision is not None else None
                    ),
                    endpoint=endpoint,
                    status_code=pre_stream_status_code,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    ttft_ms=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    error_code=pre_stream_error_code or (
                        "pii_fallback_to_local_succeeded"
                        if fallback_outcome == "fallback_to_local_succeeded"
                        else None
                    ),
                )

    # ------------------------------------------------------------------
    # PII helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_cloud_provider(decision: RoutingDecision) -> bool:
        """Return whether the routed provider is a cloud provider."""
        return decision.candidate.provider_type == ProviderType.REMOTE_OPENAI

    @staticmethod
    def _is_local_candidate(decision: RoutingDecision) -> bool:
        """Return whether the routed provider is local/on-prem."""
        return not GatewayService._is_cloud_provider(decision)

    async def _fallback_to_local_candidate(
        self,
        *,
        current_decision: RoutingDecision,
        candidates: list[Any],
        request_id: str,
    ) -> RoutingDecision:
        """Release cloud lease and pick a local candidate for fallback."""
        await self.router.release(current_decision)

        local_candidates = [
            candidate
            for candidate in candidates
            if candidate.provider_type != ProviderType.REMOTE_OPENAI
        ]
        if not local_candidates:
            raise GatewayError(
                status_code=503,
                message="PII fallback requested but no local provider candidate is available.",
                error_type="server_error",
                code="pii_fallback_no_local_provider",
            )
        try:
            return await self.router.pick_and_lease(
                candidates=local_candidates,
                request_id=request_id,
            )
        except GatewayError as exc:
            if exc.code == "no_capacity":
                raise GatewayError(
                    status_code=503,
                    message="PII fallback requested but no local provider has free capacity.",
                    error_type="server_error",
                    code="pii_fallback_no_local_capacity",
                ) from exc
            raise

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
            if pii_policy.egress.fail_action == "fallback_to_local" and is_cloud:
                raise _PIIFallbackToLocalRequested()
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

    async def _inject_rag_context(
        self,
        payload: dict[str, Any],
        auth: AuthContext,
    ) -> dict[str, Any]:
        """Optionally inject RAG Lite context into the messages.

        Looks for a ``rag`` dict in the payload (added by the frontend).
        If present, queries the backend's RAG Lite search endpoint and
        prepends a system message with the retrieved context.

        The ``rag`` key is always stripped from the payload so the upstream
        provider doesn't receive unknown fields.
        """
        rag_config = payload.pop("rag", None)
        if not rag_config or not self.rag_lite_client:
            return payload

        # Extract search query from the last user message
        messages = payload.get("messages", [])
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_query = content if isinstance(content, str) else str(content)
                break

        if not user_query:
            return payload

        results = await self.rag_lite_client.search(
            query=user_query,
            top_k=rag_config.get("top_k", 5),
            collection_ids=rag_config.get("collection_ids"),
        )

        if not results:
            return payload

        # Build context block from search results
        context_parts = []
        for r in results:
            src = r.get("filename", "unknown")
            text = r.get("chunk_text", "")
            context_parts.append(f"[Source: {src}]\n{text}")

        context_block = (
            "Use the following retrieved context to answer the user's question. "
            "If the context is not relevant, ignore it.\n\n"
            + "\n\n---\n\n".join(context_parts)
        )

        # Prepend as a system message
        payload["messages"] = [
            {"role": "system", "content": context_block},
            *messages,
        ]
        return payload


def _resolve_pii_policy(
    policy: Any | None,
) -> PIIPolicy | None:
    """Resolve effective PII policy: tenant-specific → system default → None."""
    from llm_port_api.settings import settings as _settings

    raw = policy.pii_config if policy and getattr(policy, "pii_config", None) else None
    if raw:
        return parse_pii_policy(raw)
    # Fallback to the system-wide default policy loaded from system settings DB.
    default = getattr(_settings, "pii_default_policy", None)
    if default:
        return parse_pii_policy(default)
    return None


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
