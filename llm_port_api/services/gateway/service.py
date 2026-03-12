from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from llm_port_api.db.dao.gateway_dao import GatewayDAO
from llm_port_api.db.dao.session_dao import SessionDAO
from llm_port_api.db.models.gateway import ProviderType
from llm_port_api.services.gateway.audit import AuditService
from llm_port_api.services.gateway.auth import AuthContext
from llm_port_api.services.gateway.errors import GatewayError
from llm_port_api.services.gateway.llm_adapter import LLMAdapter
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
from llm_port_api.services.gateway.file_store import FileStore
from llm_port_api.settings import settings

logger = logging.getLogger(__name__)


class _PIIFallbackToLocalRequested(Exception):
    """Signal that cloud egress should be rerouted to a local provider."""


# ── PII context system prompts ───────────────────────────────────────────────
_PII_REDACT_SYSTEM_PROMPT = (
    "IMPORTANT — Privacy notice: The user's message has been processed by an "
    "automated PII (Personally Identifiable Information) redaction system. "
    "Certain sensitive values have been replaced with placeholders such as "
    "[REDACTED_EMAIL_ADDRESS], [REDACTED_PHONE_NUMBER], [REDACTED_PERSON], "
    "[REDACTED_CREDIT_CARD], etc. These placeholders indicate where real data "
    "existed but was removed for privacy. "
    "When responding, preserve these placeholders exactly as they appear — do "
    "not attempt to guess the original values. If the user asks you to recall "
    "or fill in a redacted value, politely explain that the information was "
    "redacted for privacy."
)

_PII_TOKENIZE_SYSTEM_PROMPT = (
    "IMPORTANT — Privacy notice: The user's message has been processed by an "
    "automated PII (Personally Identifiable Information) tokenization system. "
    "Certain sensitive values have been replaced with surrogate tokens in the "
    "format [TOKEN_<hex>] (e.g. [TOKEN_a1b2c3]). Each token is a reversible "
    "placeholder for a real value that will be restored after your response. "
    "When responding, you MUST preserve these tokens exactly as they appear — "
    "do not modify, decode, or remove them. Place them in the same logical "
    "position in your answer so the de-tokenization step can restore the "
    "original values correctly."
)


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
        adapter: LLMAdapter,
        limiter: RateLimiter,
        audit: AuditService,
        observability: GatewayObservability,
        pii_client: PIIClient | None = None,
        rag_lite_client: RagLiteClient | None = None,
        session_dao: SessionDAO | None = None,
        file_store: FileStore | None = None,
    ) -> None:
        self.dao = dao
        self.router = router
        self.proxy = proxy
        self.adapter = adapter
        self.limiter = limiter
        self.audit = audit
        self.observability = observability
        self.pii_client = pii_client
        self.rag_lite_client = rag_lite_client
        self.session_dao = session_dao
        self._file_store = file_store

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
        session_id: str | None = None,
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

            # Session context injection (history + memory + summaries)
            resolved_session_id: str | None = None
            if endpoint == "/v1/chat/completions":
                payload, resolved_session_id = await self._inject_session_context(
                    payload, auth, session_id,
                )

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

            # Inject PII context system prompt when egress was sanitized
            if egress_payload is not payload and pii_policy:
                egress_payload = self._inject_pii_system_prompt(
                    egress_payload, pii_policy,
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
                    adapter_result = await self.adapter.completion(
                        provider_type=decision.candidate.provider_type,
                        base_url=decision.candidate.base_url,
                        api_key_encrypted=decision.candidate.api_key_encrypted,
                        litellm_provider=decision.candidate.litellm_provider,
                        litellm_model=decision.candidate.litellm_model,
                        extra_params=dict(decision.candidate.extra_params) if decision.candidate.extra_params else None,
                        payload=egress_payload,
                        stream=False,
                    )
                    from llm_port_api.services.gateway.llm_adapter import CompletionResult  # noqa: PLC0415
                    assert isinstance(adapter_result, CompletionResult)  # noqa: S101
                    result = UpstreamResult(
                        status_code=adapter_result.status_code,
                        payload=adapter_result.payload,
                        headers={},
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
            # Persist assistant response in session
            await self._persist_assistant_response(
                session_id_str=resolved_session_id,
                response_payload=response_payload,
                model_alias=model_alias,
                provider_instance_id=(
                    str(decision.candidate.instance_id) if decision is not None else None
                ),
                trace_id=trace_context.trace_id if trace_context is not None else None,
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
        session_id: str | None = None,
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

            # Session context injection (history + memory + summaries)
            resolved_stream_session_id: str | None = None
            payload, resolved_stream_session_id = await self._inject_session_context(
                payload, auth, session_id,
            )

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

            # Inject PII context system prompt when egress was sanitized
            if egress_payload is not payload and pii_policy:
                egress_payload = self._inject_pii_system_prompt(
                    egress_payload, pii_policy,
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
            raw_stream = self.adapter.completion(
                provider_type=decision.candidate.provider_type,
                base_url=decision.candidate.base_url,
                api_key_encrypted=decision.candidate.api_key_encrypted,
                litellm_provider=decision.candidate.litellm_provider,
                litellm_model=decision.candidate.litellm_model,
                extra_params=dict(decision.candidate.extra_params) if decision.candidate.extra_params else None,
                payload=egress_payload,
                stream=True,
            )
            # raw_stream is a coroutine returning AsyncIterator[bytes]
            raw_stream = await raw_stream  # type: ignore[misc]
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
        """Return whether the routed provider is a cloud/remote provider."""
        return decision.candidate.provider_type.value.startswith("remote_")

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
            if not candidate.provider_type.value.startswith("remote_")
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

    @staticmethod
    def _inject_pii_system_prompt(
        egress_payload: dict[str, Any],
        pii_policy: PIIPolicy,
    ) -> dict[str, Any]:
        """Prepend a system message explaining PII redaction/tokenization.

        Only modifies ``messages``-based payloads (chat completions).
        Returns a shallow-copied payload with the injected message.
        """
        messages = egress_payload.get("messages")
        if not isinstance(messages, list):
            return egress_payload

        prompt = (
            _PII_TOKENIZE_SYSTEM_PROMPT
            if pii_policy.egress.mode == "tokenize_reversible"
            else _PII_REDACT_SYSTEM_PROMPT
        )

        pii_system_msg: dict[str, str] = {"role": "system", "content": prompt}
        # Insert after any existing leading system messages so we don't
        # displace the user's own system prompt.
        insert_idx = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                insert_idx = i + 1
            else:
                break

        new_messages = list(messages)
        new_messages.insert(insert_idx, pii_system_msg)
        return {**egress_payload, "messages": new_messages}

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

    async def _inject_session_context(
        self,
        payload: dict[str, Any],
        auth: AuthContext,
        session_id_str: str | None,
    ) -> tuple[dict[str, Any], str | None]:
        """Inject session history and memory into the payload.

        Returns ``(updated_payload, resolved_session_id_hex)``
        where the session id is ``None`` when sessions are disabled.
        """
        if not session_id_str or not self.session_dao:
            return payload, None

        import uuid as _uuid  # noqa: PLC0415

        from llm_port_api.services.gateway.context_assembler import ContextAssembler  # noqa: PLC0415

        try:
            sid = _uuid.UUID(session_id_str)
        except ValueError:
            return payload, None

        sess = await self.session_dao.get_session(
            session_id=sid, tenant_id=auth.tenant_id, user_id=auth.user_id,
        )
        if not sess:
            return payload, None

        # Resolve project if the session belongs to one
        project = None
        if sess.project_id:
            project = await self.session_dao.get_project(
                project_id=sess.project_id,
                tenant_id=auth.tenant_id,
                user_id=auth.user_id,
            )

        # Resolve file store for attachment context injection
        file_store = getattr(self, "_file_store", None)

        assembler = ContextAssembler(
            dao=self.session_dao,
            max_recent_messages=settings.session_max_recent_messages,
            token_budget=settings.session_token_budget,
            file_store=file_store,
        )

        # Current request messages become the "tail" of the assembled context
        current_messages = payload.get("messages", [])
        assembled = await assembler.assemble(
            session_id=sid,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            current_messages=current_messages,
            project=project,
        )

        payload["messages"] = assembled.messages

        # Persist the user message(s)
        for msg in current_messages:
            if msg.get("role") in ("user", "system"):
                content = msg.get("content", "")
                # Handle multimodal content arrays
                content_parts_json = None
                if isinstance(content, list):
                    content_parts_json = content
                    content = " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ) or ""
                await self.session_dao.append_message(
                    session_id=sid,
                    role=msg["role"],
                    content=content,
                    content_parts_json=content_parts_json,
                )

        return payload, str(sid)

    async def _persist_assistant_response(
        self,
        *,
        session_id_str: str | None,
        response_payload: dict[str, Any],
        model_alias: str | None = None,
        provider_instance_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Store the assistant's response message in the session."""
        if not session_id_str or not self.session_dao:
            return

        import uuid as _uuid  # noqa: PLC0415

        try:
            sid = _uuid.UUID(session_id_str)
        except ValueError:
            return

        choices = response_payload.get("choices", [])
        if not choices:
            return

        msg_data = choices[0].get("message", {})
        content = msg_data.get("content", "")
        if not content:
            return

        # Handle multimodal assistant responses
        content_parts_json = None
        if isinstance(content, list):
            content_parts_json = content
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ) or ""

        usage = response_payload.get("usage", {})
        tokens = usage.get("completion_tokens")

        await self.session_dao.append_message(
            session_id=sid,
            role="assistant",
            content=content,
            content_parts_json=content_parts_json,
            model_alias=model_alias,
            provider_instance_id=(
                _uuid.UUID(provider_instance_id) if provider_instance_id else None
            ),
            token_estimate=tokens,
            trace_id=trace_id,
        )


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
