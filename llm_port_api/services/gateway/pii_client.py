"""HTTP client for the PII micro-service.

Used by the gateway pipeline to sanitize payloads before cloud egress
and before observability/audit recording.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from llm_port_api.services.gateway.pii_policy import PIIPolicy

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SanitizeResult:
    """Result of a /sanitize call to the PII service."""

    sanitized_payload: dict[str, Any]
    entities_found: list[dict[str, Any]] = field(default_factory=list)
    pii_detected: bool = False
    token_mapping: dict[str, str] | None = None


class PIIClient:
    """Async client wrapping PII service HTTP endpoints."""

    def __init__(self, *, base_url: str, http_client: httpx.AsyncClient) -> None:
        self._base = base_url.rstrip("/")
        self._http = http_client

    async def sanitize(
        self,
        *,
        payload: dict[str, Any],
        policy: PIIPolicy,
        mode: str | None = None,
    ) -> SanitizeResult:
        """Call ``POST /api/v1/pii/sanitize`` on the PII service.

        Parameters
        ----------
        payload:
            Full OpenAI-compatible request body (chat or embeddings).
        policy:
            Parsed ``PIIPolicy`` that determines mode, language, entities,
            and score threshold.
        mode:
            Override mode (``"redact"`` or ``"tokenize"``).
            If *None*, falls back to ``policy.egress.mode``.
        """
        effective_mode = mode or policy.egress.mode
        # Map policy mode names to PII service mode names
        if effective_mode == "tokenize_reversible":
            effective_mode = "tokenize"

        body: dict[str, Any] = {
            "payload": payload,
            "mode": effective_mode,
            "language": policy.presidio.language,
            "score_threshold": policy.presidio.threshold,
        }
        if policy.presidio.entities:
            body["entities"] = policy.presidio.entities

        try:
            resp = await self._http.post(
                f"{self._base}/api/v1/pii/sanitize",
                json=body,
                timeout=10.0,
                headers={"x-pii-source": "gateway"},
            )
            resp.raise_for_status()
            data = resp.json()
            return SanitizeResult(
                sanitized_payload=data.get("sanitized_payload", payload),
                entities_found=data.get("entities_found", []),
                pii_detected=bool(data.get("entities_found")),
                token_mapping=data.get("token_mapping"),
            )
        except Exception:
            logger.exception("PII sanitize call failed")
            raise

    async def detokenize(
        self,
        *,
        payload: dict[str, Any],
        token_mapping: dict[str, str],
    ) -> dict[str, Any]:
        """Call ``POST /api/v1/pii/detokenize`` to restore original text."""
        try:
            resp = await self._http.post(
                f"{self._base}/api/v1/pii/detokenize",
                json={"payload": payload, "token_mapping": token_mapping},
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json().get("payload", payload)
        except Exception:
            logger.exception("PII detokenize call failed")
            raise
