"""PII policy helpers for the gateway pipeline.

Parses the ``pii_config`` JSON from ``TenantLLMPolicy`` into typed
dataclasses and provides convenience predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PIITelemetryPolicy:
    """Controls what goes into Langfuse / audit DB."""

    enabled: bool = False
    mode: str = "sanitized"  # sanitized | metrics_only
    store_raw: bool = False


@dataclass(frozen=True, slots=True)
class PIIEgressPolicy:
    """Controls sanitization before cloud provider egress."""

    enabled_for_cloud: bool = False
    enabled_for_local: bool = False
    mode: str = "redact"  # redact | tokenize_reversible
    fail_action: str = "block"  # block | allow | fallback_to_local


@dataclass(frozen=True, slots=True)
class PIIPresidioConfig:
    """Presidio engine overrides from the policy."""

    language: str = "en"
    threshold: float = 0.6
    entities: list[str] = field(default_factory=lambda: [
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "PERSON",
        "LOCATION",
    ])


@dataclass(frozen=True, slots=True)
class PIIPolicy:
    """Full parsed PII policy for a tenant/workspace."""

    telemetry: PIITelemetryPolicy = field(default_factory=PIITelemetryPolicy)
    egress: PIIEgressPolicy = field(default_factory=PIIEgressPolicy)
    presidio: PIIPresidioConfig = field(default_factory=PIIPresidioConfig)

    @property
    def any_enabled(self) -> bool:
        """True if any PII protection is active."""
        return self.telemetry.enabled or self.egress.enabled_for_cloud or self.egress.enabled_for_local

    @property
    def egress_enabled_for(self) -> tuple[bool, bool]:
        """Return (cloud, local) enablement flags."""
        return self.egress.enabled_for_cloud, self.egress.enabled_for_local


def parse_pii_policy(raw: dict[str, Any] | None) -> PIIPolicy | None:
    """Parse ``pii_config`` JSON into a typed ``PIIPolicy``.

    Returns ``None`` when PII is not configured (column is NULL or empty).
    """
    if not raw:
        return None

    tel_raw = raw.get("telemetry", {})
    egr_raw = raw.get("egress", {})
    pre_raw = raw.get("presidio", {})

    telemetry = PIITelemetryPolicy(
        enabled=bool(tel_raw.get("enabled", False)),
        mode=str(tel_raw.get("mode", "sanitized")),
        store_raw=bool(tel_raw.get("store_raw", False)),
    )
    egress = PIIEgressPolicy(
        enabled_for_cloud=bool(egr_raw.get("enabled_for_cloud", False)),
        enabled_for_local=bool(egr_raw.get("enabled_for_local", False)),
        mode=str(egr_raw.get("mode", "redact")),
        fail_action=str(egr_raw.get("fail_action", "block")),
    )
    presidio = PIIPresidioConfig(
        language=str(pre_raw.get("language", "en")),
        threshold=float(pre_raw.get("threshold", 0.6)),
        entities=list(pre_raw.get("entities", PIIPresidioConfig.entities)),
    )

    policy = PIIPolicy(telemetry=telemetry, egress=egress, presidio=presidio)

    if not policy.any_enabled:
        return None

    return policy
