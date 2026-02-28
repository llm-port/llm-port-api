"""Modular service registry for LLM Port.

Provides a centralised way to declare, discover, and health-check
optional sidecar services (PII, Auth, RAG, etc.).

Design principles
-----------------
* **Opt-in only** -- a service is "enabled" when its URL env-var is set
  *and* the feature flag is ``True``.  If either condition is missing the
  service is treated as *disabled* and all call-sites skip it.
* **Zero-cost when off** -- the associated Docker containers never start
  (via Compose ``profiles:``), and no HTTP calls are attempted.
* **Uniform health surface** -- ``/api/v1/services`` returns the full
  manifest so the frontend can hide / show feature UI accordingly.
* **Easy to extend** -- adding a new module is: add a ``_ServiceDef``,
  register it in ``_KNOWN_SERVICES``, add the corresponding env-vars to
  ``Settings``, and tag the Docker service with a ``profiles:`` entry.

Usage from gateway / backend code::

    from llm_port_api.services.registry import service_registry

    if service_registry.is_enabled("pii"):
        client = service_registry.get_url("pii")
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ServiceStatus(StrEnum):
    """Runtime status of an optional service."""

    DISABLED = "disabled"
    CONFIGURED = "configured"  # URL is set but not yet health-checked
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass(slots=True)
class ServiceInfo:
    """Runtime descriptor for one optional service."""

    name: str
    display_name: str
    description: str
    enabled: bool
    url: str | None
    health_path: str
    status: ServiceStatus = ServiceStatus.DISABLED
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "enabled": self.enabled,
            "url": self.url if self.enabled else None,
            "status": self.status.value,
            "version": self.version,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class _ServiceDef:
    """Static definition of a known optional service."""

    name: str
    display_name: str
    description: str
    health_path: str = "/api/health"


# ------------------------------------------------------------------
# Known optional services.  To add a new module, just append here.
# ------------------------------------------------------------------
_KNOWN_SERVICES: list[_ServiceDef] = [
    _ServiceDef(
        name="pii",
        display_name="PII Protection",
        description=(
            "Presidio-based PII detection, redaction, and tokenization. "
            "Provides egress sanitization for cloud providers and "
            "telemetry sanitization for observability."
        ),
        health_path="/api/health",
    ),
    _ServiceDef(
        name="auth",
        display_name="External Authentication",
        description=(
            "Pluggable external authentication service (OIDC / SAML / LDAP). "
            "When disabled, the built-in JWT authentication is used."
        ),
        health_path="/api/health",
    ),
    _ServiceDef(
        name="rag",
        display_name="RAG Engine",
        description=(
            "Retrieval-Augmented Generation pipeline with document ingestion, "
            "chunking, embedding, and vector search."
        ),
        health_path="/api/health",
    ),
]


class ServiceRegistry:
    """Central registry of optional sidecar services.

    Instantiated once at application startup and stored on
    ``app.state.service_registry``.
    """

    def __init__(self) -> None:
        self._services: dict[str, ServiceInfo] = {}
        for svc in _KNOWN_SERVICES:
            self._services[svc.name] = ServiceInfo(
                name=svc.name,
                display_name=svc.display_name,
                description=svc.description,
                enabled=False,
                url=None,
                health_path=svc.health_path,
                status=ServiceStatus.DISABLED,
            )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(
        self,
        name: str,
        *,
        enabled: bool,
        url: str | None = None,
    ) -> None:
        """Mark a service as enabled/disabled and set its base URL.

        Called during app startup from ``lifespan_setup``.
        """
        info = self._services.get(name)
        if info is None:
            logger.warning("Unknown service '%s' in registry.configure()", name)
            return
        info.enabled = enabled and bool(url)
        info.url = url.rstrip("/") if url else None
        info.status = (
            ServiceStatus.CONFIGURED if info.enabled else ServiceStatus.DISABLED
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_enabled(self, name: str) -> bool:
        info = self._services.get(name)
        return bool(info and info.enabled)

    def get_url(self, name: str) -> str | None:
        info = self._services.get(name)
        return info.url if info and info.enabled else None

    def get_info(self, name: str) -> ServiceInfo | None:
        return self._services.get(name)

    def all_services(self) -> list[ServiceInfo]:
        return list(self._services.values())

    def enabled_services(self) -> list[ServiceInfo]:
        return [s for s in self._services.values() if s.enabled]

    # ------------------------------------------------------------------
    # Health probing
    # ------------------------------------------------------------------

    async def check_health(
        self,
        http_client: httpx.AsyncClient,
        *,
        services: list[str] | None = None,
    ) -> dict[str, ServiceStatus]:
        """Probe enabled services and update their status.

        Parameters
        ----------
        http_client:
            Shared ``httpx.AsyncClient`` (same one used by the gateway).
        services:
            If given, only check these service names.  Otherwise check
            all enabled services.

        Returns
        -------
        dict mapping service name -> new status.
        """
        targets = (
            [self._services[n] for n in services if n in self._services]
            if services
            else self.enabled_services()
        )
        results: dict[str, ServiceStatus] = {}
        for svc in targets:
            if not svc.enabled or not svc.url:
                results[svc.name] = ServiceStatus.DISABLED
                continue
            try:
                resp = await http_client.get(
                    f"{svc.url}{svc.health_path}",
                    timeout=5.0,
                )
                if resp.status_code < 400:
                    svc.status = ServiceStatus.HEALTHY
                else:
                    svc.status = ServiceStatus.UNHEALTHY
            except Exception:
                svc.status = ServiceStatus.UNHEALTHY
                logger.debug("Health check failed for %s", svc.name, exc_info=True)
            results[svc.name] = svc.status
        return results

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Full manifest suitable for the ``/api/v1/services`` response."""
        return {
            "services": [s.to_dict() for s in self._services.values()],
        }


# Module-level singleton (configured during lifespan)
service_registry = ServiceRegistry()
