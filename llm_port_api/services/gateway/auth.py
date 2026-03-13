from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llm_port_api.services.gateway.errors import GatewayError
from llm_port_api.services.gateway.jwt_secret import load_jwt_secret_from_backend_db
from llm_port_api.settings import settings


@dataclass(slots=True, frozen=True)
class AuthContext:
    """Authenticated request context extracted from JWT."""

    user_id: str
    tenant_id: str
    raw_claims: dict[str, Any]


# auto_error=False so we can return our own structured error response
# instead of FastAPI's default 403.
_bearer_scheme = HTTPBearer(auto_error=False)
_JWT_SECRET_REFRESH_LOCK = asyncio.Lock()
_JWT_SECRET_REFRESH_COOLDOWN_SEC = 15.0
_last_jwt_secret_refresh_attempt_monotonic = 0.0


async def _refresh_jwt_secret_if_needed() -> None:
    """Best-effort lazy secret refresh to recover from startup races."""
    global _last_jwt_secret_refresh_attempt_monotonic
    if settings.jwt_secret:
        return

    now = time.monotonic()
    if now - _last_jwt_secret_refresh_attempt_monotonic < _JWT_SECRET_REFRESH_COOLDOWN_SEC:
        return

    async with _JWT_SECRET_REFRESH_LOCK:
        if settings.jwt_secret:
            return
        now = time.monotonic()
        if now - _last_jwt_secret_refresh_attempt_monotonic < _JWT_SECRET_REFRESH_COOLDOWN_SEC:
            return
        _last_jwt_secret_refresh_attempt_monotonic = now
        await load_jwt_secret_from_backend_db()


def verify_token(token: str) -> dict[str, Any]:
    """Verify JWT token and return claims."""
    if not settings.jwt_secret:
        raise GatewayError(
            status_code=500,
            message="JWT secret is not configured.",
            error_type="server_error",
            code="jwt_not_configured",
        )
    try:
        decode_opts: dict[str, Any] = {
            "algorithms": [settings.jwt_algorithm],
        }
        if settings.jwt_audience:
            decode_opts["audience"] = settings.jwt_audience
        else:
            # When no audience is configured, skip audience verification so
            # tokens with an "aud" claim (e.g. fastapi-users) are accepted.
            decode_opts["options"] = {"verify_aud": False}
        if settings.jwt_issuer:
            decode_opts["issuer"] = settings.jwt_issuer
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            **decode_opts,
        )
    except jwt.PyJWTError as exc:
        raise GatewayError(
            status_code=401,
            message="Invalid JWT token.",
            code="invalid_token",
        ) from exc
    return claims


def get_auth_context_from_claims(claims: dict[str, Any]) -> AuthContext:
    """Extract user and tenant identifiers from verified claims."""
    user_id = str(claims.get("sub", "")).strip()
    if not user_id:
        raise GatewayError(
            status_code=401,
            message="JWT token does not contain subject (sub).",
            code="invalid_token_sub",
        )
    tenant_id = str(claims.get("tenant_id", "")).strip()
    if not tenant_id:
        # Default to "default" tenant for single-tenant deployments and
        # fastapi-users tokens that don't carry a tenant_id claim.
        tenant_id = "default"
    return AuthContext(user_id=user_id, tenant_id=tenant_id, raw_claims=claims)


async def get_auth_context(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthContext:
    """FastAPI dependency that validates JWT bearer token.

    Using HTTPBearer ensures Swagger UI shows the Authorize button and
    correctly includes the token in the Authorization: Bearer <token> header.
    """
    token = credentials.credentials if credentials else None
    if not token:
        raise GatewayError(
            status_code=401,
            message="Missing Authorization header.",
            code="missing_authorization",
        )
    await _refresh_jwt_secret_if_needed()
    claims = verify_token(token)
    return get_auth_context_from_claims(claims)
