from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llm_port_api.services.gateway.errors import GatewayError
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
        raise GatewayError(
            status_code=403,
            message="JWT token does not include tenant_id claim.",
            code="missing_tenant_id",
        )
    return AuthContext(user_id=user_id, tenant_id=tenant_id, raw_claims=claims)


def get_auth_context(
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
    claims = verify_token(token)
    return get_auth_context_from_claims(claims)
