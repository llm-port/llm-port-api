"""Load system settings from the backend DB at startup.

Reads non-secret ``system_setting_value`` rows from the backend database
and patches the API gateway ``settings`` singleton so the gateway picks
up admin-configured values without requiring env-var changes.

This follows the same pattern as ``jwt_secret.py`` which already reads
encrypted secrets from ``system_setting_secret``.

Keys loaded
-----------
- ``llm_port_api.pii_enabled`` → ``settings.pii_enabled``
- ``llm_port_api.pii_service_url`` → ``settings.pii_service_url``
- ``llm_port_api.pii_default_policy`` → ``settings.pii_default_policy``
- ``llm_port_api.mcp_enabled`` → ``settings.mcp_enabled``
- ``llm_port_api.mcp_service_url`` → ``settings.mcp_service_url``
- ``llm_port_api.skills_enabled`` → ``settings.skills_enabled``
- ``llm_port_api.skills_service_url`` → ``settings.skills_service_url``
"""

from __future__ import annotations

from typing import Any

import asyncpg
from loguru import logger

from llm_port_api.settings import settings

# Keys to load and their mapping to settings attributes.
_VALUE_KEYS: dict[str, str] = {
    "llm_port_api.pii_enabled": "pii_enabled",
    "llm_port_api.pii_service_url": "pii_service_url",
    "llm_port_api.pii_default_policy": "pii_default_policy",
    "llm_port_api.mcp_enabled": "mcp_enabled",
    "llm_port_api.mcp_service_url": "mcp_service_url",
    "llm_port_api.skills_enabled": "skills_enabled",
    "llm_port_api.skills_service_url": "skills_service_url",
}


async def load_system_settings_from_backend_db() -> int:
    """Load system setting values from the backend DB.

    Returns the number of settings successfully applied.
    """
    if not settings.backend_db_base:
        return 0

    dsn = (
        f"postgresql://{settings.db_user}:{settings.db_pass}"
        f"@{settings.db_host}:{settings.db_port}/{settings.backend_db_base}"
    )
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Settings loader: could not connect to backend DB — {}", exc)
        return 0

    try:
        rows = await conn.fetch(
            "SELECT key, value_json FROM system_setting_value WHERE key = ANY($1::text[])",
            list(_VALUE_KEYS.keys()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Settings loader: query failed — {}", exc)
        return 0
    finally:
        await conn.close()

    applied = 0
    for row in rows:
        db_key: str = row["key"]
        raw_json: Any = row["value_json"]

        # value_json is stored as {"value": <actual>} by SystemSettingsDAO
        if isinstance(raw_json, dict):
            value = raw_json.get("value", raw_json)
        elif isinstance(raw_json, str):
            # asyncpg may return JSONB as already-parsed dict; handle str fallback
            import json as _json

            try:
                parsed = _json.loads(raw_json)
                value = parsed.get("value", parsed) if isinstance(parsed, dict) else parsed
            except _json.JSONDecodeError:
                value = raw_json
        else:
            value = raw_json

        attr = _VALUE_KEYS.get(db_key)
        if attr is None:
            continue
        # Skip empty/blank strings so the code-level default is preserved.
        if isinstance(value, str) and not value.strip():
            continue

        try:
            setattr(settings, attr, value)
            applied += 1
            logger.info("Settings loader: applied {} = {}", db_key, _safe_preview(value))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Settings loader: failed to apply {} — {}", db_key, exc)

    if applied:
        logger.info("Settings loader: loaded {} setting(s) from backend DB.", applied)
    return applied


def _safe_preview(value: object) -> str:
    """Return a short preview of a value for logging."""
    s = str(value)
    if len(s) > 80:
        return s[:77] + "..."
    return s
