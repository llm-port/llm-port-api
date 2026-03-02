from __future__ import annotations

import base64
import hashlib

import asyncpg
from cryptography.fernet import Fernet
from loguru import logger

from llm_port_api.settings import settings

_JWT_SECRET_KEYS: tuple[str, ...] = (
    "llm_port_api.jwt_secret",
    "llm_port_backend.users_secret",
)


async def load_jwt_secret_from_backend_db() -> bool:
    """Load JWT secret from backend system settings DB.

    Preferred key is ``llm_port_api.jwt_secret``; if absent/empty, fallback to
    ``llm_port_backend.users_secret`` to keep signing and verification aligned.
    """
    if not settings.backend_db_base or not settings.settings_master_key:
        return False

    dsn = (
        f"postgresql://{settings.db_user}:{settings.db_pass}"
        f"@{settings.db_host}:{settings.db_port}/{settings.backend_db_base}"
    )
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JWT secret: could not connect to backend DB - {}", exc)
        return False

    try:
        rows = await conn.fetch(
            "SELECT key, ciphertext FROM system_setting_secret WHERE key = ANY($1::text[])",
            list(_JWT_SECRET_KEYS),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("JWT secret: query failed - {}", exc)
        return False
    finally:
        await conn.close()

    ciphertext_by_key = {str(row["key"]): str(row["ciphertext"]) for row in rows}
    if not ciphertext_by_key:
        logger.info("JWT secret: no JWT secret keys found in backend DB, using env var fallback.")
        return False

    digest = hashlib.sha256(settings.settings_master_key.encode()).digest()
    fernet = Fernet(base64.urlsafe_b64encode(digest))

    for db_key in _JWT_SECRET_KEYS:
        ciphertext = ciphertext_by_key.get(db_key, "")
        if not ciphertext:
            continue
        try:
            secret = fernet.decrypt(ciphertext.encode()).decode().strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("JWT secret: decryption failed for {} - {}", db_key, exc)
            continue
        if not secret:
            logger.warning("JWT secret: key '{}' decrypted to empty value.", db_key)
            continue

        settings.jwt_secret = secret
        if db_key == "llm_port_api.jwt_secret":
            logger.info("JWT secret loaded from backend DB (llm_port_api.jwt_secret).")
        else:
            logger.warning(
                "JWT secret loaded from backend fallback key '{}'. "
                "Set llm_port_api.jwt_secret explicitly to avoid drift.",
                db_key,
            )
        return True

    logger.info("JWT secret: backend DB keys exist but no usable secret was found.")
    return False
