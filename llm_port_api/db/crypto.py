"""Column-level encryption for sensitive chat data.

Provides two SQLAlchemy ``TypeDecorator`` types:

* **EncryptedText** – encrypts ``str`` values before INSERT/UPDATE and
  decrypts after SELECT.  Stored in the database as ``Text`` containing
  Fernet ciphertext (URL-safe base64).
* **EncryptedJSON** – serializes a Python ``dict`` / ``list`` to a JSON
  string, encrypts, and stores the ciphertext as ``Text``.  On read the
  ciphertext is decrypted and deserialized back to a Python object.

Key management
--------------
A single master key (``LLM_PORT_API_ENCRYPTION_KEY``) is stretched via
HKDF-SHA256 into independent per-purpose subkeys.  Each column declares
a *purpose* label (e.g. ``"chat-content"``, ``"chat-memory"``) so that
compromise of one derived key cannot decrypt columns tagged with a
different purpose.

Call :func:`configure` once at application startup (in ``lifespan.py``)
to activate encryption.  When the master key is empty, encryption is
**disabled** and values pass through as plaintext — this is intentional
for local development and to support brownfield migration (existing
plaintext rows are readable while new rows are encrypted).

EE extension point
------------------
Enterprise Edition can replace the key-provider by calling
:func:`set_key_provider` with a callable that, given a *purpose* string,
returns raw 32-byte key material (e.g. from HashiCorp Vault Transit).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Callable

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from sqlalchemy import Text
from sqlalchemy.types import TypeDecorator

logger = logging.getLogger(__name__)

# Module state ----------------------------------------------------------

_key_provider: Callable[[str], bytes] | None = None
"""Return 32 bytes of key material for a given *purpose* label."""

_FERNET_PREFIX = b"gAAAAA"
"""All Fernet tokens start with this prefix (version byte 0x80 → 'gA')."""


# Public API ------------------------------------------------------------


def configure(master_key: str) -> None:
    """Derive per-purpose encryption keys from *master_key* via HKDF.

    Must be called once during app startup.  If *master_key* is empty,
    encryption is disabled (plaintext passthrough).
    """
    if not master_key:
        logger.warning(
            "ENCRYPTION_KEY is not set — chat data stored in plaintext. "
            "Set LLM_PORT_API_ENCRYPTION_KEY in production.",
        )
        return

    raw = master_key.encode("utf-8")

    def _derive(purpose: str) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=purpose.encode("utf-8"),
        ).derive(raw)

    set_key_provider(_derive)


def set_key_provider(provider: Callable[[str], bytes]) -> None:
    """Replace the key provider (used by EE Vault integration)."""
    global _key_provider  # noqa: PLW0603
    _key_provider = provider


def _fernet_for(purpose: str) -> Fernet | None:
    """Return a Fernet instance keyed for *purpose*, or ``None``."""
    if _key_provider is None:
        return None
    raw = _key_provider(purpose)
    return Fernet(base64.urlsafe_b64encode(raw))


def _is_ciphertext(value: str) -> bool:
    """Heuristic: does *value* look like a Fernet token?"""
    try:
        return value.encode("ascii")[:6] == _FERNET_PREFIX
    except (UnicodeEncodeError, AttributeError):
        return False


# TypeDecorators --------------------------------------------------------


class EncryptedText(TypeDecorator):
    """A ``Text`` column whose value is Fernet-encrypted at rest.

    Usage::

        content = mapped_column(EncryptedText("chat-content"), nullable=False)
    """

    impl = Text
    cache_ok = True

    def __init__(self, purpose: str = "default") -> None:
        super().__init__()
        self._purpose = purpose

    def process_bind_param(self, value: str | None, dialect: Any) -> str | None:
        if value is None:
            return None
        f = _fernet_for(self._purpose)
        if f is None:
            return value  # encryption disabled
        return f.encrypt(value.encode("utf-8")).decode("ascii")

    def process_result_value(self, value: str | None, dialect: Any) -> str | None:
        if value is None:
            return None
        if not _is_ciphertext(value):
            return value  # plaintext (legacy or encryption disabled)
        f = _fernet_for(self._purpose)
        if f is None:
            return value  # can't decrypt without key
        try:
            return f.decrypt(value.encode("ascii")).decode("utf-8")
        except InvalidToken:
            logger.error("Failed to decrypt column (purpose=%s)", self._purpose)
            raise


class EncryptedJSON(TypeDecorator):
    """A ``Text`` column storing Fernet-encrypted JSON.

    The Python value is a ``dict`` or ``list``.  On bind it is serialized
    to a JSON string and encrypted; on result it is decrypted and
    deserialized.  The database column is ``Text`` (not ``JSON``),
    because ciphertext is not valid JSON.

    Usage::

        tool_call_json = mapped_column(
            EncryptedJSON("chat-content"), nullable=True,
        )
    """

    impl = Text
    cache_ok = True

    def __init__(self, purpose: str = "default") -> None:
        super().__init__()
        self._purpose = purpose

    def process_bind_param(
        self,
        value: dict[str, Any] | list[Any] | None,
        dialect: Any,
    ) -> str | None:
        if value is None:
            return None
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        f = _fernet_for(self._purpose)
        if f is None:
            return raw  # store as plain JSON text
        return f.encrypt(raw.encode("utf-8")).decode("ascii")

    def process_result_value(
        self,
        value: str | None,
        dialect: Any,
    ) -> dict[str, Any] | list[Any] | None:
        if value is None:
            return None
        if _is_ciphertext(value):
            f = _fernet_for(self._purpose)
            if f is None:
                return None  # can't decrypt
            try:
                value = f.decrypt(value.encode("ascii")).decode("utf-8")
            except InvalidToken:
                logger.error("Failed to decrypt JSON column (purpose=%s)", self._purpose)
                raise
        return json.loads(value)


def decrypt_value(ciphertext: str, *, purpose: str = "default") -> str:
    """Decrypt a standalone Fernet-encrypted string.

    Used by the LLM adapter to decrypt API keys stored on
    ``LLMProviderInstance.api_key_encrypted``.
    """
    f = _fernet_for(purpose)
    if f is None:
        return ciphertext  # encryption disabled — stored as plaintext
    if not _is_ciphertext(ciphertext):
        return ciphertext  # plaintext value
    return f.decrypt(ciphertext.encode("ascii")).decode("utf-8")
