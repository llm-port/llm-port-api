"""Thin async HTTP client for the Docling document conversion service.

Used by the gateway to extract text from uploaded chat attachments.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10.0
_CONVERT_TIMEOUT = 300.0


class ChatDoclingClient:
    """Calls the Docling ``/api/v1/convert`` endpoint."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=_CONNECT_TIMEOUT,
                    read=_CONVERT_TIMEOUT,
                    write=_CONVERT_TIMEOUT,
                    pool=_CONNECT_TIMEOUT,
                ),
            )
        return self._client

    async def convert(
        self,
        file_bytes: bytes,
        filename: str,
        *,
        max_pages: int | None = None,
    ) -> dict[str, Any]:
        """Convert a document and return ``{content, metadata}``."""
        client = self._ensure_client()
        options: dict[str, Any] = {"format": "markdown", "chunk": False}
        if max_pages is not None:
            options["max_pages"] = max_pages

        resp = await client.post(
            "/api/v1/convert",
            files={"file": (filename, file_bytes)},
            data={"options": json.dumps(options)},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
