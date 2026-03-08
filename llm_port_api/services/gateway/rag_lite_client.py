"""HTTP client for calling the RAG Lite search endpoint on the backend.

Used by the gateway pipeline to retrieve context before sending a chat
completion request to the upstream LLM.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class RagLiteClient:
    """Async client wrapping the backend RAG Lite search endpoint."""

    def __init__(self, *, base_url: str, http_client: httpx.AsyncClient) -> None:
        self._base = base_url.rstrip("/")
        self._http = http_client

    async def search(
        self,
        *,
        query: str,
        top_k: int = 5,
        collection_ids: list[str] | None = None,
        api_token: str | None = None,
    ) -> list[dict[str, Any]]:
        """Call ``POST /api/admin/rag/search`` on the backend.

        Returns a list of result dicts with ``chunk_text``, ``filename``,
        ``score``, etc.
        """
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if collection_ids:
            body["collection_ids"] = collection_ids

        headers: dict[str, str] = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        try:
            resp = await self._http.post(
                f"{self._base}/api/admin/rag/search",
                json=body,
                timeout=15.0,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception:
            logger.exception("RAG Lite search call failed — skipping context injection")
            return []
