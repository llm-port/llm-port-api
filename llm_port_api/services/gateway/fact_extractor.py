"""Automatic memory fact extraction from conversations.

Extracts durable facts (preferences, decisions, entities) from messages
by sending them through a configurable LLM. Extracted facts are stored
as ``candidate`` status for optional user review before activation.
"""

from __future__ import annotations

import json
import logging
import uuid

from llm_port_api.db.dao.session_dao import SessionDAO
from llm_port_api.db.models.gateway import MemoryFact, MemoryFactScope, MemoryFactStatus

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM_PROMPT = (
    "You are a fact extractor. Analyze the following conversation and extract "
    "key facts worth remembering for future conversations. Return a JSON array "
    "of objects with 'key' (short label) and 'value' (concise description) fields. "
    "Focus on: user preferences, decisions made, important entities, technical "
    "choices, and personal context. Return [] if no notable facts found. "
    "Output ONLY valid JSON, no markdown or preamble."
)


class FactExtractor:
    """Extract memory facts from conversation messages."""

    def __init__(
        self,
        *,
        dao: SessionDAO,
        proxy_fn: object = None,
    ) -> None:
        self.dao = dao
        # proxy_fn: async callable(messages) -> str
        self._proxy_fn = proxy_fn

    async def extract_from_messages(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: uuid.UUID,
        project_id: uuid.UUID | None,
        messages: list[dict[str, str]],
        scope: MemoryFactScope = MemoryFactScope.SESSION,
    ) -> list[MemoryFact]:
        """Extract facts from recent messages and store them."""
        if self._proxy_fn is None:
            return []

        conversation = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
        )

        try:
            raw = await self._proxy_fn(
                [
                    {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": conversation},
                ],
            )
        except Exception:
            logger.exception("Fact extraction failed for session %s", session_id)
            return []

        facts_data = _parse_facts_json(raw)
        if not facts_data:
            return []

        stored: list[MemoryFact] = []
        for item in facts_data:
            key = str(item.get("key", "")).strip()[:256]
            value = str(item.get("value", "")).strip()
            if not key or not value:
                continue

            fact = await self.dao.upsert_fact(
                tenant_id=tenant_id,
                user_id=user_id,
                scope=scope,
                key=key,
                value=value,
                session_id=session_id,
                project_id=project_id,
                status=MemoryFactStatus.CANDIDATE,
            )
            stored.append(fact)

        return stored


def _parse_facts_json(raw: str) -> list[dict]:
    """Best-effort parse of the LLM's fact output."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse fact extraction output as JSON")
        return []

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []
