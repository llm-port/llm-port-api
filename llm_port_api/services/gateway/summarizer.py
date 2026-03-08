"""Rolling session summarizer.

When the number of unsummarised messages in a session crosses a threshold,
the summarizer compresses older turns into a rolling summary by calling
a (configurable) LLM model alias through the gateway's own upstream proxy.
"""

from __future__ import annotations

import logging
import uuid

from llm_port_api.db.dao.session_dao import SessionDAO
from llm_port_api.db.models.gateway import ChatMessage, SessionSummary
from llm_port_api.services.gateway.context_assembler import _estimate_tokens

logger = logging.getLogger(__name__)

_SUMMARISE_SYSTEM_PROMPT = (
    "You are a concise conversation summarizer. "
    "Compress the following conversation into a brief summary that preserves "
    "key facts, decisions, and context needed to continue the conversation. "
    "Keep the summary under 300 words. Output ONLY the summary, no preamble."
)


class Summarizer:
    """Produces rolling summaries for a chat session."""

    def __init__(
        self,
        *,
        dao: SessionDAO,
        threshold: int = 10,
        proxy_fn: object = None,
    ) -> None:
        self.dao = dao
        self.threshold = threshold
        # proxy_fn is an async callable(messages, model) -> str
        # injected from GatewayService when EE or a summarizer model is configured
        self._proxy_fn = proxy_fn

    async def maybe_summarise(
        self,
        *,
        session_id: uuid.UUID,
    ) -> SessionSummary | None:
        """Check if summarisation is needed and produce one if so.

        Returns the new summary, or None if the threshold wasn't reached
        or no proxy function is configured.
        """
        if self._proxy_fn is None:
            return None

        existing = await self.dao.get_latest_summary(session_id=session_id)

        # Count messages since last summary
        messages = await self.dao.list_messages(
            session_id=session_id,
            after_message_id=existing.last_message_id if existing else None,
        )

        if len(messages) < self.threshold:
            return None

        return await self._produce_summary(
            session_id=session_id,
            previous_summary=existing,
            new_messages=messages,
        )

    async def _produce_summary(
        self,
        *,
        session_id: uuid.UUID,
        previous_summary: SessionSummary | None,
        new_messages: list[ChatMessage],
    ) -> SessionSummary | None:
        """Build summary by calling the configured LLM."""
        # Build the conversation text to summarise
        parts: list[str] = []
        if previous_summary:
            parts.append(f"[Previous Summary]\n{previous_summary.summary_text}")
        for msg in new_messages:
            parts.append(f"{msg.role}: {msg.content}")

        conversation_text = "\n\n".join(parts)

        try:
            summary_text = await self._proxy_fn(
                [
                    {"role": "system", "content": _SUMMARISE_SYSTEM_PROMPT},
                    {"role": "user", "content": conversation_text},
                ],
            )
        except Exception:
            logger.exception("Failed to produce summary for session %s", session_id)
            return None

        if not summary_text:
            return None

        last_msg = new_messages[-1]
        token_est = _estimate_tokens(summary_text)

        return await self.dao.create_summary(
            session_id=session_id,
            summary_text=summary_text,
            last_message_id=last_msg.id,
            token_estimate=token_est,
        )
