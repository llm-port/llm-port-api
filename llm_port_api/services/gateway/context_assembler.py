"""Assemble the context window for a session-aware chat completion.

The assembler builds the final ``messages`` list injected into the upstream
LLM request by fitting the following segments into a configurable token
budget (ordered by priority):

1. System instructions (project-level ``system_instructions``)
2. Latest rolling summary (compressed older history)
3. Active memory facts (user / project / session scope)
3.5. Attached document/image context
4. RAG context (if present - already injected earlier in pipeline)
5. Recent messages (most recent N turns from the session)
6. Current user message (always included)
"""

from __future__ import annotations

import base64
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from llm_port_api.db.dao.session_dao import SessionDAO
from llm_port_api.db.models.gateway import (
    ChatAttachment,
    ChatMessage,
    ChatProject,
    ExtractionStatus,
    MemoryFact,
    SessionSummary,
)
from llm_port_api.services.gateway.file_store import FileStore

logger = logging.getLogger(__name__)

# Rough token estimate for an image content part (~85 tokens).
_IMAGE_TOKEN_ESTIMATE = 85


@dataclass(slots=True)
class AssembledContext:
    """Result of context assembly."""

    messages: list[dict[str, Any]]
    summary_used: SessionSummary | None = None
    facts_used: list[MemoryFact] = field(default_factory=list)
    attachments_used: list[ChatAttachment] = field(default_factory=list)
    total_token_estimate: int = 0


def _estimate_tokens(content: str | list[Any]) -> int:
    """Rough token estimation (~4 chars per token for text, ~85 per image)."""
    if isinstance(content, str):
        return max(1, len(content) // 4)
    # Multimodal content parts list
    total = 0
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                total += max(1, len(part.get("text", "")) // 4)
            elif part.get("type") == "image_url":
                total += _IMAGE_TOKEN_ESTIMATE
            else:
                total += 10
        else:
            total += 10
    return max(1, total)


class ContextAssembler:
    """Build the context window for a session-aware request."""

    def __init__(
        self,
        *,
        dao: SessionDAO,
        max_recent_messages: int = 20,
        token_budget: int = 4096,
        file_store: FileStore | None = None,
    ) -> None:
        self.dao = dao
        self.max_recent_messages = max_recent_messages
        self.token_budget = token_budget
        self.file_store = file_store

    async def assemble(
        self,
        *,
        session_id: uuid.UUID,
        tenant_id: str,
        user_id: str,
        current_messages: list[dict[str, Any]],
        project: ChatProject | None = None,
    ) -> AssembledContext:
        """Assemble the full context window.

        ``current_messages`` contains the messages from the current request
        (typically just the latest user message).
        """
        result = AssembledContext(messages=[])
        budget = self.token_budget
        total_tokens = 0

        # 1. System instructions from project
        if project and project.system_instructions:
            sys_tokens = _estimate_tokens(project.system_instructions)
            result.messages.append({
                "role": "system",
                "content": project.system_instructions,
            })
            total_tokens += sys_tokens
            budget -= sys_tokens

        # 2. Rolling summary
        summary = await self.dao.get_latest_summary(session_id=session_id)
        if summary:
            summary_tokens = summary.token_estimate or _estimate_tokens(
                summary.summary_text,
            )
            if summary_tokens <= budget:
                result.messages.append({
                    "role": "system",
                    "content": f"[Session Summary]\n{summary.summary_text}",
                })
                result.summary_used = summary
                total_tokens += summary_tokens
                budget -= summary_tokens

        # 3. Memory facts
        facts = await self._collect_facts(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            project_id=project.id if project else None,
        )
        if facts:
            fact_lines = [f"- {f.key}: {f.value}" for f in facts]
            fact_block = "[Memory]\n" + "\n".join(fact_lines)
            fact_tokens = _estimate_tokens(fact_block)
            if fact_tokens <= budget:
                result.messages.append({
                    "role": "system",
                    "content": fact_block,
                })
                result.facts_used = facts
                total_tokens += fact_tokens
                budget -= fact_tokens

        # 3.5. Attached document/image context
        budget, total_tokens = await self._inject_attachments(
            result=result,
            session_id=session_id,
            project_id=project.id if project else None,
            budget=budget,
            total_tokens=total_tokens,
        )

        # 4. Recent messages from history (after the summary cutoff point)
        after_id = summary.last_message_id if summary else None
        recent = await self.dao.get_recent_messages(
            session_id=session_id,
            limit=self.max_recent_messages,
        )
        # If we have a summary, only include messages after the summary cutoff
        if after_id and recent:
            recent = [m for m in recent if m.created_at > summary.created_at]

        for msg in recent:
            msg_tokens = msg.token_estimate or _estimate_tokens(msg.content)
            if msg_tokens > budget:
                break
            result.messages.append({
                "role": msg.role,
                "content": msg.content,
            })
            total_tokens += msg_tokens
            budget -= msg_tokens

        # 5. Current request messages (always included)
        for msg in current_messages:
            content = msg.get("content", "")
            msg_tokens = _estimate_tokens(content)
            result.messages.append(msg)
            total_tokens += msg_tokens

        result.total_token_estimate = total_tokens
        return result

    async def _inject_attachments(
        self,
        *,
        result: AssembledContext,
        session_id: uuid.UUID,
        project_id: uuid.UUID | None,
        budget: int,
        total_tokens: int,
    ) -> tuple[int, int]:
        """Inject attachment context (documents as text, images as base64)."""
        attachments: list[ChatAttachment] = []
        attachments.extend(
            await self.dao.list_attachments_for_session(session_id=session_id),
        )
        if project_id:
            project_atts = await self.dao.list_attachments_for_project(
                project_id=project_id,
            )
            seen = {a.id for a in attachments}
            attachments.extend(a for a in project_atts if a.id not in seen)

        if not attachments:
            return budget, total_tokens

        for att in attachments:
            if att.extraction_status == ExtractionStatus.COMPLETED and att.extracted_text:
                # Document attachment → inject as system message
                truncation_note = ""
                if att.truncated:
                    truncation_note = " (truncated)"
                text_block = f"[Attached: {att.filename}{truncation_note}]\n{att.extracted_text}"
                text_tokens = _estimate_tokens(text_block)
                if text_tokens > budget:
                    logger.debug(
                        "Skipping attachment %s: %d tokens exceeds budget %d",
                        att.filename, text_tokens, budget,
                    )
                    continue
                result.messages.append({
                    "role": "system",
                    "content": text_block,
                })
                result.attachments_used.append(att)
                total_tokens += text_tokens
                budget -= text_tokens

            elif att.extraction_status == ExtractionStatus.SKIPPED and self.file_store:
                # Image attachment → inject as image_url content part
                try:
                    img_bytes = await self.file_store.get_bytes(att.storage_key)
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    data_uri = f"data:{att.content_type};base64,{b64}"
                    img_tokens = _IMAGE_TOKEN_ESTIMATE
                    if img_tokens > budget:
                        continue
                    result.messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"[Attached image: {att.filename}]"},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    })
                    result.attachments_used.append(att)
                    total_tokens += img_tokens
                    budget -= img_tokens
                except Exception:
                    logger.warning(
                        "Failed to read image attachment %s", att.filename, exc_info=True,
                    )

        return budget, total_tokens

    async def _collect_facts(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: uuid.UUID,
        project_id: uuid.UUID | None,
    ) -> list[MemoryFact]:
        """Gather active facts from all relevant scopes."""
        from llm_port_api.db.models.gateway import MemoryFactScope  # noqa: PLC0415

        facts: list[MemoryFact] = []

        # User-level facts
        facts.extend(
            await self.dao.list_active_facts(
                tenant_id=tenant_id,
                user_id=user_id,
                scope=MemoryFactScope.USER,
            ),
        )

        # Project-level facts
        if project_id:
            facts.extend(
                await self.dao.list_active_facts(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    scope=MemoryFactScope.PROJECT,
                    project_id=project_id,
                ),
            )

        # Session-level facts
        facts.extend(
            await self.dao.list_active_facts(
                tenant_id=tenant_id,
                user_id=user_id,
                scope=MemoryFactScope.SESSION,
                session_id=session_id,
            ),
        )

        return facts
