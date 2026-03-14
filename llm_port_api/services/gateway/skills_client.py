"""HTTP client for the Skills micro-service.

Used by the gateway pipeline to resolve matching skills for a
request context via the Skills service's internal API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResolvedSkill:
    """A resolved skill ready for injection into the LLM pipeline."""

    skill_id: str
    name: str
    slug: str
    version: int
    body_markdown: str
    priority: int
    score: int
    allowed_tools: list[str] | None = None
    preferred_tools: list[str] | None = None
    forbidden_tools: list[str] | None = None
    knowledge_sources: list[str] | None = None


@dataclass(slots=True)
class SkillResolveResult:
    """Result of a skill resolution call."""

    skills: list[ResolvedSkill] = field(default_factory=list)


class SkillsClient:
    """Async client wrapping Skills service internal HTTP endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        http_client: httpx.AsyncClient,
        service_token: str,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._http = http_client
        self._headers = {"Authorization": f"Bearer {service_token}"}

    async def resolve_skills(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        assistant_id: str | None = None,
        workspace_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        user_query: str | None = None,
        file_types: list[str] | None = None,
        tags: list[str] | None = None,
        max_skills: int = 3,
    ) -> SkillResolveResult:
        """Resolve matching skills for a gateway request context.

        Calls ``POST /api/internal/skills/resolve``.
        """
        body: dict[str, Any] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "assistant_id": assistant_id,
            "workspace_id": workspace_id,
            "project_id": project_id,
            "session_id": session_id,
            "user_query": user_query,
            "file_types": file_types or [],
            "tags": tags or [],
            "max_skills": max_skills,
        }
        try:
            resp = await self._http.post(
                f"{self._base}/api/internal/skills/resolve",
                json=body,
                headers=self._headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            skills = [
                ResolvedSkill(
                    skill_id=s["skill_id"],
                    name=s["name"],
                    slug=s["slug"],
                    version=s["version"],
                    body_markdown=s["body_markdown"],
                    priority=s["priority"],
                    score=s["score"],
                    allowed_tools=s.get("allowed_tools"),
                    preferred_tools=s.get("preferred_tools"),
                    forbidden_tools=s.get("forbidden_tools"),
                    knowledge_sources=s.get("knowledge_sources"),
                )
                for s in data.get("skills", [])
            ]
            return SkillResolveResult(skills=skills)
        except Exception:
            logger.exception("Skills resolve failed")
            return SkillResolveResult()

    async def record_usage(
        self,
        *,
        tenant_id: str,
        skill_id: str,
        version: int,
        session_id: str | None = None,
        user_id: str | None = None,
        matched_by: str | None = None,
    ) -> None:
        """Record skill usage telemetry (fire-and-forget).

        Calls ``POST /api/internal/skills/usage``.
        """
        body = {
            "tenant_id": tenant_id,
            "skill_id": skill_id,
            "version": version,
            "session_id": session_id,
            "user_id": user_id,
            "matched_by": matched_by,
        }
        try:
            resp = await self._http.post(
                f"{self._base}/api/internal/skills/usage",
                json=body,
                headers=self._headers,
                timeout=5.0,
            )
            resp.raise_for_status()
        except Exception:
            logger.debug("Skills usage recording failed", exc_info=True)
