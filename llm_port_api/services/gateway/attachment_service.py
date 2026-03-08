"""Service layer for chat file attachment upload, extraction and management."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from llm_port_api.db.dao.session_dao import SessionDAO
from llm_port_api.db.models.gateway import AttachmentScope, ExtractionStatus
from llm_port_api.services.gateway.docling_client import ChatDoclingClient
from llm_port_api.services.gateway.file_store import FileStore
from llm_port_api.settings import settings

logger = logging.getLogger(__name__)

# Extensions that can be read as plain UTF-8 text without Docling.
_PLAINTEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".html"}

# Extensions treated as images (no text extraction).
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

_MB = 1024 * 1024


class AttachmentError(Exception):
    """Application-level error for attachment operations."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ChatAttachmentService:
    """Handles upload, text extraction, storage and cleanup of chat attachments."""

    def __init__(
        self,
        *,
        dao: SessionDAO,
        file_store: FileStore,
        docling_client: ChatDoclingClient | None = None,
    ) -> None:
        self.dao = dao
        self.file_store = file_store
        self.docling = docling_client

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def upload(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        tenant_id: str,
        user_id: str,
        session_id: uuid.UUID | None = None,
        project_id: uuid.UUID | None = None,
        scope: AttachmentScope = AttachmentScope.SESSION,
    ) -> dict[str, Any]:
        """Validate, store, extract and persist a chat attachment.

        Returns ``{attachment, extracted_text_length, token_estimate}``.
        """
        ext = os.path.splitext(filename)[1].lower()
        allowed = {
            e.strip()
            for e in settings.chat_upload_allowed_extensions.split(",")
            if e.strip()
        }
        if ext not in allowed:
            raise AttachmentError(
                f"File type '{ext}' is not allowed. Allowed: {', '.join(sorted(allowed))}",
            )

        size_bytes = len(file_bytes)
        max_bytes = settings.chat_upload_max_file_mb * _MB
        if size_bytes > max_bytes:
            raise AttachmentError(
                f"File exceeds maximum size of {settings.chat_upload_max_file_mb} MB.",
                status_code=413,
            )

        # Check per-session attachment count limit
        if session_id is not None:
            existing = await self.dao.list_attachments_for_session(
                session_id=session_id,
            )
            if len(existing) >= settings.chat_max_attachments_per_session:
                raise AttachmentError(
                    f"Maximum of {settings.chat_max_attachments_per_session} "
                    "attachments per session reached.",
                )

        # Check total storage limit for tenant+user
        stats = await self.dao.attachment_stats(
            tenant_id=tenant_id, user_id=user_id,
        )
        total_mb = (stats["total_bytes"] + size_bytes) / _MB
        if total_mb > settings.chat_max_total_attachment_mb:
            raise AttachmentError(
                f"Total attachment storage would exceed {settings.chat_max_total_attachment_mb} MB.",
                status_code=413,
            )

        # Store file
        scope_id = str(session_id or project_id or "unscoped")
        file_id = uuid.uuid4().hex
        storage_key = f"{tenant_id}/{scope_id}/{file_id}/{filename}"
        await self.file_store.put_bytes(storage_key, file_bytes)

        # Extract text
        extracted_text: str | None = None
        extraction_status = ExtractionStatus.PENDING
        page_count: int | None = None
        truncated = False

        if ext in _IMAGE_EXTENSIONS:
            extraction_status = ExtractionStatus.SKIPPED
        elif ext in _PLAINTEXT_EXTENSIONS:
            try:
                extracted_text = file_bytes.decode("utf-8")
                extraction_status = ExtractionStatus.COMPLETED
            except UnicodeDecodeError:
                extraction_status = ExtractionStatus.FAILED
        elif self.docling is not None:
            try:
                result = await self.docling.convert(
                    file_bytes,
                    filename,
                    max_pages=settings.chat_attachment_max_pages,
                )
                extracted_text = result.get("content", "")
                metadata = result.get("metadata", {})
                page_count = metadata.get("page_count")
                if page_count and page_count > settings.chat_attachment_max_pages:
                    truncated = True
                    extracted_text += (
                        f"\n\n[Note: Only the first {settings.chat_attachment_max_pages} "
                        f"of {page_count} pages were processed. For full-document "
                        "search, upload to a Knowledge Base.]"
                    )
                extraction_status = ExtractionStatus.COMPLETED
            except Exception:
                logger.warning(
                    "Docling extraction failed for %s", filename, exc_info=True,
                )
                extraction_status = ExtractionStatus.FAILED
        else:
            raise AttachmentError(
                f"Cannot extract text from '{ext}' files. "
                "The document conversion service (Docling) is not configured. "
                "Please upload a plaintext file (.txt, .md, .csv, .json, .html) "
                "or enable the Docling service.",
            )

        attachment = await self.dao.create_attachment(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            storage_key=storage_key,
            extracted_text=extracted_text,
            extraction_status=extraction_status,
            scope=scope,
            page_count=page_count,
            truncated=truncated,
        )

        text_len = len(extracted_text) if extracted_text else 0
        token_est = max(1, text_len // 4) if text_len else 0

        return {
            "attachment": attachment,
            "extracted_text_length": text_len,
            "token_estimate": token_est,
        }

    # ------------------------------------------------------------------
    # Read / List / Delete
    # ------------------------------------------------------------------

    async def get_attachment(
        self,
        *,
        attachment_id: uuid.UUID,
        tenant_id: str,
        user_id: str,
    ) -> Any | None:
        return await self.dao.get_attachment(
            attachment_id=attachment_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    async def list_for_session(self, *, session_id: uuid.UUID) -> list[Any]:
        return await self.dao.list_attachments_for_session(session_id=session_id)

    async def list_for_project(self, *, project_id: uuid.UUID) -> list[Any]:
        return await self.dao.list_attachments_for_project(project_id=project_id)

    async def delete_attachment(
        self,
        *,
        attachment_id: uuid.UUID,
        tenant_id: str,
        user_id: str,
    ) -> bool:
        att = await self.dao.get_attachment(
            attachment_id=attachment_id,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        if not att:
            return False
        await self.file_store.delete(att.storage_key)
        return await self.dao.delete_attachment(attachment_id=attachment_id)

    async def get_file_bytes(self, storage_key: str) -> bytes:
        return await self.file_store.get_bytes(storage_key)

    async def cleanup_session_attachments(
        self,
        *,
        session_id: uuid.UUID,
    ) -> int:
        """Delete all files and DB records for a session's attachments."""
        attachments = await self.dao.list_attachments_for_session(
            session_id=session_id,
        )
        for att in attachments:
            try:
                await self.file_store.delete(att.storage_key)
            except Exception:
                logger.warning(
                    "Failed to delete file %s for attachment %s",
                    att.storage_key, att.id,
                )
        return await self.dao.delete_attachments_for_session(
            session_id=session_id,
        )

    async def stats(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
    ) -> dict[str, int]:
        return await self.dao.attachment_stats(
            tenant_id=tenant_id, user_id=user_id,
        )
