"""REST endpoints for chat file attachments."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette import status

from llm_port_api.db.dao.session_dao import SessionDAO
from llm_port_api.db.models.gateway import AttachmentScope
from llm_port_api.services.gateway.attachment_service import (
    AttachmentError,
    ChatAttachmentService,
)
from llm_port_api.services.gateway.auth import AuthContext, get_auth_context
from llm_port_api.services.gateway.docling_client import ChatDoclingClient
from llm_port_api.services.gateway.session_schemas import (
    AttachmentDTO,
    AttachmentStatsDTO,
    AttachmentUploadResponse,
)
from llm_port_api.settings import settings

router = APIRouter(prefix="/v1/sessions", tags=["attachments"])


def _build_service(request: Request, dao: SessionDAO) -> ChatAttachmentService:
    file_store = getattr(request.app.state, "chat_file_store", None)
    if file_store is None:
        from llm_port_api.services.gateway.errors import GatewayError  # noqa: PLC0415

        raise GatewayError(
            status_code=501,
            message="Chat attachments are not enabled.",
            code="attachments_disabled",
        )

    docling: ChatDoclingClient | None = None
    if settings.chat_docling_url:
        docling = ChatDoclingClient(settings.chat_docling_url)

    return ChatAttachmentService(
        dao=dao, file_store=file_store, docling_client=docling,
    )


# ── Session-scoped attachments ────────────────────────────────────


@router.post(
    "/{session_id}/attachments",
    status_code=status.HTTP_201_CREATED,
)
async def upload_session_attachment(
    session_id: uuid.UUID,
    file: UploadFile,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    """Upload a file attachment scoped to a chat session."""
    svc = _build_service(request, dao)
    file_bytes = await file.read()
    try:
        result = await svc.upload(
            file_bytes=file_bytes,
            filename=file.filename or "unnamed",
            content_type=file.content_type or "application/octet-stream",
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            session_id=session_id,
            scope=AttachmentScope.SESSION,
        )
    except AttachmentError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.message, "code": "attachment_error"}},
        )

    dto = AttachmentDTO.model_validate(result["attachment"])
    resp = AttachmentUploadResponse(
        attachment=dto,
        extracted_text_length=result["extracted_text_length"],
        token_estimate=result["token_estimate"],
    )
    return JSONResponse(
        status_code=201,
        content=resp.model_dump(mode="json"),
    )


@router.get("/{session_id}/attachments")
async def list_session_attachments(
    session_id: uuid.UUID,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    svc = _build_service(request, dao)
    attachments = await svc.list_for_session(session_id=session_id)
    data = [
        AttachmentDTO.model_validate(a).model_dump(mode="json")
        for a in attachments
    ]
    return JSONResponse(status_code=200, content={"data": data})


@router.delete(
    "/{session_id}/attachments/{attachment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_session_attachment(
    session_id: uuid.UUID,
    attachment_id: uuid.UUID,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    svc = _build_service(request, dao)
    deleted = await svc.delete_attachment(
        attachment_id=attachment_id,
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
    )
    if not deleted:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Attachment not found", "code": "not_found"}},
        )
    return JSONResponse(status_code=204, content=None)


# ── Project-scoped attachments ────────────────────────────────────


@router.post(
    "/projects/{project_id}/attachments",
    status_code=status.HTTP_201_CREATED,
)
async def upload_project_attachment(
    project_id: uuid.UUID,
    file: UploadFile,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    """Upload a file attachment scoped to a project."""
    svc = _build_service(request, dao)
    file_bytes = await file.read()
    try:
        result = await svc.upload(
            file_bytes=file_bytes,
            filename=file.filename or "unnamed",
            content_type=file.content_type or "application/octet-stream",
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            project_id=project_id,
            scope=AttachmentScope.PROJECT,
        )
    except AttachmentError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.message, "code": "attachment_error"}},
        )

    dto = AttachmentDTO.model_validate(result["attachment"])
    resp = AttachmentUploadResponse(
        attachment=dto,
        extracted_text_length=result["extracted_text_length"],
        token_estimate=result["token_estimate"],
    )
    return JSONResponse(
        status_code=201,
        content=resp.model_dump(mode="json"),
    )


@router.get("/projects/{project_id}/attachments")
async def list_project_attachments(
    project_id: uuid.UUID,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    svc = _build_service(request, dao)
    attachments = await svc.list_for_project(project_id=project_id)
    data = [
        AttachmentDTO.model_validate(a).model_dump(mode="json")
        for a in attachments
    ]
    return JSONResponse(status_code=200, content={"data": data})


# ── Stats ─────────────────────────────────────────────────────────


@router.get("/attachments/stats")
async def attachment_stats(
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    svc = _build_service(request, dao)
    raw = await svc.stats(tenant_id=auth.tenant_id, user_id=auth.user_id)
    dto = AttachmentStatsDTO(
        total_count=raw["count"],
        total_bytes=raw["total_bytes"],
    )
    return JSONResponse(status_code=200, content=dto.model_dump(mode="json"))
