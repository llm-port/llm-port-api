"""CRUD endpoints for chat projects and sessions."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from starlette import status

from llm_port_api.db.dao.session_dao import SessionDAO
from llm_port_api.services.gateway.auth import AuthContext, get_auth_context
from llm_port_api.services.gateway.errors import GatewayError, error_response
from llm_port_api.services.gateway.session_schemas import (
    MessageDTO,
    ProjectCreateRequest,
    ProjectDTO,
    ProjectUpdateRequest,
    SessionCreateRequest,
    SessionDTO,
    SessionUpdateRequest,
    SummaryDTO,
)

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])

_UNSET = object()

# ── Projects ──────────────────────────────────────────────────────


@router.get("/projects")
async def list_projects(
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    projects = await dao.list_projects(
        tenant_id=auth.tenant_id, user_id=auth.user_id,
    )
    data = [ProjectDTO.model_validate(p).model_dump(mode="json") for p in projects]
    return JSONResponse(status_code=200, content={"data": data})


@router.post("/projects", status_code=status.HTTP_201_CREATED)
async def create_project(
    body: ProjectCreateRequest,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    # Enforce capacity limit
    capacity = getattr(request.app.state, "_resource_capacity", {})
    limit = capacity.get("projects")
    if limit is not None:
        count = await dao.count_projects(
            tenant_id=auth.tenant_id, user_id=auth.user_id,
        )
        if count >= limit:
            raise GatewayError(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                message=(
                    f"Project limit reached ({limit}). "
                    "Upgrade to LLM.port Enterprise for unlimited projects."
                ),
                code="resource_limit_reached",
            )

    project = await dao.create_project(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        name=body.name,
        description=body.description,
        system_instructions=body.system_instructions,
        model_alias=body.model_alias,
        metadata_json=body.metadata_json,
    )
    data = ProjectDTO.model_validate(project).model_dump(mode="json")
    return JSONResponse(status_code=201, content=data)


@router.get("/projects/{project_id}")
async def get_project(
    project_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    project = await dao.get_project(
        project_id=project_id, tenant_id=auth.tenant_id, user_id=auth.user_id,
    )
    if not project:
        raise GatewayError(
            status_code=404,
            message="Project not found.",
            code="project_not_found",
        )
    data = ProjectDTO.model_validate(project).model_dump(mode="json")
    return JSONResponse(status_code=200, content=data)


@router.patch("/projects/{project_id}")
async def update_project(
    project_id: uuid.UUID,
    body: ProjectUpdateRequest,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    fields = body.model_dump(exclude_unset=True)
    project = await dao.update_project(
        project_id=project_id,
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        **fields,
    )
    if not project:
        raise GatewayError(
            status_code=404,
            message="Project not found.",
            code="project_not_found",
        )
    data = ProjectDTO.model_validate(project).model_dump(mode="json")
    return JSONResponse(status_code=200, content=data)


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: uuid.UUID,
    request: Request,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> None:
    # Clean up attachment files before DB cascade deletes records
    file_store = getattr(request.app.state, "chat_file_store", None)
    if file_store:
        attachments = await dao.list_attachments_for_project(project_id=project_id)
        for att in attachments:
            try:
                await file_store.delete(att.storage_key)
            except Exception:
                pass
    deleted = await dao.delete_project(
        project_id=project_id, tenant_id=auth.tenant_id, user_id=auth.user_id,
    )
    if not deleted:
        raise GatewayError(
            status_code=404,
            message="Project not found.",
            code="project_not_found",
        )


# ── Sessions ──────────────────────────────────────────────────────


@router.get("")
async def list_sessions(
    project_id: uuid.UUID | None = None,
    session_status: str | None = None,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    from llm_port_api.db.models.gateway import SessionStatus  # noqa: PLC0415

    status_enum = None
    if session_status:
        try:
            status_enum = SessionStatus(session_status)
        except ValueError:
            pass

    sessions = await dao.list_sessions(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        project_id=project_id,
        status=status_enum,
    )
    data = [SessionDTO.model_validate(s).model_dump(mode="json") for s in sessions]
    return JSONResponse(status_code=200, content={"data": data})


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreateRequest,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    # Validate project ownership if project_id is given
    if body.project_id:
        project = await dao.get_project(
            project_id=body.project_id,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
        )
        if not project:
            raise GatewayError(
                status_code=404,
                message="Project not found.",
                code="project_not_found",
            )

    sess = await dao.create_session(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        project_id=body.project_id,
        title=body.title,
        metadata_json=body.metadata_json,
    )
    data = SessionDTO.model_validate(sess).model_dump(mode="json")
    return JSONResponse(status_code=201, content=data)


@router.get("/{session_id}")
async def get_session(
    session_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    sess = await dao.get_session(
        session_id=session_id, tenant_id=auth.tenant_id, user_id=auth.user_id,
    )
    if not sess:
        raise GatewayError(
            status_code=404,
            message="Session not found.",
            code="session_not_found",
        )
    data = SessionDTO.model_validate(sess).model_dump(mode="json")
    return JSONResponse(status_code=200, content=data)


@router.patch("/{session_id}")
async def update_session(
    session_id: uuid.UUID,
    body: SessionUpdateRequest,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    from llm_port_api.db.models.gateway import SessionStatus  # noqa: PLC0415

    fields = body.model_dump(exclude_unset=True)
    if "status" in fields and fields["status"] is not None:
        try:
            fields["status"] = SessionStatus(fields["status"])
        except ValueError:
            raise GatewayError(
                status_code=400,
                message=f"Invalid status: {fields['status']}",
                code="invalid_status",
            )
    sess = await dao.update_session(
        session_id=session_id,
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        **fields,
    )
    if not sess:
        raise GatewayError(
            status_code=404,
            message="Session not found.",
            code="session_not_found",
        )
    data = SessionDTO.model_validate(sess).model_dump(mode="json")
    return JSONResponse(status_code=200, content=data)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> None:
    deleted = await dao.delete_session(
        session_id=session_id, tenant_id=auth.tenant_id, user_id=auth.user_id,
    )
    if not deleted:
        raise GatewayError(
            status_code=404,
            message="Session not found.",
            code="session_not_found",
        )


# ── Messages ──────────────────────────────────────────────────────


@router.get("/{session_id}/messages")
async def list_messages(
    session_id: uuid.UUID,
    limit: int = 100,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    # Verify session ownership
    sess = await dao.get_session(
        session_id=session_id, tenant_id=auth.tenant_id, user_id=auth.user_id,
    )
    if not sess:
        raise GatewayError(
            status_code=404,
            message="Session not found.",
            code="session_not_found",
        )

    messages = await dao.list_messages(session_id=session_id, limit=limit)
    data = [MessageDTO.model_validate(m).model_dump(mode="json") for m in messages]
    return JSONResponse(status_code=200, content={"data": data})


# ── Summaries ─────────────────────────────────────────────────────


@router.get("/{session_id}/summary")
async def get_session_summary(
    session_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    sess = await dao.get_session(
        session_id=session_id, tenant_id=auth.tenant_id, user_id=auth.user_id,
    )
    if not sess:
        raise GatewayError(
            status_code=404,
            message="Session not found.",
            code="session_not_found",
        )
    summary = await dao.get_latest_summary(session_id=session_id)
    if not summary:
        return JSONResponse(status_code=200, content={"data": None})
    data = SummaryDTO.model_validate(summary).model_dump(mode="json")
    return JSONResponse(status_code=200, content={"data": data})
