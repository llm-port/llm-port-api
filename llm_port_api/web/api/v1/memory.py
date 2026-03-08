"""CRUD endpoints for memory facts."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from starlette import status

from llm_port_api.db.dao.session_dao import SessionDAO
from llm_port_api.db.models.gateway import MemoryFactScope, MemoryFactStatus
from llm_port_api.services.gateway.auth import AuthContext, get_auth_context
from llm_port_api.services.gateway.errors import GatewayError
from llm_port_api.services.gateway.session_schemas import (
    MemoryFactCreateRequest,
    MemoryFactDTO,
    MemoryFactUpdateRequest,
)

router = APIRouter(prefix="/v1/memory", tags=["memory"])


@router.get("/facts")
async def list_facts(
    scope: str | None = None,
    project_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    scope_enum: MemoryFactScope | None = None
    if scope:
        try:
            scope_enum = MemoryFactScope(scope)
        except ValueError:
            raise GatewayError(
                status_code=400,
                message=f"Invalid scope: {scope}",
                code="invalid_scope",
            )

    facts = await dao.list_active_facts(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        scope=scope_enum,
        project_id=project_id,
        session_id=session_id,
    )
    data = [MemoryFactDTO.model_validate(f).model_dump(mode="json") for f in facts]
    return JSONResponse(status_code=200, content={"data": data})


@router.post("/facts", status_code=status.HTTP_201_CREATED)
async def create_fact(
    body: MemoryFactCreateRequest,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    try:
        scope_enum = MemoryFactScope(body.scope)
    except ValueError:
        raise GatewayError(
            status_code=400,
            message=f"Invalid scope: {body.scope}",
            code="invalid_scope",
        )

    fact = await dao.upsert_fact(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        scope=scope_enum,
        key=body.key,
        value=body.value,
        confidence=body.confidence,
        session_id=body.session_id,
        project_id=body.project_id,
        status=MemoryFactStatus.ACTIVE,
    )
    data = MemoryFactDTO.model_validate(fact).model_dump(mode="json")
    return JSONResponse(status_code=201, content=data)


@router.patch("/facts/{fact_id}")
async def update_fact(
    fact_id: uuid.UUID,
    body: MemoryFactUpdateRequest,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> JSONResponse:
    fields = body.model_dump(exclude_unset=True)

    if "status" in fields and fields["status"] is not None:
        try:
            status_enum = MemoryFactStatus(fields["status"])
        except ValueError:
            raise GatewayError(
                status_code=400,
                message=f"Invalid status: {fields['status']}",
                code="invalid_status",
            )
        updated = await dao.update_fact_status(
            fact_id=fact_id,
            tenant_id=auth.tenant_id,
            user_id=auth.user_id,
            status=status_enum,
        )
        if not updated:
            raise GatewayError(
                status_code=404,
                message="Memory fact not found.",
                code="fact_not_found",
            )

    return JSONResponse(status_code=200, content={"ok": True})


@router.delete("/facts/{fact_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_fact(
    fact_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    dao: SessionDAO = Depends(),
) -> None:
    deleted = await dao.delete_fact(
        fact_id=fact_id, tenant_id=auth.tenant_id, user_id=auth.user_id,
    )
    if not deleted:
        raise GatewayError(
            status_code=404,
            message="Memory fact not found.",
            code="fact_not_found",
        )
