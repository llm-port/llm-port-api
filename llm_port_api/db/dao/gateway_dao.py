import uuid
from dataclasses import dataclass

from fastapi import Depends
from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_port_api.db.dependencies import get_db_session
from llm_port_api.db.models.gateway import (
    LLMGatewayRequestLog,
    LLMModelAlias,
    LLMPoolMembership,
    LLMProviderInstance,
    ProviderHealthStatus,
    ProviderType,
    TenantLLMPolicy,
)


@dataclass(slots=True, frozen=True)
class RoutedInstance:
    """Resolved candidate instance for a model alias."""

    alias: str
    instance_id: uuid.UUID
    provider_type: ProviderType
    base_url: str
    weight: float
    max_concurrency: int
    api_key_encrypted: str | None = None
    litellm_provider: str | None = None
    litellm_model: str | None = None
    extra_params: dict | None = None


class GatewayDAO:
    """Data access layer for gateway control metadata."""

    def __init__(self, session: AsyncSession = Depends(get_db_session)) -> None:
        self.session = session

    async def list_enabled_aliases_for_tenant(
        self,
        tenant_id: str,
    ) -> list[LLMModelAlias]:
        """List aliases visible for tenant that have at least one healthy provider."""
        policy = await self.get_tenant_policy(tenant_id)

        # Only return aliases backed by at least one healthy, enabled provider.
        healthy_provider = (
            exists()
            .where(
                LLMPoolMembership.model_alias == LLMModelAlias.alias,
                LLMPoolMembership.enabled.is_(True),
            )
            .where(
                LLMProviderInstance.id == LLMPoolMembership.provider_instance_id,
                LLMProviderInstance.enabled.is_(True),
                LLMProviderInstance.health_status == ProviderHealthStatus.HEALTHY,
            )
        )

        query = select(LLMModelAlias).where(
            LLMModelAlias.enabled.is_(True),
            healthy_provider,
        )
        if policy and policy.allowed_model_aliases:
            query = query.where(LLMModelAlias.alias.in_(policy.allowed_model_aliases))
        result = await self.session.execute(query.order_by(LLMModelAlias.alias.asc()))
        return list(result.scalars().all())

    async def get_tenant_policy(self, tenant_id: str) -> TenantLLMPolicy | None:
        """Fetch tenant policy."""
        result = await self.session.execute(
            select(TenantLLMPolicy).where(TenantLLMPolicy.tenant_id == tenant_id),
        )
        return result.scalar_one_or_none()

    async def resolve_candidates(
        self,
        *,
        alias: str,
        tenant_id: str,
    ) -> list[RoutedInstance]:
        """Resolve enabled and healthy candidates for the given alias/tenant."""
        policy = await self.get_tenant_policy(tenant_id)

        # Enforce tenant alias allowlist before running the heavier join.
        if (
            policy
            and policy.allowed_model_aliases
            and alias not in policy.allowed_model_aliases
        ):
            return []

        query = (
            select(LLMModelAlias, LLMPoolMembership, LLMProviderInstance)
            .join(
                LLMPoolMembership,
                LLMPoolMembership.model_alias == LLMModelAlias.alias,
            )
            .join(
                LLMProviderInstance,
                LLMProviderInstance.id == LLMPoolMembership.provider_instance_id,
            )
            .where(
                LLMModelAlias.alias == alias,
                LLMModelAlias.enabled.is_(True),
                LLMPoolMembership.enabled.is_(True),
                LLMProviderInstance.enabled.is_(True),
                LLMProviderInstance.health_status == ProviderHealthStatus.HEALTHY,
            )
        )
        if policy and policy.allowed_provider_types:
            query = query.where(
                LLMProviderInstance.type.in_(policy.allowed_provider_types),
            )

        rows = (await self.session.execute(query)).all()
        candidates: list[RoutedInstance] = []
        for _, membership, instance in rows:
            weight = (
                membership.weight_override
                if membership.weight_override is not None
                else instance.weight
            )
            candidates.append(
                RoutedInstance(
                    alias=alias,
                    instance_id=instance.id,
                    provider_type=instance.type,
                    base_url=instance.base_url.rstrip("/"),
                    weight=float(weight),
                    max_concurrency=max(instance.max_concurrency, 1),
                    api_key_encrypted=instance.api_key_encrypted,
                    litellm_provider=instance.litellm_provider,
                    litellm_model=instance.litellm_model,
                    extra_params=instance.extra_params,
                ),
            )
        return candidates

    async def insert_request_log(
        self,
        *,
        request_id: str,
        trace_id: str | None,
        tenant_id: str,
        user_id: str,
        model_alias: str | None,
        provider_instance_id: uuid.UUID | None,
        endpoint: str,
        status_code: int,
        latency_ms: int,
        ttft_ms: int | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        error_code: str | None,
    ) -> LLMGatewayRequestLog:
        """Insert request audit log row."""
        row = LLMGatewayRequestLog(
            request_id=request_id,
            trace_id=trace_id,
            tenant_id=tenant_id,
            user_id=user_id,
            model_alias=model_alias,
            provider_instance_id=provider_instance_id,
            endpoint=endpoint,
            status_code=status_code,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            error_code=error_code,
        )
        self.session.add(row)
        await self.session.flush()
        return row
