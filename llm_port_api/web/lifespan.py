from __future__ import annotations

import base64
import hashlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI
from loguru import logger
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import (
    DEPLOYMENT_ENVIRONMENT,
    SERVICE_NAME,
    TELEMETRY_SDK_LANGUAGE,
    Resource,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import set_tracer_provider
from prometheus_fastapi_instrumentator.instrumentation import (
    PrometheusFastApiInstrumentator,
)
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from llm_port_api.services.gateway.observability import GatewayObservability
from llm_port_api.services.gateway.proxy import create_shared_http_client
from llm_port_api.services.rabbit.lifespan import init_rabbit, shutdown_rabbit
from llm_port_api.services.redis.lifespan import init_redis, shutdown_redis
from llm_port_api.services.registry import service_registry
from llm_port_api.settings import settings
from llm_port_api.tkq import broker


def setup_opentelemetry(app: FastAPI) -> None:  # pragma: no cover
    """
    Enables opentelemetry instrumentation.

    :param app: current application.
    """
    if not settings.opentelemetry_endpoint:
        return

    tracer_provider = TracerProvider(
        resource=Resource(
            attributes={
                SERVICE_NAME: "llm_port_api",
                TELEMETRY_SDK_LANGUAGE: "python",
                DEPLOYMENT_ENVIRONMENT: settings.environment,
            },
        ),
    )

    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=settings.opentelemetry_endpoint,
                insecure=True,
            ),
        ),
    )

    excluded_endpoints = [
        app.url_path_for("health_check"),
        app.url_path_for("openapi"),
        app.url_path_for("swagger_ui_html"),
        app.url_path_for("swagger_ui_redirect"),
        app.url_path_for("redoc_html"),
        "/metrics",
    ]

    FastAPIInstrumentor().instrument_app(
        app,
        tracer_provider=tracer_provider,
        excluded_urls=",".join(excluded_endpoints),
    )
    RedisInstrumentor().instrument(
        tracer_provider=tracer_provider,
    )
    SQLAlchemyInstrumentor().instrument(
        tracer_provider=tracer_provider,
        engine=app.state.db_engine.sync_engine,
    )
    AioPikaInstrumentor().instrument(
        tracer_provider=tracer_provider,
    )

    set_tracer_provider(tracer_provider=tracer_provider)


def stop_opentelemetry(app: FastAPI) -> None:  # pragma: no cover
    """
    Disables opentelemetry instrumentation.

    :param app: current application.
    """
    if not settings.opentelemetry_endpoint:
        return

    FastAPIInstrumentor().uninstrument_app(app)
    RedisInstrumentor().uninstrument()
    SQLAlchemyInstrumentor().uninstrument()
    AioPikaInstrumentor().uninstrument()


async def _load_jwt_secret_from_backend_db() -> None:  # pragma: no cover
    """Read llm_port_api.jwt_secret from the backend's system_setting_secret table.

    Called on startup when ``settings.backend_db_base`` and
    ``settings.settings_master_key`` are both configured.  The backend encrypts
    its secrets with Fernet (SHA-256(master_key) as the key), so we replicate
    the same derivation here to decrypt.
    """
    if not settings.backend_db_base or not settings.settings_master_key:
        return

    dsn = (
        f"postgresql://{settings.db_user}:{settings.db_pass}"
        f"@{settings.db_host}:{settings.db_port}/{settings.backend_db_base}"
    )
    try:
        conn = await asyncpg.connect(dsn)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JWT secret: could not connect to backend DB — {}", exc)
        return

    try:
        row = await conn.fetchrow(
            "SELECT ciphertext FROM system_setting_secret WHERE key = $1",
            "llm_port_api.jwt_secret",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("JWT secret: query failed — {}", exc)
        return
    finally:
        await conn.close()

    if row is None:
        logger.info("JWT secret: llm_port_api.jwt_secret not set in backend DB, using env var fallback.")
        return

    try:
        from cryptography.fernet import Fernet  # noqa: PLC0415

        digest = hashlib.sha256(settings.settings_master_key.encode()).digest()
        fernet = Fernet(base64.urlsafe_b64encode(digest))
        secret = fernet.decrypt(row["ciphertext"].encode()).decode()
    except Exception as exc:  # noqa: BLE001
        logger.warning("JWT secret: decryption failed — {}", exc)
        return

    settings.jwt_secret = secret
    logger.info("JWT secret loaded from backend DB (llm_port_api.jwt_secret).")


def setup_prometheus(app: FastAPI) -> None:  # pragma: no cover
    """
    Enables prometheus integration.

    :param app: current application.
    """
    PrometheusFastApiInstrumentator(should_group_status_codes=False).instrument(
        app,
    ).expose(app, should_gzip=True, name="prometheus_metrics")


def _setup_db(app: FastAPI) -> None:  # pragma: no cover
    """
    Initialize async SQLAlchemy engine and session factory.

    :param app: current fastapi application.
    """
    engine = create_async_engine(str(settings.db_url), echo=settings.db_echo)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    app.state.db_engine = engine
    app.state.db_session_factory = session_factory


def _setup_gateway_observability(app: FastAPI) -> None:
    app.state.gateway_observability = GatewayObservability(
        enabled=settings.langfuse_enabled,
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        tracing_enabled=settings.langfuse_tracing_enabled,
        release=settings.langfuse_release,
        debug=settings.langfuse_debug,
    )


def _setup_service_registry(app: FastAPI) -> None:
    """Configure the modular service registry from env-vars.

    Each optional service is "enabled" when *both* its feature flag
    (``<name>_enabled``) is ``True`` **and** its URL is set.
    """
    service_registry.configure(
        "pii", enabled=settings.pii_enabled, url=settings.pii_service_url,
    )
    service_registry.configure(
        "auth", enabled=settings.auth_enabled, url=settings.auth_service_url,
    )
    service_registry.configure(
        "rag", enabled=settings.rag_enabled, url=settings.rag_service_url,
    )
    app.state.service_registry = service_registry


@asynccontextmanager
async def lifespan_setup(
    app: FastAPI,
) -> AsyncGenerator[None, None]:  # pragma: no cover
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :return: function that actually performs actions.
    """

    app.middleware_stack = None
    await _load_jwt_secret_from_backend_db()
    if not broker.is_worker_process:
        await broker.startup()
    _setup_db(app)
    _setup_gateway_observability(app)
    _setup_service_registry(app)
    app.state.http_client = create_shared_http_client(
        timeout_sec=settings.http_timeout_sec,
    )
    setup_opentelemetry(app)
    init_redis(app)
    init_rabbit(app)
    setup_prometheus(app)
    app.middleware_stack = app.build_middleware_stack()

    yield
    if not broker.is_worker_process:
        await broker.shutdown()
    await app.state.http_client.aclose()
    await app.state.db_engine.dispose()
    app.state.gateway_observability.shutdown()
    await shutdown_redis(app)
    await shutdown_rabbit(app)
    stop_opentelemetry(app)
