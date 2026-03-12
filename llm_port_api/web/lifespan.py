from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
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

from llm_port_api.services.gateway.file_store import LocalFileStore
from llm_port_api.services.gateway.observability import GatewayObservability
from llm_port_api.services.gateway.proxy import create_shared_http_client
from llm_port_api.services.gateway.jwt_secret import load_jwt_secret_from_backend_db
from llm_port_api.services.gateway.settings_loader import load_system_settings_from_backend_db
from llm_port_api.services.rabbit.lifespan import init_rabbit, shutdown_rabbit
from llm_port_api.services.cache import NoOpCache, RedisCache
from llm_port_api.services.registry import service_registry
from llm_port_api.settings import settings
from llm_port_api.tkq import broker

# ── Optional EE plugin ────────────────────────────────────────────
try:
    from llm_port_ee.plugins.gateway import gateway_plugin  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    gateway_plugin = None
# ──────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)


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
    if settings.redis_enabled:
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
    if settings.redis_enabled:
        RedisInstrumentor().uninstrument()
    SQLAlchemyInstrumentor().uninstrument()
    AioPikaInstrumentor().uninstrument()


async def _load_jwt_secret_from_backend_db() -> None:  # pragma: no cover
    """Compatibility wrapper for startup JWT secret loading."""
    await load_jwt_secret_from_backend_db()


def setup_prometheus(app: FastAPI) -> None:  # pragma: no cover
    """
    Enables prometheus integration.

    :param app: current application.
    """
    PrometheusFastApiInstrumentator(should_group_status_codes=False).instrument(
        app,
    ).expose(app, should_gzip=True, name="prometheus_metrics")


def _init_cache(app: FastAPI) -> None:  # pragma: no cover
    """Initialise the ``CacheBackend`` — ``RedisCache`` when a Redis URL is
    configured, ``NoOpCache`` otherwise (fail-open for Core deployments).
    """
    if settings.redis_enabled:
        from redis.asyncio import ConnectionPool

        pool = ConnectionPool.from_url(str(settings.redis_url))
        app.state.cache_backend = RedisCache(pool)
    else:
        app.state.cache_backend = NoOpCache()


def _setup_db(app: FastAPI) -> None:  # pragma: no cover
    """
    Initialize async SQLAlchemy engine and session factory.

    :param app: current fastapi application.
    """
    engine = create_async_engine(
        str(settings.db_url),
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    app.state.db_engine = engine
    app.state.db_session_factory = session_factory


def _probe_observability_pro() -> bool:
    """Check whether enterprise observability is available.

    Returns ``True`` when the ``llm_port_ee`` gateway plugin is
    installed, indicating enterprise observability features are
    available in-process.
    """
    return gateway_plugin is not None and gateway_plugin.is_available()


def _setup_gateway_observability(app: FastAPI) -> None:
    observability_pro_available = _probe_observability_pro()

    app.state.gateway_observability = GatewayObservability(
        enabled=settings.langfuse_enabled,
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        tracing_enabled=settings.langfuse_tracing_enabled,
        release=settings.langfuse_release,
        debug=settings.langfuse_debug,
        observability_pro_available=observability_pro_available,
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

    from llm_port_api.db.crypto import configure as configure_crypto

    app.middleware_stack = None
    configure_crypto(settings.encryption_key)

    # Configure LiteLLM module defaults
    import litellm as _litellm  # noqa: PLC0415
    _litellm.drop_params = settings.litellm_drop_params
    _litellm.request_timeout = settings.litellm_request_timeout
    if not settings.litellm_verbose:
        _litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    await _load_jwt_secret_from_backend_db()
    await load_system_settings_from_backend_db()
    # Connect to RabbitMQ with retries — RMQ may still be starting.
    if not broker.is_worker_process:
        import asyncio

        for attempt in range(1, 13):
            try:
                await broker.startup()
                break
            except Exception:
                if attempt == 12:
                    raise
                log.warning(
                    "RabbitMQ not ready (attempt %d/12). Retrying in 5 s\u2026",
                    attempt,
                )
                await asyncio.sleep(5)

    _setup_db(app)
    _setup_gateway_observability(app)
    _setup_service_registry(app)
    app.state.http_client = create_shared_http_client(
        timeout_sec=settings.http_timeout_sec,
    )
    # Resource capacity limits (EE plugin overrides to None for unlimited)
    app.state._resource_capacity = {"projects": 0x3}
    if gateway_plugin is not None and hasattr(gateway_plugin, "override_resource_capacity"):
        gateway_plugin.override_resource_capacity(app.state._resource_capacity)
    # Chat file store
    if settings.sessions_enabled:
        app.state.chat_file_store = LocalFileStore(settings.chat_file_store_root)
    else:
        app.state.chat_file_store = None
    setup_opentelemetry(app)
    _init_cache(app)
    init_rabbit(app)
    setup_prometheus(app)
    app.middleware_stack = app.build_middleware_stack()

    yield
    if not broker.is_worker_process:
        await broker.shutdown()
    await app.state.http_client.aclose()
    await app.state.db_engine.dispose()
    app.state.gateway_observability.shutdown()
    await app.state.cache_backend.close()
    await shutdown_rabbit(app)
    stop_opentelemetry(app)
