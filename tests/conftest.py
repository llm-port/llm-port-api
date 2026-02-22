import uuid
from typing import Any, AsyncGenerator
from unittest.mock import Mock

import pytest
from aio_pika import Channel
from aio_pika.abc import AbstractExchange, AbstractQueue
from aio_pika.pool import Pool
from fakeredis import FakeServer
from fakeredis.aioredis import FakeConnection
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis.asyncio import ConnectionPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_port_api.db.dependencies import get_db_session
from llm_port_api.db.meta import meta
from llm_port_api.db.models import load_all_models
from llm_port_api.services.gateway.observability import GatewayObservability
from llm_port_api.services.rabbit.dependencies import get_rmq_channel_pool
from llm_port_api.services.rabbit.lifespan import init_rabbit, shutdown_rabbit
from llm_port_api.services.redis.dependency import get_redis_pool
from llm_port_api.web.application import get_app


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """
    Backend for anyio pytest plugin.

    :return: backend name.
    """
    return "asyncio"


@pytest.fixture
async def test_rmq_pool() -> AsyncGenerator[Channel, None]:
    """
    Create rabbitMQ pool.

    :yield: channel pool.
    """
    app_mock = Mock()
    init_rabbit(app_mock)
    yield app_mock.state.rmq_channel_pool
    await shutdown_rabbit(app_mock)


@pytest.fixture
async def test_exchange_name() -> str:
    """
    Name of an exchange to use in tests.

    :return: name of an exchange.
    """
    return uuid.uuid4().hex


@pytest.fixture
async def test_routing_key() -> str:
    """
    Name of routing key to use while binding test queue.

    :return: key string.
    """
    return uuid.uuid4().hex


@pytest.fixture
async def test_exchange(
    test_exchange_name: str,
    test_rmq_pool: Pool[Channel],
) -> AsyncGenerator[AbstractExchange, None]:
    """
    Creates test exchange.

    :param test_exchange_name: name of an exchange to create.
    :param test_rmq_pool: channel pool for rabbitmq.
    :yield: created exchange.
    """
    try:
        async with test_rmq_pool.acquire() as conn:
            exchange = await conn.declare_exchange(
                name=test_exchange_name,
                auto_delete=True,
            )
            yield exchange
            await exchange.delete(if_unused=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"RabbitMQ unavailable in test environment: {exc}")


@pytest.fixture
async def test_queue(
    test_exchange: AbstractExchange,
    test_rmq_pool: Pool[Channel],
    test_routing_key: str,
) -> AsyncGenerator[AbstractQueue, None]:
    """
    Creates queue connected to exchange.

    :param test_exchange: exchange to bind queue to.
    :param test_rmq_pool: channel pool for rabbitmq.
    :param test_routing_key: routing key to use while binding.
    :yield: queue binded to test exchange.
    """
    async with test_rmq_pool.acquire() as conn:
        queue = await conn.declare_queue(name=uuid.uuid4().hex)
        await queue.bind(
            exchange=test_exchange,
            routing_key=test_routing_key,
        )
        yield queue

        await queue.delete(if_unused=False, if_empty=False)


@pytest.fixture
async def fake_redis_pool() -> AsyncGenerator[ConnectionPool, None]:
    """
    Get instance of a fake redis.

    :yield: FakeRedis instance.
    """
    server = FakeServer()
    server.connected = True
    pool = ConnectionPool(connection_class=FakeConnection, server=server)

    yield pool

    await pool.disconnect()


@pytest.fixture(scope="session")
async def db_engine(anyio_backend: Any) -> AsyncGenerator[Any, None]:
    """
    Create in-memory sqlite engine for tests.

    :yield: async engine.
    """
    load_all_models()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(meta.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine: Any) -> AsyncGenerator[AsyncSession, None]:
    """
    Get transaction-scoped db session.

    :yield: async db session.
    """
    async with db_engine.connect() as conn:
        tx = await conn.begin()
        session_factory = async_sessionmaker(conn, expire_on_commit=False)
        session = session_factory()
        try:
            yield session
        finally:
            await session.close()
            await tx.rollback()


@pytest.fixture
def fastapi_app(
    db_session: AsyncSession,
    fake_redis_pool: ConnectionPool,
    test_rmq_pool: Pool[Channel],
) -> FastAPI:
    """
    Fixture for creating FastAPI app.

    :return: fastapi app with mocked dependencies.
    """
    application = get_app()
    application.state.redis_pool = fake_redis_pool
    application.state.rmq_channel_pool = test_rmq_pool
    application.state.gateway_observability = GatewayObservability(enabled=False)
    application.dependency_overrides[get_db_session] = lambda: db_session
    application.dependency_overrides[get_redis_pool] = lambda: fake_redis_pool
    application.dependency_overrides[get_rmq_channel_pool] = lambda: test_rmq_pool
    return application


@pytest.fixture
async def client(
    fastapi_app: FastAPI,
    anyio_backend: Any,
) -> AsyncGenerator[AsyncClient, None]:
    """
    Fixture that creates client for requesting server.

    :param fastapi_app: the application.
    :yield: client for the app.
    """
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(
        transport=transport, base_url="http://test", timeout=2.0,
    ) as ac:
        yield ac
