import enum
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from yarl import URL

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO

    # Variables for the database
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "llm_user"
    db_pass: str = "llm_user"
    db_base: str = "llm_api"
    db_echo: bool = False
    db_url_override: str | None = None

    # Variables for Redis
    redis_host: str = "llm_port_api-redis"
    redis_port: int = 6379
    redis_user: Optional[str] = None
    redis_pass: Optional[str] = None
    redis_base: Optional[int] = None

    # Variables for RabbitMQ
    rabbit_host: str = "llm_port_api-rmq"
    rabbit_port: int = 5672
    rabbit_user: str = "guest"
    rabbit_pass: str = "guest"
    rabbit_vhost: str = "/"

    rabbit_pool_size: int = 2
    rabbit_channel_pool_size: int = 10

    # This variable is used to define
    # multiproc_dir. It's required for [uvi|guni]corn projects.
    prometheus_dir: Path = TEMP_DIR / "prom"

    # Sentry's configuration.
    sentry_dsn: Optional[str] = None
    sentry_sample_rate: float = 1.0

    # Grpc endpoint for opentelemetry.
    # E.G. http://localhost:4317
    opentelemetry_endpoint: Optional[str] = None

    # JWT settings (compatible with backend-issued tokens)
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_audience: str = ""
    jwt_issuer: str = ""

    # Gateway behavior
    http_timeout_sec: float = 30.0
    lease_ttl_sec: int = 90
    retry_pre_first_token: int = 1
    request_max_body_bytes: int = 2 * 1024 * 1024
    stream_idle_timeout_sec: float = 60.0

    # Langfuse observability
    langfuse_enabled: bool = False
    langfuse_host: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_tracing_enabled: bool = True
    langfuse_release: str | None = None

    # Optional module: PII service
    pii_service_url: str | None = None
    pii_enabled: bool = False

    # Optional module: External auth service
    auth_service_url: str | None = None
    auth_enabled: bool = False

    # Optional module: RAG engine (exposed via backend, status tracked here)
    rag_service_url: str | None = None
    rag_enabled: bool = False

    langfuse_debug: bool = False

    @property
    def db_url(self) -> URL:
        """
        Assemble database URL from settings.

        :return: database URL.
        """
        if self.db_url_override:
            return URL(self.db_url_override)
        return URL.build(
            scheme="postgresql+asyncpg",
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_pass,
            path=f"/{self.db_base}",
        )

    @property
    def redis_url(self) -> URL:
        """
        Assemble REDIS URL from settings.

        :return: redis URL.
        """
        path = ""
        if self.redis_base is not None:
            path = f"/{self.redis_base}"
        return URL.build(
            scheme="redis",
            host=self.redis_host,
            port=self.redis_port,
            user=self.redis_user,
            password=self.redis_pass,
            path=path,
        )

    @property
    def rabbit_url(self) -> URL:
        """
        Assemble RabbitMQ URL from settings.

        :return: rabbit URL.
        """
        return URL.build(
            scheme="amqp",
            host=self.rabbit_host,
            port=self.rabbit_port,
            user=self.rabbit_user,
            password=self.rabbit_pass,
            path=self.rabbit_vhost,
        )

    model_config = SettingsConfigDict(
        env_file=(".env", "llm_port_api/.env"),
        env_prefix="LLM_PORT_API_",
        env_file_encoding="utf-8",
    )


settings = Settings()
