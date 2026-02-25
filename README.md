# llm_port_api

OpenAI-compatible LLM gateway service.

## MVP Endpoints
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`

Internal template/debug endpoints remain under `/api/*`.

## Quick Start
```bash
poetry install
poetry run python -m llm_port_api
```

Swagger UI:
- `/api/docs`

## Docker Compose (Shared Stack)
From `llm_port_shared/`:
```bash
# Normal containerized run
docker compose up -d llm-port-api

# Dev mode with bind mount + reload (no image rebuild per code change)
docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up -d llm-port-api
```

## Environment Variables
Core variables:
- `LLM_PORT_API_HOST`
- `LLM_PORT_API_PORT`
- `LLM_PORT_API_RELOAD`
- `LLM_PORT_API_ENVIRONMENT`

Database:
- `LLM_PORT_API_DB_HOST`
- `LLM_PORT_API_DB_PORT`
- `LLM_PORT_API_DB_USER`
- `LLM_PORT_API_DB_PASS`
- `LLM_PORT_API_DB_BASE` (default `llm_api`)
- `LLM_PORT_API_DB_ECHO`

JWT:
- `LLM_PORT_API_JWT_SECRET`
- `LLM_PORT_API_JWT_ALGORITHM` (default `HS256`)

Gateway behavior:
- `LLM_PORT_API_HTTP_TIMEOUT_SEC`
- `LLM_PORT_API_LEASE_TTL_SEC`
- `LLM_PORT_API_RETRY_PRE_FIRST_TOKEN`
- `LLM_PORT_API_REQUEST_MAX_BODY_BYTES`
- `LLM_PORT_API_STREAM_IDLE_TIMEOUT_SEC`

Langfuse:
- `LLM_PORT_API_LANGFUSE_ENABLED`
- `LLM_PORT_API_LANGFUSE_HOST`
- `LLM_PORT_API_LANGFUSE_PUBLIC_KEY`
- `LLM_PORT_API_LANGFUSE_SECRET_KEY`
- `LLM_PORT_API_LANGFUSE_TRACING_ENABLED`
- `LLM_PORT_API_LANGFUSE_RELEASE`
- `LLM_PORT_API_LANGFUSE_DEBUG`

Redis/Rabbit:
- `LLM_PORT_API_REDIS_HOST`
- `LLM_PORT_API_REDIS_PORT`
- `LLM_PORT_API_REDIS_USER`
- `LLM_PORT_API_REDIS_PASS`
- `LLM_PORT_API_REDIS_BASE`
- `LLM_PORT_API_RABBIT_HOST`
- `LLM_PORT_API_RABBIT_PORT`
- `LLM_PORT_API_RABBIT_USER`
- `LLM_PORT_API_RABBIT_PASS`
- `LLM_PORT_API_RABBIT_VHOST`

## Migrations
```bash
alembic upgrade head
```

Rollback:
```bash
alembic downgrade base
```

## Curl Smoke Checks
```bash
# models
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer $TOKEN"

# chat (non-stream)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}]}'

# chat (stream)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -N \
  -d '{"model":"qwen3-32b","stream":true,"messages":[{"role":"user","content":"Hello"}]}'

# embeddings
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-32b","input":"The quick brown fox"}'
```

## Quality Gates
```bash
poetry run ruff check .
poetry run mypy llm_port_api
poetry run pytest -vv
```

## Additional Docs
- `docs/api-sequence.md`
- `docs/gateway-architecture.md`
- `docs/database.md`
