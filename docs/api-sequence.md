# API Sequence Diagrams

This document describes request/response flow for the OpenAI-compatible gateway endpoints.

## `/v1/models`

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as llm_port_api
    participant AUTH as Auth
    participant DAO as GatewayDAO (Postgres)

    C->>API: GET /v1/models + Bearer JWT
    API->>AUTH: verify token (sub, tenant_id)
    AUTH-->>API: AuthContext
    API->>DAO: list_enabled_aliases_for_tenant(tenant_id)
    DAO-->>API: aliases[]
    API-->>C: 200 ListModelsResponse
```

## `/v1/chat/completions` (non-stream)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as llm_port_api
    participant AUTH as Auth
    participant DAO as GatewayDAO (Postgres)
    participant RL as RateLimiter (Redis)
    participant RT as RouterService
    participant LEASE as LeaseManager (Redis)
    participant UP as Upstream Provider
    participant LF as Langfuse
    participant AUDIT as AuditService (Postgres)

    C->>API: POST /v1/chat/completions (stream=false)
    API->>AUTH: verify JWT + claims
    AUTH-->>API: AuthContext(sub, tenant_id)
    API->>DAO: get_tenant_policy(tenant_id)
    DAO-->>API: policy
    API->>RL: check RPM/TPM
    RL-->>API: allowed
    API->>RT: resolve alias + choose candidate
    RT->>LEASE: acquire(instance_id, request_id)
    LEASE-->>RT: acquired
    RT-->>API: routing decision
    API->>LF: start trace/generation
    API->>UP: proxy JSON request
    alt pre-first-token upstream failure
        API->>UP: retry once (MVP)
    end
    UP-->>API: OpenAI-compatible JSON
    API->>LF: finalize success/failure
    API->>LEASE: release in finally
    API->>AUDIT: write request log
    API-->>C: upstream status + JSON body
```

## `/v1/chat/completions` (stream)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as llm_port_api
    participant AUTH as Auth
    participant DAO as GatewayDAO (Postgres)
    participant RL as RateLimiter (Redis)
    participant RT as RouterService
    participant LEASE as LeaseManager (Redis)
    participant UP as Upstream Provider
    participant STREAM as SSE Wrapper
    participant LF as Langfuse
    participant AUDIT as AuditService (Postgres)

    C->>API: POST /v1/chat/completions (stream=true)
    API->>AUTH: verify JWT + claims
    AUTH-->>API: AuthContext
    API->>DAO: get_tenant_policy
    API->>RL: check RPM/TPM
    API->>RT: resolve alias + candidate
    RT->>LEASE: acquire
    LEASE-->>RT: acquired
    API->>LF: start trace/generation
    API->>UP: open upstream SSE stream
    API->>STREAM: wrap stream (TTFT/usage extraction)
    loop for each SSE chunk
        UP-->>STREAM: data: {...}
        STREAM-->>C: passthrough chunk
    end
    UP-->>STREAM: data: [DONE]
    STREAM-->>C: data: [DONE]
    API->>LF: finalize stream status/ttft/usage
    API->>LEASE: release in finally
    API->>AUDIT: write request log (latency/ttft/usage)
```

## `/v1/embeddings`

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant API as llm_port_api
    participant AUTH as Auth
    participant DAO as GatewayDAO (Postgres)
    participant RL as RateLimiter (Redis)
    participant RT as RouterService
    participant LEASE as LeaseManager (Redis)
    participant UP as Upstream Provider
    participant LF as Langfuse
    participant AUDIT as AuditService (Postgres)

    C->>API: POST /v1/embeddings
    API->>AUTH: verify JWT + claims
    API->>DAO: get_tenant_policy
    API->>RL: check RPM/TPM
    API->>RT: resolve alias + candidate
    RT->>LEASE: acquire
    API->>LF: start trace/generation
    API->>UP: proxy embeddings request
    UP-->>API: embeddings response
    API->>LF: finalize success/failure
    API->>LEASE: release in finally
    API->>AUDIT: write request log
    API-->>C: upstream status + embeddings JSON
```

## Error Envelope

All endpoint failures are returned in OpenAI-compatible shape:

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Human readable message",
    "param": null,
    "code": "machine_readable_code"
  }
}
```

## Admin Graph Consumption

`llm_port_backend` consumes `llm_gateway_request_log` from the `llm_api` database to power `/api/llm/graph/traces` and `/api/llm/graph/traces/stream`.
The frontend visualizer (`/admin/llm/graph`) subscribes to backend SSE only; it never calls Langfuse directly.
