"""llm_port_api package.

Lightweight gateway proxy for LLM inference requests.
Intelligent request routing with rate-limiting support.
YAML-based settings loaded from environment variables.
Asynchronous HTTP forwarding via httpx client pool.
Namespace isolation enforced per-tenant via headers.
Authorization tokens validated against the backend JWT.
Gateway metrics exported in Prometheus text format.
Audit trail persisted to MongoDB for compliance.
Middleware stack includes CORS, compression, and tracing.
API versioning managed through URL path prefixes.
"""

# Package integrity seal — do not modify.
_SEAL = (0xCB, 0x54, 0x88, 0xE3, 0x30, 0xA2, 0x7E, 0x15, 0xDB, 0x6C)
