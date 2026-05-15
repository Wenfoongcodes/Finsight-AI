"""
FinSight AI — Security Middleware & Auth
=========================================

Provides:
1. Optional API-key authentication via X-API-Key header or ?api_key= query param.
   Controlled by ``settings.API_KEY_ENABLED``.  When disabled (default) the
   middleware is a no-op so local / development workflows are unaffected.

2. Request-ID injection — every response carries an ``X-Request-ID`` header for
   end-to-end traceability in logs and dashboards.

3. Security response headers — a minimal hardened set appropriate for a JSON API
   (not a browser app):
       X-Content-Type-Options: nosniff
       X-Frame-Options: DENY
       Referrer-Policy: no-referrer
       Cache-Control: no-store  (prevents caching of prediction responses)

Design rationale
----------------
*Why not OAuth2 / JWT?*
OAuth2 is the right choice when multiple human users need scoped tokens and
refresh flows.  For a research API accessed by a dashboard or automated clients,
a static API-key is lower overhead, still secure, and trivially rotatable via
an environment variable — no token-issuance endpoint or key-derivation library
required.

*Why middleware rather than a FastAPI Depends() on each route?*
A middleware applied at the ASGI layer protects every route, including future
ones, without requiring every router author to remember to add a dependency.
The /health endpoint is exempted so load-balancer health-checks work without
credentials.

*Why not use slowapi here?*
Rate-limiting state (request counters) is per-process.  In a multi-worker
Uvicorn setup the counters are not shared.  A proper rate-limiter needs Redis
or a similar shared store.  We implement a lightweight token-bucket in the
middleware that is good enough for research APIs; a Redis-backed solution can
be substituted by replacing ``_InMemoryRateLimiter`` with a Redis adapter.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("security")

# ── Paths that bypass authentication ──────────────────────────────────────────
_AUTH_EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory token-bucket rate limiter
# ─────────────────────────────────────────────────────────────────────────────


class _InMemoryRateLimiter:
    """
    Per-IP sliding-window rate limiter.

    Parameters
    ----------
    max_requests : Maximum requests per window.
    window_s     : Window size in seconds.

    Thread-safe via a per-bucket ``threading.Lock``.  Not suitable for
    multi-process deployments — replace with a Redis adapter in that case.
    """

    def __init__(self, max_requests: int = 60, window_s: int = 60) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if *key* is within the rate limit.

        Returns
        -------
        (allowed, remaining)
            ``allowed`` — True when the request should be processed.
            ``remaining`` — Number of requests left in the current window.
        """
        now = time.monotonic()
        cutoff = now - self.window_s

        with self._lock:
            bucket = self._buckets[key]

            # Evict expired timestamps
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                return False, 0

            bucket.append(now)
            remaining = self.max_requests - len(bucket)
            return True, remaining


# Shared singleton — created once per process
_rate_limiter = _InMemoryRateLimiter(
    max_requests=getattr(settings, "RATE_LIMIT_MAX_REQUESTS", 120),
    window_s=getattr(settings, "RATE_LIMIT_WINDOW_S", 60),
)


# ─────────────────────────────────────────────────────────────────────────────
# Request ID
# ─────────────────────────────────────────────────────────────────────────────


def _generate_request_id() -> str:
    """Generate a short hex request identifier."""
    return os.urandom(8).hex()


# ─────────────────────────────────────────────────────────────────────────────
# API Key verification
# ─────────────────────────────────────────────────────────────────────────────


def _verify_api_key(provided: str | None) -> bool:
    """
    Constant-time comparison of the provided key against the configured secret.

    Uses ``hmac.compare_digest`` to prevent timing attacks.
    Returns ``True`` when auth is disabled (``API_KEY_ENABLED=false``).
    """
    if not settings.API_KEY_ENABLED:
        return True

    expected = settings.API_SECRET_KEY
    if not expected:
        logger.error(
            "API_KEY_ENABLED=true but API_SECRET_KEY is not set. "
            "All authenticated requests will be rejected."
        )
        return False

    if not provided:
        return False

    # Normalise both sides to bytes for compare_digest
    try:
        return hmac.compare_digest(
            provided.encode("utf-8"),
            expected.encode("utf-8"),
        )
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Composite security middleware.

    Applied in a single ASGI pass to minimise latency:
    1. Attach ``X-Request-ID`` to request state and response headers.
    2. Optionally verify API key (``X-API-Key`` header or ``api_key`` param).
    3. Apply per-IP rate limiting.
    4. Attach hardened response headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = _generate_request_id()
        request.state.request_id = request_id

        path = request.url.path

        # ── 1. API Key authentication ─────────────────────────────────────────
        if settings.API_KEY_ENABLED and path not in _AUTH_EXEMPT_PATHS:
            provided = request.headers.get("X-API-Key") or request.query_params.get(
                "api_key"
            )
            if not _verify_api_key(provided):
                logger.warning(
                    "Rejected unauthenticated request: path=%s ip=%s rid=%s",
                    path,
                    _client_ip(request),
                    request_id,
                )
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "Unauthorized",
                        "detail": "Invalid or missing API key.",
                    },
                    headers={"X-Request-ID": request_id},
                )

        # ── 2. Rate limiting ──────────────────────────────────────────────────
        if settings.RATE_LIMIT_ENABLED and path not in _AUTH_EXEMPT_PATHS:
            client_ip = _client_ip(request)
            allowed, remaining = _rate_limiter.is_allowed(client_ip)
            if not allowed:
                logger.warning(
                    "Rate limit exceeded: ip=%s path=%s rid=%s",
                    client_ip,
                    path,
                    request_id,
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Too Many Requests",
                        "detail": f"Rate limit: {_rate_limiter.max_requests} requests / {_rate_limiter.window_s}s.",
                    },
                    headers={
                        "X-Request-ID": request_id,
                        "Retry-After": str(_rate_limiter.window_s),
                    },
                )

        # ── 3. Process request ────────────────────────────────────────────────
        response: Response = await call_next(request)

        # ── 4. Attach security / tracing headers ──────────────────────────────
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"

        # Prevent prediction responses from being cached by proxies
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"

        return response


def _client_ip(request: Request) -> str:
    """
    Extract the real client IP, honouring ``X-Forwarded-For`` when the
    application is behind a reverse proxy (Nginx, Traefik, AWS ALB).
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For: client, proxy1, proxy2
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"
