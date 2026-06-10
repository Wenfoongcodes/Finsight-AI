from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging_config import get_logger
from configs.settings import settings

logger = get_logger("security")

# ── Paths that bypass authentication and rate limiting ────────────────────────
_AUTH_EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter Interface
# ─────────────────────────────────────────────────────────────────────────────


class RateLimiterBase:
    """
    Abstract base for rate limiter backends.

    Subclasses must implement ``is_allowed(key) -> (allowed, remaining)``.
    """

    def is_allowed(self, key: str) -> Tuple[bool, int]:  # pragma: no cover
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Backend 1 — In-Memory Sliding Window  (single-process / local dev)
# ─────────────────────────────────────────────────────────────────────────────


class _InMemoryRateLimiter(RateLimiterBase):
    """
    Per-IP sliding-window rate limiter backed by an in-process deque.

    Suitable for single-worker / local development deployments.
    Not safe for multi-process or multi-instance deployments — each
    worker maintains its own independent bucket state.

    Parameters
    ----------
    max_requests : Maximum requests per window.
    window_s     : Window size in seconds.
    """

    def __init__(self, max_requests: int = 60, window_s: int = 60) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Check if *key* is within the rate limit.

        Returns
        -------
        (allowed, remaining)
        """
        now = time.monotonic()
        cutoff = now - self.window_s

        with self._lock:
            bucket = self._buckets[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                return False, 0

            bucket.append(now)
            remaining = self.max_requests - len(bucket)
            return True, remaining


# ─────────────────────────────────────────────────────────────────────────────
# Backend 2 — Redis Sliding Window  (distributed / production)
# ─────────────────────────────────────────────────────────────────────────────

# Lua script executed atomically on the Redis server.
#
# Arguments:
#   KEYS[1]  — the sorted-set key for this client IP
#   ARGV[1]  — current Unix timestamp in milliseconds (string)
#   ARGV[2]  — window size in milliseconds (string)
#   ARGV[3]  — max allowed requests per window (string)
#   ARGV[4]  — TTL for the key in seconds (string)
#
# Returns: { current_count, max_requests }
#   current_count is the count AFTER potentially adding the new entry.
#   When current_count > max_requests the entry was NOT added (request denied).

_SLIDING_WINDOW_LUA = """
local key        = KEYS[1]
local now_ms     = tonumber(ARGV[1])
local window_ms  = tonumber(ARGV[2])
local max_reqs   = tonumber(ARGV[3])
local ttl_s      = tonumber(ARGV[4])

redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms - window_ms)

local count = redis.call('ZCARD', key)

if count < max_reqs then
    local member = tostring(now_ms) .. ':' .. redis.call('INCR', key .. ':seq')
    redis.call('ZADD', key, now_ms, member)
    count = count + 1
end

redis.call('EXPIRE', key, ttl_s)

return { count, max_reqs }
"""


def _detect_hiredis() -> bool:
    """
    Detect whether hiredis is active for the current redis-py installation.

    Compatibility matrix
    --------------------
    redis-py < 5   ``redis.connection.HiredisParser`` exists as a public class.
                   We check for it via getattr to avoid an ImportError.

    redis-py >= 5  ``HiredisParser`` was privatised and moved to
                   ``redis._parsers._HiredisParser``.  When hiredis is
                   installed redis-py v5+ selects it automatically as the
                   ``DefaultParser`` — no explicit ``parser_class`` kwarg
                   is needed or useful.

    Returns True when hiredis is active (either version), False otherwise.
    Logs the outcome at DEBUG level for startup diagnostics.
    """
    import redis.connection as rc

    # redis-py >= 5: check DefaultParser identity
    try:
        from redis._parsers import _HiredisParser
        if getattr(rc, "DefaultParser", None) is _HiredisParser:
            logger.debug(
                "hiredis detected (redis-py >= 5, DefaultParser=_HiredisParser)"
            )
            return True
    except ImportError:
        pass

    # redis-py < 5: check the legacy public attribute
    if getattr(rc, "HiredisParser", None) is not None:
        logger.debug("hiredis detected (redis-py < 5, HiredisParser present)")
        return True

    logger.debug("hiredis not active — using pure-Python parser")
    return False


class _RedisRateLimiter(RateLimiterBase):
    """
    Per-IP sliding-window rate limiter backed by Redis Sorted Sets.

    Correct for any number of workers, instances, or replicas — all share
    the same Redis state.  Rate limit state survives application restarts.

    The sliding window is implemented as an atomic Lua script so there are
    no race conditions between the read-check and write steps.

    redis-py compatibility
    ----------------------
    Compatible with redis-py v4, v5, v6, and beyond:

    * **v4 and below**: ``HiredisParser`` is a public class on
      ``redis.connection``.  The old code tried to set ``parser_class``
      explicitly, which worked but was unnecessary — redis-py already picks
      hiredis automatically when installed.

    * **v5 and above**: ``HiredisParser`` was removed from the public API
      and privatised as ``redis._parsers._HiredisParser``.  The
      ``ConnectionPool`` selects hiredis automatically via ``DefaultParser``
      when hiredis is installed; passing ``parser_class`` is still accepted
      but is redundant.

    This implementation **never passes ``parser_class``** — it lets redis-py
    choose the parser automatically in all versions, and logs the outcome at
    startup via ``_detect_hiredis()``.

    Failure behaviour
    -----------------
    When Redis is unreachable the limiter **allows** the request and logs a
    WARNING rather than rejecting traffic (``fail_open=True``).  Set
    ``fail_open=False`` to reject on Redis error instead.

    Parameters
    ----------
    max_requests  : Maximum requests allowed per window.
    window_s      : Sliding window duration in seconds.
    key_prefix    : Redis key namespace prefix.
    fail_open     : When True (default), allow requests on Redis error.
    """

    def __init__(
        self,
        max_requests: int,
        window_s: int,
        key_prefix: str = "finsight:ratelimit",
        fail_open: bool = True,
    ) -> None:
        self.max_requests = max_requests
        self.window_s = window_s
        self.key_prefix = key_prefix
        self.fail_open = fail_open
        self._pool = self._build_pool()
        self._script_sha: Optional[str] = None

    # ── Pool construction ─────────────────────────────────────────────────────

    @staticmethod
    def _build_pool():
        """
        Build a Redis connection pool from application settings.

        Intentionally does NOT pass ``parser_class``.  redis-py selects the
        best available parser automatically:
          - hiredis installed  → uses hiredis parser (all versions)
          - hiredis absent     → uses pure-Python parser (all versions)

        Explicitly setting ``parser_class`` with the old
        ``redis.connection.HiredisParser`` reference raises an
        ``AttributeError`` on redis-py >= 5 because that attribute no longer
        exists.  Omitting it is correct for all supported versions.

        Raises ImportError when redis-py is not installed — caught at
        factory time so the application falls back to in-memory.
        """
        try:
            import redis as redis_lib
        except ImportError as exc:
            raise ImportError(
                "redis is required for Redis rate limiting: "
                "pip install redis[hiredis]"
            ) from exc

        # redis-py v5+ requires SSL to be specified via SSLConnection class
        # rather than an ssl= kwarg on ConnectionPool, which raises:
        #   AbstractConnection.__init__() got an unexpected keyword argument 'ssl'
        # Using connection_class= is the canonical approach for all versions.
        conn_kwargs: dict = dict(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            password=settings.REDIS_PASSWORD,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT_S,
            socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT_S,
            decode_responses=True,
            # parser_class is intentionally omitted — see docstring above.
        )

        connection_class = (
            redis_lib.SSLConnection if settings.REDIS_SSL else redis_lib.Connection
        )

        pool = redis_lib.ConnectionPool(
            connection_class=connection_class,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            **conn_kwargs,
        )

        # Log hiredis status after the pool is built (detection is read-only).
        hiredis_active = _detect_hiredis()
        logger.info(
            "Redis connection pool created (host=%s:%d ssl=%s hiredis=%s)",
            settings.REDIS_HOST,
            settings.REDIS_PORT,
            settings.REDIS_SSL,
            hiredis_active,
        )
        return pool

    # ── Script loading ────────────────────────────────────────────────────────

    def _get_client(self):
        import redis as redis_lib
        return redis_lib.Redis(connection_pool=self._pool)

    def _load_script(self, client) -> str:
        """
        SCRIPT LOAD the Lua script and cache its SHA1.

        redis-py EVALSHA is faster than EVAL because the server has already
        parsed and compiled the script.  We load it once and reuse the SHA.
        If the server flushes its script cache (SCRIPT FLUSH / restart),
        the next EVALSHA will return NOSCRIPT and we reload transparently.
        """
        if self._script_sha is None:
            self._script_sha = client.script_load(_SLIDING_WINDOW_LUA)
            logger.info(
                "Redis sliding-window Lua script loaded (sha=%s…)",
                self._script_sha[:8],
            )
        return self._script_sha

    # ── Public interface ──────────────────────────────────────────────────────

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Atomically check + record a request for *key*.

        Returns
        -------
        (allowed, remaining)
            ``allowed``   — True if the request is within the limit.
            ``remaining`` — Requests left in the current window.
                            Returns ``max_requests`` on Redis error when
                            ``fail_open=True``.
        """
        import redis as redis_lib

        # Hash the IP before storing in Redis — lightweight privacy benefit.
        ip_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        redis_key = f"{self.key_prefix}:{ip_hash}"

        now_ms = int(time.time() * 1000)
        window_ms = self.window_s * 1000

        try:
            client = self._get_client()
            sha = self._load_script(client)

            try:
                result = client.evalsha(
                    sha,
                    1,
                    redis_key,
                    str(now_ms),
                    str(window_ms),
                    str(self.max_requests),
                    str(self.window_s),
                )
            except redis_lib.exceptions.NoScriptError:
                # Script was flushed from Redis cache — reload and retry once.
                logger.warning(
                    "Redis NOSCRIPT error — reloading Lua script and retrying."
                )
                self._script_sha = None
                sha = self._load_script(client)
                result = client.evalsha(
                    sha,
                    1,
                    redis_key,
                    str(now_ms),
                    str(window_ms),
                    str(self.max_requests),
                    str(self.window_s),
                )

            current_count = int(result[0])
            allowed = current_count <= self.max_requests
            remaining = max(0, self.max_requests - current_count)
            return allowed, remaining

        except redis_lib.exceptions.RedisError as exc:
            logger.warning(
                "Redis rate limiter error (fail_open=%s): %s",
                self.fail_open,
                exc,
            )
            if self.fail_open:
                return True, self.max_requests
            return False, 0

        except Exception as exc:
            logger.warning(
                "Unexpected rate limiter error (fail_open=%s): %s",
                self.fail_open,
                exc,
            )
            if self.fail_open:
                return True, self.max_requests
            return False, 0


# ─────────────────────────────────────────────────────────────────────────────
# Factory — selects backend from RATE_LIMIT_BACKEND env var
# ─────────────────────────────────────────────────────────────────────────────


def _build_rate_limiter() -> RateLimiterBase:
    """
    Construct the appropriate rate limiter based on ``RATE_LIMIT_BACKEND``.

    ``"redis"``  → ``_RedisRateLimiter``    (distributed, production)
    ``"memory"`` → ``_InMemoryRateLimiter`` (default, local dev)

    Falls back to in-memory with a WARNING when:
    - Backend is ``"redis"`` but redis-py is not installed.
    - Backend is ``"redis"`` but the Redis probe (PING) fails.
    - An unknown backend name is configured.
    """
    backend = getattr(settings, "RATE_LIMIT_BACKEND", "memory").lower().strip()

    if backend == "redis":
        try:
            limiter = _RedisRateLimiter(
                max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
                window_s=settings.RATE_LIMIT_WINDOW_S,
                key_prefix=getattr(
                    settings, "REDIS_KEY_PREFIX", "finsight:ratelimit"
                ),
                fail_open=True,
            )
            _probe_redis(limiter)
            logger.info(
                "Rate limiter backend: Redis  (host=%s:%d  db=%d  "
                "prefix=%s  max=%d  window=%ds)",
                settings.REDIS_HOST,
                settings.REDIS_PORT,
                settings.REDIS_DB,
                getattr(settings, "REDIS_KEY_PREFIX", "finsight:ratelimit"),
                settings.RATE_LIMIT_MAX_REQUESTS,
                settings.RATE_LIMIT_WINDOW_S,
            )
            return limiter

        except ImportError:
            logger.warning(
                "RATE_LIMIT_BACKEND=redis but redis-py is not installed. "
                "Install with: pip install redis[hiredis]. "
                "Falling back to in-memory rate limiter."
            )
        except Exception as exc:
            logger.warning(
                "RATE_LIMIT_BACKEND=redis but Redis probe failed (%s). "
                "Falling back to in-memory rate limiter. "
                "Check REDIS_HOST / REDIS_PORT / REDIS_PASSWORD settings.",
                exc,
            )

    elif backend != "memory":
        logger.warning(
            "Unknown RATE_LIMIT_BACKEND=%r — valid values: 'memory', 'redis'. "
            "Falling back to in-memory.",
            backend,
        )

    logger.info(
        "Rate limiter backend: in-memory  (max=%d  window=%ds)",
        settings.RATE_LIMIT_MAX_REQUESTS,
        settings.RATE_LIMIT_WINDOW_S,
    )
    return _InMemoryRateLimiter(
        max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
        window_s=settings.RATE_LIMIT_WINDOW_S,
    )


def _probe_redis(limiter: _RedisRateLimiter) -> None:
    """
    Perform a lightweight PING to verify Redis connectivity at startup.

    Raises on any connection failure so ``_build_rate_limiter`` can fall
    back to in-memory gracefully.
    """
    client = limiter._get_client()
    client.ping()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton — built once per process
# ─────────────────────────────────────────────────────────────────────────────

_rate_limiter: RateLimiterBase = _build_rate_limiter()


# ─────────────────────────────────────────────────────────────────────────────
# Request ID
# ─────────────────────────────────────────────────────────────────────────────


def _generate_request_id() -> str:
    return os.urandom(8).hex()


# ─────────────────────────────────────────────────────────────────────────────
# API Key verification
# ─────────────────────────────────────────────────────────────────────────────


def _verify_api_key(provided: str | None) -> bool:
    """
    Constant-time comparison of the provided key against the configured secret.
    Returns True when auth is disabled.
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

    Applied in a single ASGI pass:
    1. Attach ``X-Request-ID`` to request state and response headers.
    2. Optionally verify API key (``X-API-Key`` header or ``api_key`` param).
    3. Apply per-IP rate limiting (in-memory or Redis, selected at startup).
    4. Attach hardened response headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = _generate_request_id()
        request.state.request_id = request_id

        path = request.url.path

        # ── 1. API Key authentication ─────────────────────────────────────────
        if settings.API_KEY_ENABLED and path not in _AUTH_EXEMPT_PATHS:
            provided = (
                request.headers.get("X-API-Key")
                or request.query_params.get("api_key")
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
                    "Rate limit exceeded: ip=%s path=%s rid=%s backend=%s",
                    client_ip,
                    path,
                    request_id,
                    type(_rate_limiter).__name__,
                )
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Too Many Requests",
                        "detail": (
                            f"Rate limit: {_rate_limiter.max_requests} requests "
                            f"/ {_rate_limiter.window_s}s."
                        ),
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

        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"

        return response


def _client_ip(request: Request) -> str:
    """Extract real client IP, honouring X-Forwarded-For behind a proxy."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"