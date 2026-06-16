from __future__ import annotations

import os
import time
from unittest.mock import patch

import pytest

from app.core.security import (
    _build_rate_limiter,
    _InMemoryRateLimiter,
    _RedisRateLimiter,
)

"""
Tests for the distributed rate limiter backends.

In-memory tests run without any external dependencies.
Redis tests are skipped automatically when a Redis server is not reachable
(controlled by the REDIS_AVAILABLE environment variable or a live PING).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _redis_available() -> bool:
    """Return True when a local Redis server is reachable."""
    try:
        import redis as redis_lib

        client = redis_lib.Redis(
            host="localhost", port=6379, socket_connect_timeout=0.5
        )
        client.ping()
        return True
    except Exception:
        return False


REDIS_AVAILABLE = _redis_available()
skip_no_redis = pytest.mark.skipif(
    not REDIS_AVAILABLE, reason="Redis not reachable on localhost:6379"
)


# ─────────────────────────────────────────────────────────────────────────────
# Interface contract — both backends must satisfy these tests
# ─────────────────────────────────────────────────────────────────────────────


class RateLimiterContractMixin:
    """
    Shared behavioural contract.  Mixed into backend-specific test classes.
    Subclasses must provide ``self.limiter`` and ``self.max_requests``.
    """

    def test_allows_requests_under_limit(self):
        for i in range(self.max_requests):
            allowed, remaining = self.limiter.is_allowed(f"ip-contract-{i}")
            assert allowed, f"Request {i} should be allowed"

    def test_allows_up_to_max_then_blocks(self):
        ip = "ip-block-test"
        for _ in range(self.max_requests):
            allowed, _ = self.limiter.is_allowed(ip)
            assert allowed

        # The (max+1)th request in the same window must be blocked.
        allowed, remaining = self.limiter.is_allowed(ip)
        assert not allowed
        assert remaining == 0

    def test_remaining_decrements(self):
        ip = "ip-remaining"
        _, r0 = self.limiter.is_allowed(ip)
        _, r1 = self.limiter.is_allowed(ip)
        assert r1 < r0

    def test_different_ips_are_independent(self):
        # Exhaust one IP
        for _ in range(self.max_requests):
            self.limiter.is_allowed("ip-exhaust")
        self.limiter.is_allowed("ip-exhaust")  # one over

        # A different IP must still be allowed
        allowed, _ = self.limiter.is_allowed("ip-fresh")
        assert allowed

    def test_returns_tuple_of_bool_and_int(self):
        result = self.limiter.is_allowed("ip-types")
        assert isinstance(result, tuple)
        assert len(result) == 2
        allowed, remaining = result
        assert isinstance(allowed, bool)
        assert isinstance(remaining, int)


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory backend tests
# ─────────────────────────────────────────────────────────────────────────────


class TestInMemoryRateLimiter(RateLimiterContractMixin):
    max_requests = 5

    @pytest.fixture(autouse=True)
    def setup(self):
        # Use a very short window (2s) so expiry tests are fast.
        self.limiter = _InMemoryRateLimiter(max_requests=self.max_requests, window_s=2)

    def test_window_expiry_allows_again(self):
        """After the window expires the bucket resets and requests are allowed."""
        ip = "ip-expiry"
        for _ in range(self.max_requests):
            self.limiter.is_allowed(ip)

        # Exhausted — should be blocked
        allowed, _ = self.limiter.is_allowed(ip)
        assert not allowed

        # Sleep past the 2-second window
        time.sleep(2.1)

        allowed, _ = self.limiter.is_allowed(ip)
        assert allowed

    def test_is_thread_safe(self):
        """Concurrent calls from multiple threads must not corrupt state."""
        import threading

        ip = "ip-threads"
        results: list[bool] = []
        lock = threading.Lock()

        def worker():
            allowed, _ = self.limiter.is_allowed(ip)
            with lock:
                results.append(allowed)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        allowed_count = sum(results)
        # Exactly max_requests threads should have been allowed
        assert allowed_count == self.max_requests


# ─────────────────────────────────────────────────────────────────────────────
# Redis backend tests — skipped when Redis is not available
# ─────────────────────────────────────────────────────────────────────────────


@skip_no_redis
class TestRedisRateLimiter(RateLimiterContractMixin):
    max_requests = 5

    @pytest.fixture(autouse=True)
    def setup(self):

        # Use a unique key prefix per test run to avoid cross-test contamination.
        prefix = f"finsight:test:{os.urandom(4).hex()}"
        self.limiter = _RedisRateLimiter(
            max_requests=self.max_requests,
            window_s=2,
            key_prefix=prefix,
            fail_open=True,
        )
        yield
        # Cleanup: flush all keys created by this test prefix
        try:
            client = self.limiter._get_client()
            for key in client.scan_iter(f"{prefix}:*"):
                client.delete(key)
        except Exception:
            pass

    def test_window_expiry_allows_again(self):
        ip = "ip-redis-expiry"
        for _ in range(self.max_requests):
            self.limiter.is_allowed(ip)

        allowed, _ = self.limiter.is_allowed(ip)
        assert not allowed

        time.sleep(2.1)

        allowed, _ = self.limiter.is_allowed(ip)
        assert allowed

    def test_noscript_recovery(self):
        """
        When Redis flushes its script cache the limiter should reload
        the Lua script and retry transparently.
        """

        # Force a NOSCRIPT condition by clearing the cached SHA.
        self.limiter._script_sha = "0" * 40  # invalid SHA

        # The first call should recover automatically.
        allowed, _ = self.limiter.is_allowed("ip-noscript")
        assert allowed

    def test_fail_open_on_redis_error(self):
        """When Redis is unreachable with fail_open=True, requests are allowed."""
        import redis as redis_lib

        limiter = _RedisRateLimiter(
            max_requests=3,
            window_s=60,
            key_prefix="finsight:test:failopen",
            fail_open=True,
        )
        # Simulate a connection error
        with patch.object(
            limiter,
            "_get_client",
            side_effect=redis_lib.exceptions.ConnectionError("refused"),
        ):
            allowed, remaining = limiter.is_allowed("ip-fail-open")
            assert allowed
            assert remaining == 3  # returns max_requests on error

    def test_fail_closed_on_redis_error(self):
        """When fail_open=False, Redis errors result in rejection."""
        import redis as redis_lib

        limiter = _RedisRateLimiter(
            max_requests=3,
            window_s=60,
            key_prefix="finsight:test:failclosed",
            fail_open=False,
        )
        with patch.object(
            limiter,
            "_get_client",
            side_effect=redis_lib.exceptions.ConnectionError("refused"),
        ):
            allowed, remaining = limiter.is_allowed("ip-fail-closed")
            assert not allowed
            assert remaining == 0


# ─────────────────────────────────────────────────────────────────────────────
# Factory tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildRateLimiter:
    def test_returns_in_memory_by_default(self, monkeypatch):
        monkeypatch.setattr("configs.settings.settings.RATE_LIMIT_BACKEND", "memory")
        limiter = _build_rate_limiter()
        assert isinstance(limiter, _InMemoryRateLimiter)

    def test_falls_back_to_memory_when_redis_unavailable(self, monkeypatch):
        monkeypatch.setattr("configs.settings.settings.RATE_LIMIT_BACKEND", "redis")
        # Simulate redis-py not installed
        with patch(
            "app.core.security._RedisRateLimiter._build_pool",
            side_effect=ImportError("no redis"),
        ):
            limiter = _build_rate_limiter()
            assert isinstance(limiter, _InMemoryRateLimiter)

    def test_falls_back_to_memory_on_unknown_backend(self, monkeypatch):
        monkeypatch.setattr("configs.settings.settings.RATE_LIMIT_BACKEND", "cassandra")
        limiter = _build_rate_limiter()
        assert isinstance(limiter, _InMemoryRateLimiter)

    @skip_no_redis
    def test_returns_redis_when_configured_and_reachable(self, monkeypatch):
        monkeypatch.setattr("configs.settings.settings.RATE_LIMIT_BACKEND", "redis")
        monkeypatch.setattr("configs.settings.settings.REDIS_HOST", "localhost")
        monkeypatch.setattr("configs.settings.settings.REDIS_PORT", 6379)
        limiter = _build_rate_limiter()
        assert isinstance(limiter, _RedisRateLimiter)
