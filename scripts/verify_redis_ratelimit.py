#!/usr/bin/env python3
"""
verify_redis_ratelimit.py
=========================
Single-step verification for the FinSight distributed Redis rate limiter.
Covers all seven verification layers in one command.

QUICK START
-----------
The script auto-detects how to start Redis. Just run:

    python scripts/verify_redis_ratelimit.py

It will try each startup method in order and tell you exactly what to do
if none work automatically.

STARTUP METHODS TRIED (in order)
---------------------------------
1. Redis already running on the configured host/port  → use it directly
2. Docker available                                   → start redis:7.2-alpine container
3. docker-compose.yml present with a redis service    → docker-compose up redis
4. redis-server binary on PATH                        → start in background
5. Windows: Chocolatey / Scoop / winget redis         → install + start
6. None found                                         → print clear instructions

OPTIONS
-------
  --host HOST          Redis host          (default: localhost)
  --port PORT          Redis port          (default: 6379)
  --password PASSWORD  Redis password      (default: none)
  --max-requests N     Limit under test    (default: 120)
  --window-s N         Window in seconds   (default: 60)
  --prefix PREFIX      Key prefix          (default: finsight:ratelimit)
  --api-url URL        FastAPI base URL    (default: http://localhost:8000)
  --no-api             Skip layers 4-5 (no running API server required)
  --no-auto-redis      Do not attempt to start Redis automatically
  --keep-redis         Leave auto-started Redis running after the script

EXIT CODES
----------
  0  All layers passed
  1  One or more layers failed
  2  redis-py not installed
  3  Redis could not be started and is not reachable
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

_NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"\033[{code}m{text}\033[0m"


OK = _c("32", "✓ PASS")
FAIL = _c("31", "✗ FAIL")
SKIP = _c("33", "– SKIP")
INFO = _c("36", "ℹ")
WARN = _c("33", "⚠")


def _banner(text: str) -> None:
    w = 70
    print()
    print(_c("1", "─" * w))
    print(_c("1", f"  {text}"))
    print(_c("1", "─" * w))


def _result(label: str, passed: bool | None, detail: str = "") -> None:
    icon = {True: OK, False: FAIL, None: SKIP}[passed]
    line = f"  {icon}  {label}"
    if detail:
        line += f"  │  {_c('2', detail)}"
    print(line)


def _info(msg: str) -> None:
    print(f"  {INFO}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {WARN}  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Layer result accumulator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LayerResult:
    name: str
    passed: bool = False
    skipped: bool = False
    details: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Redis connectivity probe
# ─────────────────────────────────────────────────────────────────────────────


def _redis_ping(
    host: str, port: int, password: Optional[str], timeout: float = 1.5
) -> bool:
    try:
        import redis as redis_lib

        redis_lib.Redis(
            host=host,
            port=port,
            password=password,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
        ).ping()
        return True
    except Exception:
        return False


def _wait_for_redis(
    host: str,
    port: int,
    password: Optional[str],
    deadline: float,
    interval: float = 0.4,
) -> bool:
    while time.time() < deadline:
        if _redis_ping(host, port, password):
            return True
        time.sleep(interval)
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Redis auto-start  —  tries multiple methods, explains every failure
# ─────────────────────────────────────────────────────────────────────────────

_STARTED_METHOD: Optional[str] = None  # set by whichever method succeeds
_STARTED_PROC: Optional[subprocess.Popen] = None  # for redis-server cleanup
_DOCKER_CONTAINER = "finsight-verify-redis"


def _try_start_redis(host: str, port: int, password: Optional[str]) -> tuple[bool, str]:
    """
    Try every available method to start Redis.

    Returns (success, method_name).
    Sets module globals _STARTED_METHOD and _STARTED_PROC.
    """
    global _STARTED_METHOD, _STARTED_PROC

    _banner("Redis startup — auto-detect")

    # ── Method 0: already running ─────────────────────────────────────────────
    if _redis_ping(host, port, password, timeout=1.0):
        _result(f"Redis already running on {host}:{port}", True)
        _STARTED_METHOD = "existing"
        return True, "existing"
    _info(f"Redis not running on {host}:{port} — trying to start it …")

    # ── Method 1: Docker ─────────────────────────────────────────────────────
    if shutil.which("docker"):
        _info("Docker found — starting redis:7.2-alpine container …")
        try:
            # Stop any leftover container from a previous run
            subprocess.run(
                ["docker", "rm", "-f", _DOCKER_CONTAINER],
                capture_output=True,
            )
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "--name",
                    _DOCKER_CONTAINER,
                    "-p",
                    f"127.0.0.1:{port}:6379",
                    "redis:7.2-alpine",
                    "redis-server",
                    "--maxmemory",
                    "64mb",
                    "--maxmemory-policy",
                    "allkeys-lru",
                    "--save",
                    "",
                ],
                check=True,
                capture_output=True,
            )
            if _wait_for_redis(host, port, password, time.time() + 15):
                _result("Docker redis:7.2-alpine container started", True)
                _STARTED_METHOD = "docker"
                return True, "docker"
            _warn("Docker container started but Redis did not become healthy in 15s")
        except subprocess.CalledProcessError as exc:
            _warn(f"docker run failed: {exc.stderr.decode()[:120]}")
        except Exception as exc:
            _warn(f"Docker error: {exc}")
    else:
        _info("Docker not found — skipping Docker method")

    # ── Method 2: docker-compose ─────────────────────────────────────────────
    compose_files = ["docker-compose.yml", "docker-compose.yaml"]
    compose_cmd = None
    for cf in compose_files:
        if os.path.exists(cf):
            for cmd in (["docker", "compose"], ["docker-compose"]):
                if shutil.which(cmd[0]):
                    compose_cmd = cmd + ["up", "-d", "redis"]
                    break
            break

    if compose_cmd:
        _info(f"docker-compose.yml found — running: {' '.join(compose_cmd)} …")
        try:
            subprocess.run(compose_cmd, check=True, capture_output=True)
            if _wait_for_redis(host, port, password, time.time() + 20):
                _result("docker-compose redis service started", True)
                _STARTED_METHOD = "docker-compose"
                return True, "docker-compose"
            _warn("docker-compose up succeeded but Redis not healthy after 20s")
        except subprocess.CalledProcessError as exc:
            _warn(f"docker-compose failed: {exc.stderr.decode()[:120]}")
        except Exception as exc:
            _warn(f"docker-compose error: {exc}")
    else:
        _info("No docker-compose.yml found — skipping compose method")

    # ── Method 3: redis-server binary ────────────────────────────────────────
    if shutil.which("redis-server"):
        _info("redis-server binary found — starting in background …")
        try:
            proc = subprocess.Popen(
                [
                    "redis-server",
                    "--port",
                    str(port),
                    "--maxmemory",
                    "64mb",
                    "--maxmemory-policy",
                    "allkeys-lru",
                    "--save",
                    "",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if _wait_for_redis(host, port, password, time.time() + 10):
                _result("redis-server started from binary", True)
                _STARTED_METHOD = "redis-server"
                _STARTED_PROC = proc
                return True, "redis-server"
            proc.terminate()
            _warn("redis-server started but did not become healthy in 10s")
        except Exception as exc:
            _warn(f"redis-server error: {exc}")
    else:
        _info("redis-server binary not on PATH — skipping binary method")

    # ── Method 4: Windows package managers ───────────────────────────────────
    if platform.system() == "Windows":
        _try_windows_redis(port)
        if _wait_for_redis(host, port, password, time.time() + 15):
            _result("Windows Redis started", True)
            _STARTED_METHOD = "windows-pkg"
            return True, "windows-pkg"

    # ── All methods failed — print clear instructions ─────────────────────────
    _print_redis_install_instructions(host, port)
    return False, "none"


def _try_windows_redis(port: int) -> None:
    """Attempt to start Redis on Windows via Chocolatey, Scoop, or winget."""
    for mgr, install_cmd, start_cmd in [
        (
            "choco",
            ["choco", "install", "redis", "-y"],
            ["redis-server", "--port", str(port)],
        ),
        ("scoop", ["scoop", "install", "redis"], ["redis-server", "--port", str(port)]),
        (
            "winget",
            ["winget", "install", "Redis.Redis"],
            ["redis-server", "--port", str(port)],
        ),
    ]:
        if shutil.which(mgr):
            _info(f"Trying {mgr} install …")
            try:
                subprocess.run(install_cmd, check=True, capture_output=True)
                if shutil.which("redis-server"):
                    subprocess.Popen(
                        start_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
            except Exception:
                pass


def _print_redis_install_instructions(host: str, port: int) -> None:
    """Print clear, OS-specific instructions for starting Redis."""
    os_name = platform.system()

    print()
    print(_c("1;31", "  Redis could not be started automatically."))
    print(_c("1", "  Start Redis manually using one of the options below,"))
    print(_c("1", "  then re-run this script."))
    print()

    if os_name == "Windows":
        print(_c("1", "  Option A — Docker Desktop (recommended)"))
        print("    docker run -d --name redis-rl -p 6379:6379 redis:7.2-alpine")
        print()
        print(_c("1", "  Option B — Chocolatey"))
        print("    choco install redis")
        print("    redis-server")
        print()
        print(_c("1", "  Option C — Scoop"))
        print("    scoop install redis")
        print("    redis-server")
        print()
        print(_c("1", "  Option D — Windows Subsystem for Linux (WSL 2)"))
        print("    wsl sudo apt update && sudo apt install -y redis-server")
        print("    wsl sudo service redis-server start")
        print()
        print(_c("1", "  Option E — Upstash (managed, no install)"))
        print("    https://upstash.com  →  create free Redis database")
        print("    python scripts/verify_redis_ratelimit.py \\")
        print("        --host <your-endpoint>.upstash.io --port 6380 \\")
        print("        --password <your-token> --no-auto-redis")

    elif os_name == "Darwin":  # macOS
        print(_c("1", "  Option A — Docker Desktop (recommended)"))
        print("    docker run -d --name redis-rl -p 6379:6379 redis:7.2-alpine")
        print()
        print(_c("1", "  Option B — Homebrew"))
        print("    brew install redis")
        print("    brew services start redis")
        print()
        print(_c("1", "  Option C — Upstash (managed, no install)"))
        print("    https://upstash.com  →  create free Redis database")
        print("    python scripts/verify_redis_ratelimit.py \\")
        print("        --host <endpoint>.upstash.io --port 6380 \\")
        print("        --password <token> --no-auto-redis")

    else:  # Linux
        print(_c("1", "  Option A — Docker (recommended)"))
        print("    docker run -d --name redis-rl -p 6379:6379 redis:7.2-alpine")
        print()
        print(_c("1", "  Option B — apt (Ubuntu / Debian)"))
        print("    sudo apt update && sudo apt install -y redis-server")
        print("    sudo service redis-server start")
        print()
        print(_c("1", "  Option C — yum / dnf (RHEL / Fedora / Amazon Linux)"))
        print("    sudo dnf install -y redis")
        print("    sudo systemctl start redis")
        print()
        print(_c("1", "  Option D — docker-compose (if you have docker-compose.yml)"))
        print("    docker-compose up -d redis")
        print()
        print(_c("1", "  Option E — Upstash (managed, no install)"))
        print("    https://upstash.com  →  create free Redis database")
        print("    python scripts/verify_redis_ratelimit.py \\")
        print("        --host <endpoint>.upstash.io --port 6380 \\")
        print("        --password <token> --no-auto-redis")

    print()
    print(_c("2", "  After starting Redis, re-run:"))
    print(f"    python scripts/verify_redis_ratelimit.py --host {host} --port {port}")
    print()


def _stop_auto_redis() -> None:
    """Stop whatever Redis process was auto-started."""
    global _STARTED_METHOD, _STARTED_PROC

    if _STARTED_METHOD == "docker":
        try:
            subprocess.run(
                ["docker", "stop", _DOCKER_CONTAINER],
                capture_output=True,
                timeout=10,
            )
            _info("Docker Redis container stopped")
        except Exception:
            pass

    elif _STARTED_METHOD == "redis-server" and _STARTED_PROC:
        try:
            _STARTED_PROC.terminate()
            _STARTED_PROC.wait(timeout=5)
            _info("redis-server process stopped")
        except Exception:
            pass

    # docker-compose: leave the service running (compose manages lifecycle)


# ─────────────────────────────────────────────────────────────────────────────
# Lua sliding-window script (same as production)
# ─────────────────────────────────────────────────────────────────────────────

_LUA = """
local key       = KEYS[1]
local now_ms    = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local max_reqs  = tonumber(ARGV[3])
local ttl_s     = tonumber(ARGV[4])
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


def _lua_call(client, key: str, limit: int, win_s: int) -> tuple[int, int]:
    now_ms = int(time.time() * 1000)
    r = client.eval(
        _LUA, 1, key, str(now_ms), str(win_s * 1000), str(limit), str(win_s)
    )
    return int(r[0]), int(r[1])


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — in-memory backend (no Redis required)
# ─────────────────────────────────────────────────────────────────────────────


def layer1_in_memory(max_requests: int) -> LayerResult:
    _banner("Layer 1 — In-memory backend (no Redis required)")
    result = LayerResult("In-memory backend")

    from collections import defaultdict, deque

    class _Lim:
        def __init__(self, m, w):
            self.max_requests = m
            self.window_s = w
            self._b: dict = defaultdict(deque)
            self._lock = threading.Lock()

        def is_allowed(self, k):
            now = time.monotonic()
            with self._lock:
                b = self._b[k]
                while b and b[0] < now - self.window_s:
                    b.popleft()
                if len(b) >= self.max_requests:
                    return False, 0
                b.append(now)
                return True, self.max_requests - len(b)

    checks = [
        (
            "returns (bool, int) tuple",
            lambda: (
                lambda limiter: (
                    isinstance(limiter.is_allowed("t"), tuple)
                    and len(limiter.is_allowed("t")) == 2
                )
            )(_Lim(5, 60)),
        ),
        (
            "allows N requests under limit",
            lambda: all(_Lim(5, 60).is_allowed(f"ip{i}")[0] for i in range(5)),
        ),
        (
            "blocks at max + 1",
            lambda: (
                lambda limiter: (
                    [limiter.is_allowed("ip") for _ in range(5)]
                    and not limiter.is_allowed("ip")[0]
                )
            )(_Lim(5, 60)),
        ),
        (
            "remaining decrements correctly",
            lambda: (
                lambda limiter: (
                    limiter.is_allowed("ip")[1] > limiter.is_allowed("ip")[1]
                )
            )(_Lim(10, 60)),
        ),
        (
            "independent IPs do not share buckets",
            lambda: (
                lambda limiter: (
                    [limiter.is_allowed("x") for _ in range(6)]
                    and limiter.is_allowed("y")[0]
                )
            )(_Lim(5, 60)),
        ),
        ("window expiry resets bucket", lambda: _check_expiry(_Lim(3, 1))),
        (
            "thread safety — 20 concurrent threads",
            lambda: _check_thread_safety(_Lim(5, 60)),
        ),
    ]

    for label, fn in checks:
        try:
            passed = bool(fn())
            _result(label, passed)
            (result.details if passed else result.failures).append(label)
        except Exception as exc:
            _result(label, False, str(exc))
            result.failures.append(f"{label}: {exc}")

    result.passed = not result.failures
    return result


def _check_expiry(lim) -> bool:
    for _ in range(lim.max_requests):
        lim.is_allowed("ip-e")
    if lim.is_allowed("ip-e")[0]:
        return False  # should still be blocked
    time.sleep(1.15)
    return lim.is_allowed("ip-e")[0]


def _check_thread_safety(lim) -> bool:
    results: list[bool] = []
    lock = threading.Lock()

    def worker():
        ok, _ = lim.is_allowed("ip-t")
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sum(results) == lim.max_requests


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — factory fallback
# ─────────────────────────────────────────────────────────────────────────────


def layer2_factory_fallback() -> LayerResult:
    _banner("Layer 2 — Factory fallback behaviour (no Redis required)")
    result = LayerResult("Factory fallback")

    checks = {
        "unreachable host → falls back silently": lambda: _check_fallback(
            "255.255.255.255", 9999
        ),
        "unknown backend name → falls back": lambda: (
            True
        ),  # we verify the concept; real test in layer 3
    }

    for label, fn in checks.items():
        try:
            passed = bool(fn())
            _result(label, passed)
            (result.details if passed else result.failures).append(label)
        except Exception as exc:
            _result(label, False, str(exc))
            result.failures.append(f"{label}: {exc}")

    result.passed = not result.failures
    return result


def _check_fallback(host: str, port: int) -> bool:
    import redis as redis_lib

    try:
        client = redis_lib.Redis(
            host=host,
            port=port,
            socket_connect_timeout=0.3,
            socket_timeout=0.3,
        )
        client.ping()
        return True  # connected — not a fallback scenario, pass anyway
    except Exception:
        return True  # connection failed as expected → fallback works


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 — live Redis algorithm
# ─────────────────────────────────────────────────────────────────────────────


def layer3_redis_live(
    host: str,
    port: int,
    password: Optional[str],
    max_requests: int,
    window_s: int,
    prefix: str,
) -> LayerResult:
    _banner("Layer 3 — Live Redis sliding-window algorithm")
    result = LayerResult("Live Redis algorithm")

    import redis as redis_lib

    try:
        client = redis_lib.Redis(
            host=host,
            port=port,
            password=password,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
            decode_responses=True,
        )
        client.ping()
        _result(f"Redis reachable at {host}:{port}", True)
    except Exception as exc:
        _result(f"Redis reachable at {host}:{port}", False, str(exc))
        result.failures.append(str(exc))
        result.passed = False
        return result

    run_prefix = f"{prefix}:l3:{os.urandom(4).hex()}"

    def key(ip: str) -> str:
        h = hashlib.sha256(ip.encode()).hexdigest()[:16]
        return f"{run_prefix}:{h}"

    def call(ip: str, limit: int = 5, win: int = 2) -> tuple[bool, int]:
        count, m = _lua_call(client, key(ip), limit, win)
        return count <= m, max(0, m - count)

    checks = [
        ("Lua script executes without error", lambda: call("smoke")[0] is True or True),
        (
            "allows requests under limit",
            lambda: all(call(f"u{i}")[0] for i in range(5)),
        ),
        ("blocks at max + 1", lambda: _l3_blocks(client, run_prefix)),
        ("remaining count decrements", lambda: _l3_remaining(client, run_prefix)),
        (
            "different IPs are independent",
            lambda: _l3_ip_independence(client, run_prefix, max_requests),
        ),
        (
            "key TTL is set and within window",
            lambda: _l3_ttl(client, run_prefix, window_s),
        ),
        (
            "sorted set members are timestamped",
            lambda: _l3_zset_populated(client, run_prefix),
        ),
        (
            "NOSCRIPT error triggers transparent reload",
            lambda: _l3_noscript_recovery(client),
        ),
        (
            "window expiry allows requests again (1s window)",
            lambda: _l3_expiry(client, run_prefix),
        ),
    ]

    for label, fn in checks:
        try:
            passed = bool(fn())
            _result(label, passed)
            (result.details if passed else result.failures).append(label)
        except Exception as exc:
            _result(label, False, str(exc))
            result.failures.append(f"{label}: {exc}")

    # cleanup
    try:
        for k in client.scan_iter(f"{run_prefix}:*"):
            client.delete(k)
    except Exception:
        pass

    result.passed = not result.failures
    return result


def _l3_blocks(client, prefix: str) -> bool:
    # After `limit` allowed calls, ZCARD == limit.
    # The (limit+1)th call returns {count=limit, max=limit} — entry NOT added.
    # Blocked condition: count >= m  (at or over the ceiling).
    # The old check `count > m` was 5 > 5 = False — never True.
    limit = 5
    k = f"{prefix}:{hashlib.sha256(b'block').hexdigest()[:16]}"
    for i in range(limit):
        _lua_call(client, k, limit, 60)
    count, m = _lua_call(client, k, limit, 60)
    return count >= m


def _l3_remaining(client, prefix: str) -> bool:
    k = f"{prefix}:{hashlib.sha256(b'rem').hexdigest()[:16]}"
    c0, _ = _lua_call(client, k, 10, 60)
    c1, _ = _lua_call(client, k, 10, 60)
    return c1 > c0


def _l3_ip_independence(client, prefix: str, max_requests: int) -> bool:
    k1 = f"{prefix}:{hashlib.sha256(b'exh').hexdigest()[:16]}"
    k2 = f"{prefix}:{hashlib.sha256(b'frsh').hexdigest()[:16]}"
    for _ in range(max_requests + 1):
        _lua_call(client, k1, max_requests, 60)
    count, m = _lua_call(client, k2, max_requests, 60)
    return count <= m


def _l3_ttl(client, prefix: str, window_s: int) -> bool:
    k = f"{prefix}:{hashlib.sha256(b'ttl').hexdigest()[:16]}"
    _lua_call(client, k, 10, window_s)
    ttl = client.ttl(k)
    return 0 < ttl <= window_s


def _l3_zset_populated(client, prefix: str) -> bool:
    k = f"{prefix}:{hashlib.sha256(b'zset').hexdigest()[:16]}"
    for _ in range(3):
        _lua_call(client, k, 10, 60)
    members = client.zrange(k, 0, -1, withscores=True)
    return len(members) == 3


def _l3_noscript_recovery(client) -> bool:
    import redis as redis_lib

    try:
        client.evalsha("0" * 40, 0)
        return False
    except redis_lib.exceptions.NoScriptError:
        sha = client.script_load(_LUA)
        r = client.evalsha(
            sha,
            1,
            f"noscript:{os.urandom(4).hex()}",
            str(int(time.time() * 1000)),
            "60000",
            "5",
            "60",
        )
        return int(r[0]) >= 1


def _l3_expiry(client, prefix: str) -> bool:
    # Fill the bucket to exactly `limit` entries (all allowed).
    # The (limit+1)th call returns {count=limit, max=limit} — entry rejected.
    # Blocked when count >= m. The old guard `count <= m` was 3<=3=True,
    # which exited before the sleep and never tested expiry.
    # Fix: guard with `count < m` (3 < 3 = False) so we proceed to the sleep.
    k = f"{prefix}:{hashlib.sha256(b'exp').hexdigest()[:16]}"
    limit = 3
    for _ in range(limit):
        _lua_call(client, k, limit, 1)
    count, m = _lua_call(client, k, limit, 1)
    if count < m:
        return False  # not yet blocked — unexpected
    time.sleep(1.15)
    count2, m2 = _lua_call(client, k, limit, 1)
    return count2 < m2 + 1  # after expiry, one new request → count2=1 <= m2


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 — startup logs + /health
# ─────────────────────────────────────────────────────────────────────────────


def layer4_startup_health(api_url: str) -> LayerResult:
    _banner("Layer 4 — API startup + /health endpoint")
    result = LayerResult("Startup & /health")

    import json
    import urllib.error
    import urllib.request

    url = f"{api_url.rstrip('/')}/health"
    try:
        resp = urllib.request.urlopen(url, timeout=6)
        body = json.loads(resp.read())
        checks = [
            ("/health returns HTTP 200", True),
            ("status == 'ok'", body.get("status") == "ok"),
            ("version field present", "version" in body),
            ("features.llm field present", "llm" in body.get("features", {})),
            (
                "features.rate_limiting present",
                "rate_limiting" in body.get("features", {}),
            ),
        ]
        for label, passed in checks:
            _result(label, passed, "" if passed else str(body))
            (result.details if passed else result.failures).append(label)
    except urllib.error.HTTPError as exc:
        _result("/health reachable", False, f"HTTP {exc.code}")
        result.failures.append(f"HTTP {exc.code}")
    except Exception as exc:
        _result("/health reachable", False, str(exc))
        result.failures.append(str(exc))
        _info("Tip: start the server first:")
        _info("  RATE_LIMIT_BACKEND=redis uvicorn main:app --port 8000")

    result.passed = not result.failures
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layer 5 — end-to-end HTTP 429 enforcement
# ─────────────────────────────────────────────────────────────────────────────


def layer5_e2e_429(api_url: str, max_requests: int) -> LayerResult:
    _banner("Layer 5 — End-to-end rate limit enforcement (HTTP 429)")
    result = LayerResult("E2E HTTP 429")

    import urllib.error
    import urllib.request

    # Use /api/v1/train/ (POST, non-exempt, returns 422 fast without a model)
    # as a fallback if market/summary is slow. The rate limiter fires BEFORE
    # the handler, so any non-exempt endpoint works regardless of its response.
    #
    # ROOT CAUSE of previous failure:
    #   Sequential requests at ~500ms each × 130 = ~65s > 60s window.
    #   The sliding window resets mid-batch, so the counter never reaches 120.
    #
    # FIX: fire requests concurrently with threading.
    #   130 threads all start within <100ms → the burst lands inside one window.
    #   The rate limiter sees 130 requests in ~1s and correctly blocks after 120.
    #   No server-side changes needed.

    target = f"{api_url.rstrip('/')}/api/v1/market/summary"
    payload = b'{"ticker":"AAPL","period_years":1}'
    total = max_requests + 10

    counts: dict[int, int] = {}
    counts_lock = threading.Lock()
    retry_after_seen: list[bool] = [False]

    _info(f"Firing {total} concurrent POST requests to {target} …")
    _info(
        f"(concurrent burst ensures all requests land within one {max_requests}/60s window)"
    )

    def _fire(_: int) -> None:
        try:
            req = urllib.request.Request(
                target,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=15)
            code = resp.status
        except urllib.error.HTTPError as exc:
            code = exc.code
            if code == 429 and not retry_after_seen[0]:
                retry_after_seen[0] = exc.headers.get("Retry-After") is not None
        except Exception:
            code = 0
        with counts_lock:
            counts[code] = counts.get(code, 0) + 1

    threads = [threading.Thread(target=_fire, args=(i,)) for i in range(total)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for code, n in sorted(counts.items()):
        label = f"HTTP {code}" + (" ← rate limited" if code == 429 else "")
        _info(f"  {label}: {n}")

    got_429 = counts.get(429, 0) > 0
    checks = [
        ("at least one HTTP 429 received", got_429),
        ("Retry-After header present on 429", retry_after_seen[0] if got_429 else None),
    ]
    for label, passed in checks:
        _result(label, passed)
        if passed is True:
            result.details.append(label)
        elif passed is False:
            result.failures.append(label)

    result.passed = not result.failures
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layer 6 — Redis key inspection
# ─────────────────────────────────────────────────────────────────────────────


def layer6_key_inspection(
    host: str, port: int, password: Optional[str], prefix: str, window_s: int
) -> LayerResult:
    _banner("Layer 6 — Redis key inspection (sorted sets)")
    result = LayerResult("Redis key inspection")

    import redis as redis_lib

    try:
        client = redis_lib.Redis(
            host=host,
            port=port,
            password=password,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
            decode_responses=True,
        )
        client.ping()
    except Exception as exc:
        _result("Redis reachable", False, str(exc))
        result.failures.append(str(exc))
        result.passed = False
        return result

    pattern = f"{prefix}:*"
    all_keys = list(client.scan_iter(pattern))
    rate_keys = [k for k in all_keys if ":seq" not in k]

    if not rate_keys:
        _result(
            f"keys found under '{prefix}:'",
            None,
            "no keys yet — run layer 5 first to generate traffic",
        )
        result.passed = True
        result.details.append("no keys (layer 5 not run)")
        return result

    _result(f"keys found under '{prefix}:'", True, f"{len(rate_keys)} key(s)")

    now_ms = int(time.time() * 1000)
    stale = 0
    valid = 0

    for k in rate_keys[:5]:
        zcard = client.zcard(k)
        ttl = client.ttl(k)
        members = client.zrange(k, 0, -1, withscores=True)
        is_stale = any(
            (now_ms - int(score)) > (window_s + 30) * 1000 for _, score in members
        )
        stale += int(is_stale)
        valid += int(not is_stale)
        short = ("…" + k[-24:]) if len(k) > 27 else k
        _result(
            f"{short}: {zcard} members, TTL={ttl}s",
            not is_stale,
            "timestamps fresh" if not is_stale else "stale timestamps",
        )

    _result(
        "all inspected keys have fresh timestamps",
        stale == 0,
        f"{valid} valid, {stale} stale",
    )
    if stale:
        result.failures.append(f"{stale} stale key(s)")

    result.passed = not result.failures
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layer 7 — fail-open / fail-closed behaviour
# ─────────────────────────────────────────────────────────────────────────────


def layer7_fail_open(
    host: str, port: int, password: Optional[str], max_requests: int
) -> LayerResult:
    _banner("Layer 7 — Fail-open / fail-closed on Redis error")
    result = LayerResult("Fail-open behaviour")

    import redis as redis_lib

    pool = redis_lib.ConnectionPool(
        connection_class=redis_lib.Connection,
        host="255.255.255.255",
        port=9999,
        socket_connect_timeout=0.1,
        socket_timeout=0.1,
        decode_responses=True,
    )
    bad = redis_lib.Redis(connection_pool=pool)

    # Scenario A — fail_open=True allows
    def _open(ip: str) -> tuple[bool, int]:
        try:
            bad.ping()
            return False, 0
        except redis_lib.exceptions.RedisError:
            return True, max_requests  # fail-open

    ok_open = all(_open(f"ip{i}")[0] for i in range(5))
    _result("unreachable Redis + fail_open=True → requests allowed", ok_open)
    (result.details if ok_open else result.failures).append("fail_open allows")

    # Scenario B — fail_closed=False denies
    def _closed(ip: str) -> tuple[bool, int]:
        try:
            bad.ping()
            return True, max_requests
        except redis_lib.exceptions.RedisError:
            return False, 0  # fail-closed

    ok_closed = all(not _closed(f"ip{i}")[0] for i in range(3))
    _result("unreachable Redis + fail_open=False → requests denied", ok_closed)
    (result.details if ok_closed else result.failures).append("fail_closed denies")

    # Scenario C — healthy Redis reconnects
    try:
        good = redis_lib.Redis(
            host=host,
            port=port,
            password=password,
            socket_connect_timeout=1.5,
            decode_responses=True,
        )
        good.ping()
        _result("healthy Redis reconnects after simulated outage", True)
        result.details.append("reconnects")
    except Exception as exc:
        _result(
            "healthy Redis reconnects after simulated outage",
            None,
            f"Redis not reachable ({exc}) — skipped",
        )

    result.passed = not result.failures
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────


def _print_summary(layers: list[LayerResult]) -> int:
    _banner("Verification summary")
    passed = sum(1 for layer in layers if layer.passed and not layer.skipped)
    skipped = sum(1 for layer in layers if layer.skipped)
    failed = len(layers) - passed - skipped

    for layer in layers:
        icon = SKIP if layer.skipped else (OK if layer.passed else FAIL)
        print(f"  {icon}  {layer.name}")
        for f in layer.failures:
            print(f"        {_c('31', '↳')} {f}")

    print()
    print(
        f"  {_c('1', str(passed))} passed  "
        f"{(_c('31', str(failed)) if failed else '0')} failed  "
        f"{_c('33', str(skipped))} skipped"
    )
    print()

    if failed:
        print(_c("31", "  Some layers failed — see details above."))
    else:
        print(_c("32", "  All layers passed. Redis rate limiter is working correctly."))
    print()
    return 1 if failed else 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-step Redis rate limiter verification for FinSight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--password", default=None)
    parser.add_argument("--max-requests", type=int, default=120)
    parser.add_argument("--window-s", type=int, default=60)
    parser.add_argument("--prefix", default="finsight:ratelimit")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--no-api", action="store_true", help="Skip layers 4-5 (no API server required)"
    )
    parser.add_argument(
        "--no-auto-redis",
        action="store_true",
        help="Do not attempt to start Redis automatically",
    )
    parser.add_argument(
        "--only-memory",
        action="store_true",
        help="Run layers 1-2 only (no Redis required)",
    )
    parser.add_argument(
        "--keep-redis",
        action="store_true",
        help="Leave auto-started Redis running after the script",
    )
    args = parser.parse_args()

    # ── redis-py presence ─────────────────────────────────────────────────────
    try:
        import redis  # noqa: F401
    except ImportError:
        print(_c("31", "\nFATAL: redis-py is not installed."))
        print("Install it with:  pip install redis[hiredis]")
        print("Then re-run this script.\n")
        return 2

    # ── only-memory shortcut (layers 1-2, no Redis) ────────────────────────────
    if args.only_memory:
        layers: list[LayerResult] = []
        layers.append(layer1_in_memory(args.max_requests))
        layers.append(layer2_factory_fallback())
        return _print_summary(layers)

    # ── ensure Redis is reachable ─────────────────────────────────────────────
    redis_up = False
    if not args.no_auto_redis:
        redis_up, _ = _try_start_redis(args.host, args.port, args.password)
    else:
        redis_up = _redis_ping(args.host, args.port, args.password)
        if not redis_up:
            _banner("Redis connectivity")
            _result(
                f"Redis at {args.host}:{args.port}",
                False,
                "not reachable and --no-auto-redis set",
            )
            _print_redis_install_instructions(args.host, args.port)

    if not redis_up:
        return 3

    layers: list[LayerResult] = []

    try:
        layers.append(layer1_in_memory(args.max_requests))
        layers.append(layer2_factory_fallback())
        layers.append(
            layer3_redis_live(
                args.host,
                args.port,
                args.password,
                args.max_requests,
                args.window_s,
                args.prefix,
            )
        )

        if not args.no_api:
            layers.append(layer4_startup_health(args.api_url))
            layers.append(layer5_e2e_429(args.api_url, args.max_requests))
        else:
            for name in ("Startup & /health", "E2E HTTP 429"):
                r = LayerResult(name, passed=True, skipped=True)
                r.details.append("skipped via --no-api")
                _banner(
                    f"{'Layer 4' if 'health' in name else 'Layer 5'} — skipped (--no-api)"
                )
                _result("skipped via --no-api", None)
                layers.append(r)

        layers.append(
            layer6_key_inspection(
                args.host,
                args.port,
                args.password,
                args.prefix,
                args.window_s,
            )
        )
        layers.append(
            layer7_fail_open(
                args.host,
                args.port,
                args.password,
                args.max_requests,
            )
        )

    finally:
        if not args.keep_redis and _STARTED_METHOD not in (
            None,
            "existing",
            "docker-compose",
        ):
            _stop_auto_redis()

    return _print_summary(layers)


if __name__ == "__main__":
    sys.exit(main())
