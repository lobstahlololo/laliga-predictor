"""Typed caching layer for the prediction API.

Design goals:

* **Pluggable**: a small :class:`Cache` protocol with an in-memory default and
  an optional Redis backend. Neither the API nor the tests depend on Redis
  being installed or reachable.
* **Safe by default**: every Redis operation is wrapped; a connection error
  degrades to a cache miss and the request still completes.
* **Correct invalidation**: callers include the model version, the fixture
  identity/kickoff and a data fingerprint in every key, so a promotion, a data
  refresh or a kickoff change cannot return a stale prediction. TTLs are a
  safety net on top of that, not the only invalidation mechanism.
* **Concurrency-safe**: :class:`InMemoryCache` uses a re-entrant lock, and
  :func:`cached_call` uses per-key locks so concurrent misses compute once.

Environment configuration:

* ``LALIGA_CACHE_BACKEND``: ``memory`` (default), ``redis`` or ``none``.
* ``LALIGA_REDIS_URL``: default ``redis://localhost:6379/0``.
* ``LALIGA_CACHE_TTL``: default TTL in seconds (default ``300``).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger(__name__)

T = TypeVar("T")

CACHE_BACKEND_ENV = "LALIGA_CACHE_BACKEND"
CACHE_TTL_ENV = "LALIGA_CACHE_TTL"
REDIS_URL_ENV = "LALIGA_REDIS_URL"
DEFAULT_TTL_SECONDS = 300.0
KEY_PREFIX = "laliga:"


@runtime_checkable
class Cache(Protocol, Generic[T]):
    """Minimal typed cache contract shared by all backends."""

    def get(self, key: str) -> T | None:
        ...

    def set(self, key: str, value: T, ttl: float | None = None) -> bool:
        ...

    def delete(self, key: str) -> None:
        ...

    def clear(self) -> None:
        ...

    def close(self) -> None:
        ...


class NullCache(Generic[T]):
    """No-op backend used when caching is explicitly disabled."""

    backend = "none"

    def get(self, key: str) -> T | None:
        return None

    def set(self, key: str, value: T, ttl: float | None = None) -> bool:
        return False

    def delete(self, key: str) -> None:
        return None

    def clear(self) -> None:
        return None

    def close(self) -> None:
        return None


class InMemoryCache(Generic[T]):
    """Thread-safe TTL cache with bounded size (oldest entry evicted first)."""

    backend = "memory"

    def __init__(
        self,
        *,
        default_ttl: float = DEFAULT_TTL_SECONDS,
        max_entries: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.default_ttl = float(default_ttl)
        self.max_entries = max(1, int(max_entries))
        self._clock = clock
        self._lock = threading.RLock()
        self._store: OrderedDict[str, tuple[float | None, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _expired(expires_at: float | None, now: float) -> bool:
        return expires_at is not None and expires_at <= now

    def get(self, key: str) -> T | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self.misses += 1
                return None
            expires_at, value = item
            if self._expired(expires_at, self._clock()):
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: T, ttl: float | None = None) -> bool:
        lifetime = self.default_ttl if ttl is None else float(ttl)
        if lifetime <= 0:
            return False
        with self._lock:
            expires_at = self._clock() + lifetime if lifetime is not None else None
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)
            return True

    def delete(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def close(self) -> None:
        self.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class RedisCache(Generic[T]):
    """Optional Redis backend that never raises into request handling."""

    backend = "redis"

    def __init__(
        self,
        client: Any | None = None,
        *,
        url: str | None = None,
        default_ttl: float = DEFAULT_TTL_SECONDS,
        prefix: str = KEY_PREFIX,
    ) -> None:
        self.default_ttl = float(default_ttl)
        self.prefix = prefix
        self._client = client if client is not None else self._connect(url)
        self._available = self._client is not None

    @staticmethod
    def _connect(url: str | None) -> Any | None:
        try:
            import redis  # type: ignore import-not-found  # optional dependency
        except Exception as exc:  # noqa: BLE001 - cache is optional
            logger.warning("Redis backend requested but the redis package is unavailable: %s", exc)
            return None
        try:
            client = redis.Redis.from_url(
                url or os.environ.get(REDIS_URL_ENV, "redis://localhost:6379/0"),
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=False,
            )
            client.ping()
            return client
        except Exception as exc:  # noqa: BLE001 - fall back to a cache miss
            logger.warning("Redis cache unavailable: %s", exc)
            return None

    @property
    def available(self) -> bool:
        return self._available

    def _name(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> T | None:
        if not self._available:
            return None
        try:
            raw = self._client.get(self._name(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis GET failed (treating as a miss): %s", exc)
            self._available = False
            return None
        if raw is None:
            return None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("Discarding an unreadable Redis cache entry for %s", key)
            return None

    def set(self, key: str, value: T, ttl: float | None = None) -> bool:
        if not self._available:
            return False
        lifetime = max(1, int(self.default_ttl if ttl is None else ttl))
        try:
            payload = json.dumps(value, default=str)
            return bool(self._client.setex(self._name(key), lifetime, payload))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis SET failed (continuing uncached): %s", exc)
            return False

    def delete(self, key: str) -> None:
        if not self._available:
            return
        try:
            self._client.delete(self._name(key))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis DELETE failed: %s", exc)

    def clear(self) -> None:
        if not self._available:
            return
        try:
            for key in self._client.scan_iter(match=f"{self.prefix}*"):
                self._client.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis CLEAR failed: %s", exc)

    def close(self) -> None:
        try:
            if self._client is not None and hasattr(self._client, "close"):
                self._client.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass


def build_cache(
    backend: str | None = None,
    *,
    ttl: float | None = None,
    redis_client: Any | None = None,
) -> Cache:
    """Construct the cache selected by the environment (memory by default)."""
    selected = (backend or os.environ.get(CACHE_BACKEND_ENV, "memory")).strip().lower() or "memory"
    lifetime = float(ttl if ttl is not None else os.environ.get(CACHE_TTL_ENV, DEFAULT_TTL_SECONDS))
    if selected in {"none", "off", "disabled"}:
        return NullCache()
    if selected == "memory":
        return InMemoryCache(default_ttl=lifetime)
    if selected == "redis":
        cache = RedisCache(redis_client, default_ttl=lifetime)
        if cache.available:
            return cache
        logger.warning("Redis is unavailable; using the in-memory cache for this process.")
        return InMemoryCache(default_ttl=lifetime)
    raise ValueError(f"Unsupported cache backend: {selected!r}")


# ---------------------------------------------------------------------------
# Single-flight helper
# ---------------------------------------------------------------------------

_KEY_LOCKS: dict[str, threading.Lock] = {}
_KEY_LOCKS_GUARD = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _KEY_LOCKS_GUARD:
        lock = _KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _KEY_LOCKS[key] = lock
        return lock


def cached_call(
    cache: Cache | None,
    key: str,
    loader: Callable[[], T],
    *,
    ttl: float | None = None,
) -> T:
    """Return a cached value or compute it once (per-key single flight).

    Loader exceptions are propagated and nothing is cached, so failures are
    never persisted as stale results.
    """
    if cache is None:
        return loader()
    cached = cache.get(key)
    if cached is not None:
        return cached
    with _lock_for(key):
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = loader()
        try:
            cache.set(key, value, ttl=ttl)
        except Exception as exc:  # noqa: BLE001 - caching is best effort
            logger.warning("Cache write failed: %s", exc)
        return value


def cache_key(*parts: Any) -> str:
    """Join key parts into a stable, delimiter-safe key fragment."""
    encoded = []
    for part in parts:
        text = "" if part is None else str(part).strip()
        encoded.append(text.replace("|", "/"))
    return "|".join(encoded)
