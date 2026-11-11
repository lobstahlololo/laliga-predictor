"""Typed cache: backends, TTL, invalidation, fallback and single flight."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.api import cache as cache_module
from src.api.cache import (
    InMemoryCache,
    NullCache,
    RedisCache,
    build_cache,
    cache_key,
    cached_call,
)


def test_memory_cache_hit_miss_and_expiry():
    now = [1000.0]
    cache = InMemoryCache(default_ttl=10, clock=lambda: now[0])

    assert cache.get("missing") is None
    assert cache.set("key", {"value": 1}) is True
    assert cache.get("key") == {"value": 1}
    assert cache.hits == 1 and cache.misses == 1

    now[0] += 9.9
    assert cache.get("key") == {"value": 1}
    now[0] += 0.2
    assert cache.get("key") is None


def test_memory_cache_delete_clear_and_bounded_eviction():
    cache = InMemoryCache(default_ttl=60, max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None  # oldest evicted
    assert cache.get("b") == 2

    cache.delete("b")
    assert cache.get("b") is None
    cache.set("d", 4)
    cache.clear()
    assert len(cache) == 0


def test_ttl_zero_disables_storage():
    cache = InMemoryCache(default_ttl=60)
    assert cache.set("k", 1, ttl=0) is False
    assert cache.get("k") is None


def test_build_cache_selects_memory_none_and_unknown(monkeypatch):
    monkeypatch.delenv(cache_module.CACHE_BACKEND_ENV, raising=False)
    assert isinstance(build_cache(), InMemoryCache)

    monkeypatch.setenv(cache_module.CACHE_BACKEND_ENV, "none")
    assert isinstance(build_cache(), NullCache)

    monkeypatch.setenv(cache_module.CACHE_BACKEND_ENV, "nonsense")
    with pytest.raises(ValueError, match="Unsupported cache backend"):
        build_cache()

    monkeypatch.setenv(cache_module.CACHE_BACKEND_ENV, "memory")
    monkeypatch.setenv(cache_module.CACHE_TTL_ENV, "12")
    cache = build_cache()
    assert isinstance(cache, InMemoryCache)
    assert cache.default_ttl == 12.0


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.ttls = {}

    def get(self, key):
        return self.data.get(key)

    def setex(self, key, ttl, value):
        self.data[key] = value
        self.ttls[key] = ttl
        return True

    def delete(self, key):
        self.data.pop(key, None)
        return True

    def scan_iter(self, match=None):
        prefix = (match or "").rstrip("*")
        return [key for key in list(self.data) if key.startswith(prefix)]

    def close(self):
        return None


class BrokenRedis:
    def get(self, key):
        raise ConnectionError("redis down")

    def setex(self, key, ttl, value):
        raise ConnectionError("redis down")

    def delete(self, key):
        raise ConnectionError("redis down")

    def scan_iter(self, match=None):
        raise ConnectionError("redis down")


def test_redis_cache_round_trip_and_prefix():
    client = FakeRedis()
    cache = RedisCache(client, default_ttl=45)
    assert cache.set("predictions|1", {"probs": [0.5, 0.3, 0.2]}) is True
    assert list(client.data)[0].startswith(cache_module.KEY_PREFIX)
    assert cache.get("predictions|1") == {"probs": [0.5, 0.3, 0.2]}
    assert client.ttls[list(client.data)[0]] == 45
    cache.delete("predictions|1")
    assert cache.get("predictions|1") is None


def test_redis_failure_never_raises_and_builder_falls_back(monkeypatch):
    broken = BrokenRedis()
    cache = RedisCache(broken)
    assert cache.available is True  # injected client is accepted as-is
    assert cache.get("k") is None
    assert cache.set("k", 1) is False
    assert cache.available is False

    # A Redis backend that cannot be constructed degrades to memory.
    class UnavailableRedis:
        available = False

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(cache_module, "RedisCache", UnavailableRedis)
    fallback = build_cache("redis")
    assert isinstance(fallback, InMemoryCache)


def test_redis_unreadable_payload_is_a_miss():
    client = FakeRedis()
    client.data["laliga:key"] = b"not-json"
    cache = RedisCache(client)
    assert cache.get("key") is None


def test_cached_call_is_single_flight_under_concurrency():
    cache = InMemoryCache(default_ttl=60)
    calls = {"n": 0}
    guard = threading.Lock()

    def loader():
        with guard:
            calls["n"] += 1
        time.sleep(0.05)
        return {"value": 42}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: cached_call(cache, "same-key", loader), range(8)))

    assert results == [{"value": 42}] * 8
    assert calls["n"] == 1


def test_cached_call_does_not_cache_exceptions():
    cache = InMemoryCache(default_ttl=60)
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        raise RuntimeError("loader failed")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="loader failed"):
            cached_call(cache, "key", failing)
    assert calls["n"] == 2
    assert cache.get("key") is None


def test_cached_call_without_cache_always_computes():
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return calls["n"]

    assert cached_call(None, "key", loader) == 1
    assert cached_call(None, "key", loader) == 2


def test_cache_key_is_stable_and_delimiter_safe():
    assert cache_key("predict", 1, None, "Real|Madrid") == "predict|1||Real/Madrid"
    assert cache_key("a", "b") == cache_key("a", "b")
    assert cache_key("a", "b") != cache_key("a", "c")
