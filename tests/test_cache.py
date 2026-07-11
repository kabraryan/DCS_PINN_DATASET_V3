from __future__ import annotations

from pathlib import Path

from benchmark.cache import cache_key, cached, clear


def test_key_is_stable_and_order_sensitive():
    assert cache_key("a", 1) == cache_key("a", 1)
    assert cache_key("a", 1) != cache_key(1, "a")


def test_key_changes_with_params():
    assert cache_key("ep", {"r0_um": 4.0}) != cache_key("ep", {"r0_um": 0.7})


def test_cached_computes_once(tmp_path: Path):
    calls = []

    def compute():
        calls.append(1)
        return 42.0

    k = cache_key("x")
    assert cached(tmp_path, k, compute) == 42.0
    assert cached(tmp_path, k, compute) == 42.0
    assert len(calls) == 1


def test_cached_roundtrips_none(tmp_path: Path):
    k = cache_key("deficit-is-none")
    assert cached(tmp_path, k, lambda: None) is None
    assert cached(tmp_path, k, lambda: 999.0) is None   # served from cache


def test_clear_removes_entries(tmp_path: Path):
    k = cache_key("y")
    cached(tmp_path, k, lambda: 1.0)
    clear(tmp_path)
    calls = []
    cached(tmp_path, k, lambda: (calls.append(1), 2.0)[1])
    assert len(calls) == 1
