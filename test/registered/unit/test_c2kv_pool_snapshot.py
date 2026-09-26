"""Reconcile incremental telemetry against actual CPU pool state and callbacks."""
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from sglang.srt.mem_cache.c2kv_pool import C2KVPool
from sglang.srt.observability import paper_telemetry


def make_pool():
    return C2KVPool(16, 16, dtype=torch.float32, num_kv_heads=1,
                    head_dim=4, value_head_dim=4, num_layers=2, device="cpu")


def store(pool, key, length, repair=False):
    values = [(torch.ones((length, 4)), torch.zeros((length, 4))) for _ in range(2)]
    positions = torch.arange(length).unsqueeze(0)
    if repair:
        return pool.store_repair(key, values, positions, original_seq_len=length,
                                 already_rotated=False, repair_mode="test")
    return pool.store(key, values, torch.ones((1, length), dtype=torch.bool),
                      positions, length * 8)


def scanned(pool):
    pinned = sum(entry.gist_len for key, entry in pool._cache.items()
                 if pool._pin_counts.get(key, 0) > 0)
    evictable = sum(entry.gist_len for key, entry in pool._cache.items()
                    if pool._pin_counts.get(key, 0) == 0)
    return pool.current_tokens(), pinned, evictable


def test_all_mutations_and_allocator_callbacks_match_scan(monkeypatch):
    pool = make_pool()
    observed = []

    def sample(event, **kwargs):
        # A callback must not inherit the pool lock: telemetry takes its own
        # lock before reading pool accounting in a different thread as well.
        with ThreadPoolExecutor(max_workers=1) as executor:
            snapshot = executor.submit(pool.token_accounting).result(timeout=2)
        assert snapshot == scanned(pool)
        observed.append((event, snapshot))

    monkeypatch.setattr(paper_telemetry, "sample", sample)
    store(pool, "pinned", 3)
    assert pool.pin_many(["pinned", "pinned"])
    assert pool.pin("pinned")
    store(pool, "pinned", 5, repair=True)
    pool.unpin("pinned")
    assert pool.token_accounting() == (5, 5, 0)
    store(pool, "victim", 5)
    store(pool, "large", 10)
    assert pool.get("victim") is None
    assert any(current != pinned + evictable for _, (current, pinned, evictable) in observed)
    assert not pool.pin_many(["large", "missing"])
    assert pool._pin_counts.get("large", 0) == 0
    pool.unpin_many(["pinned", "pinned", "missing"])
    store(pool, "large", 2, repair=True)
    pool.clear()
    assert pool.token_accounting() == (0, 0, 0)


def test_randomized_transitions_reconcile_without_changing_lru():
    pool = make_pool()
    rng = random.Random(728)
    for _ in range(240):
        key = str(rng.randrange(8))
        operation = rng.randrange(7)
        if operation <= 1:
            try:
                store(pool, key, rng.randrange(1, 9), repair=operation == 1)
            except ValueError as error:
                assert "cannot allocate" in str(error)
        elif operation == 2:
            pool.pin(key)
        elif operation == 3:
            pool.unpin(key)
        elif operation == 4:
            pool.get(key)
        elif operation == 5:
            pool.pin_many([key, str(rng.randrange(8)), key])
        else:
            pool.clear()
        before = list(pool._cache)
        assert pool.token_accounting() == scanned(pool)
        assert list(pool._cache) == before
        current, pinned, evictable = pool.token_accounting()
        assert current == pinned + evictable == sum(entry.gist_len for entry in pool._cache.values())


def test_failed_copy_retains_only_completed_evictions(monkeypatch):
    pool = make_pool()
    store(pool, "old", 8)
    store(pool, "victim", 8)
    assert pool.pin("old")

    def fail_copy(layer):
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(pool.kv_cache, "get_key_buffer", fail_copy)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        store(pool, "new", 4)
    assert pool.token_accounting() == scanned(pool) == (8, 8, 0)
    assert list(pool._cache) == ["old"]
    with pytest.raises(RuntimeError, match="injected copy failure"):
        store(pool, "old", 10, repair=True)
    assert pool.token_accounting() == scanned(pool) == (8, 8, 0)


def test_reader_cannot_observe_half_published_pin(monkeypatch):
    pool = make_pool()
    store(pool, "key", 3)
    started, release, read_started, read_finished = Event(), Event(), Event(), Event()
    original = pool._publish_token_accounting

    def delayed_publication():
        started.set()
        assert release.wait(timeout=3)
        original()

    def read():
        read_started.set()
        result = pool.token_accounting()
        read_finished.set()
        return result

    monkeypatch.setattr(pool, "_publish_token_accounting", delayed_publication)
    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(pool.pin, "key")
        assert started.wait(timeout=3)
        reader = executor.submit(read)
        assert read_started.wait(timeout=3)
        try:
            assert not read_finished.wait(timeout=.02)
        finally:
            release.set()
        assert writer.result(timeout=3)
        assert reader.result(timeout=3) == (3, 3, 0)
