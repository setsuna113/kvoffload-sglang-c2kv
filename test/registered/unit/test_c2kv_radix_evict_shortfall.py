"""CPU allocation and prefix-retention checks for shortfall-only eviction."""
import abc
import importlib.util
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "raw_prefix_test_support", Path(__file__).with_name("test_c2kv_raw_prefix_cache.py")
)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


class EvictParams(NamedTuple):
    num_tokens: int
    swa_num_tokens: int = 0


class SWATokenToKVPoolAllocator:
    pass


allocator_defs = support._source_defs(
    support.MEM_CACHE / "allocator.py",
    {"abc": abc, "torch": torch,
     "paper_telemetry": SimpleNamespace(sample=lambda *_: None)},
    names={"BaseTokenToKVPoolAllocator", "TokenToKVPoolAllocator"},
)
TokenAllocator = allocator_defs["TokenToKVPoolAllocator"]
common = support._source_defs(
    support.MEM_CACHE / "common.py",
    {"os": os, "time": time, "EvictParams": EvictParams,
     "TokenToKVPoolAllocator": TokenAllocator,
     "RadixCache": support.RadixCache,
     "SessionAwareCache": support.SessionAwareCache,
     "SWATokenToKVPoolAllocator": SWATokenToKVPoolAllocator,
     "get_global_server_args": lambda: SimpleNamespace(enable_c2kv=True),
     "logger": logging.getLogger(__name__)},
    names={"_c2kv_shortfall_eviction_supported", "evict_from_tree_cache", "alloc_token_slots"},
)


def make_split_prefix_cache(need_sort=False):
    cache = support.make_cache()
    cache.is_chunk_cache = lambda: False
    allocator = TokenAllocator(16, torch.bfloat16, "cpu", None, need_sort)
    cache.token_to_kv_pool_allocator = allocator
    # Nine cached slots become an eight-token prefix and a one-token leaf.
    key = support.RadixKey(list(range(10, 19)))
    cache.insert(support.InsertParams(key, allocator.alloc(9)))
    raw = cache.match_prefix(
        support.MatchPrefixParams(support.RadixKey(list(range(10, 18))))
    )
    raw.last_device_node.last_access_time = 0.0
    child = next(iter(raw.last_device_node.children.values()))
    child.last_access_time = 1.0
    live = allocator.alloc(3)
    assert allocator.available_size() == 4
    return cache, allocator, live, raw.last_device_node


@pytest.mark.parametrize("enabled,expected_cached", [(False, 0), (True, 8)])
@pytest.mark.parametrize("need_sort", [False, True])
def test_one_slot_shortfall_preserves_reusable_parent(
    monkeypatch, enabled, expected_cached, need_sort
):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", "1" if enabled else "0")
    cache, allocator, live, _ = make_split_prefix_cache(need_sort)
    allocated = common["alloc_token_slots"](cache, 5)
    assert len(allocated) == 5
    retained = cache.match_prefix(
        support.MatchPrefixParams(support.RadixKey(list(range(10, 18))))
    ).device_indices
    assert len(retained) == expected_cached
    occupied = torch.cat((live, allocated, retained)).tolist()
    assert len(set(occupied)) == len(occupied)
    assert len(occupied) + allocator.available_size() == allocator.size


def test_released_sorted_slots_count_toward_the_shortfall(monkeypatch):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", "1")
    cache, allocator, live, _ = make_split_prefix_cache(need_sort=True)
    released = allocator.alloc(4)
    allocator.free(released)
    assert len(allocator.free_pages) == 0
    assert len(allocator.release_pages) == allocator.available_size() == 4
    allocated, state = common["alloc_token_slots"](cache, 5, backup_state=True)
    assert len(allocated) == 5 and cache.total_size() == 8
    assert allocator.available_size() == 0
    allocator.restore_state(state)
    assert allocator.available_size() == 5
    assert len(set(torch.cat((live, allocator.alloc(5))).tolist())) == 8


def test_shortfall_does_not_release_locked_prefix(monkeypatch):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", "1")
    cache, allocator, _, raw_node = make_split_prefix_cache()
    cache.inc_lock_ref(raw_node)
    allocator.alloc(4)
    common["evict_from_tree_cache"](cache, 2)
    assert allocator.available_size() == 1
    assert allocator.alloc(2) is None
    assert cache.protected_size() == cache.total_size() == 8
    cache.dec_lock_ref(raw_node)
    common["evict_from_tree_cache"](cache, 2)
    assert len(allocator.alloc(2)) == 2


def test_sufficient_free_slots_do_not_evict(monkeypatch):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", "1")
    cache, allocator, _, _ = make_split_prefix_cache()
    common["evict_from_tree_cache"](cache, 4)
    assert cache.total_size() == 9 and allocator.available_size() == 4


def test_session_wrapper_uses_inner_radix_shortfall(monkeypatch):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", "true")
    cache, allocator, _, _ = make_split_prefix_cache()
    wrapped = support.SessionAwareCache(cache)
    assert len(common["alloc_token_slots"](wrapped, 5)) == 5
    assert cache.total_size() == 8 and allocator.available_size() == 0


def test_default_preserves_legacy_eviction(monkeypatch):
    monkeypatch.delenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", raising=False)
    monkeypatch.delenv("C2KV_RADIX_EVICT_TRACE", raising=False)
    cache, allocator, _, _ = make_split_prefix_cache()
    assert len(common["alloc_token_slots"](cache, 5)) == 5
    assert cache.total_size() == 0 and allocator.available_size() == 8


@pytest.mark.parametrize("enabled,target,evicted,free_after", [
    (False, 5, 9, 13), (True, 1, 1, 5),
])
def test_trace_records_requested_and_actual_eviction(
    monkeypatch, caplog, enabled, target, evicted, free_after
):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", str(int(enabled)))
    monkeypatch.setenv("C2KV_RADIX_EVICT_TRACE", "1")
    caplog.set_level(logging.INFO)
    cache, _, _, _ = make_split_prefix_cache()
    common["alloc_token_slots"](cache, 5)
    rows = [r.message for r in caplog.records if "C2KV_RADIX_EVICT " in r.message]
    assert len(rows) == 1
    assert f"requested=5 available_before=4 target={target} evicted={evicted}" in rows[0]
    assert f"available_after={free_after} shortfall_only={enabled}" in rows[0]
    assert float(rows[0].split("elapsed_seconds=")[1]) >= 0


def test_cache_subclass_retains_legacy_contract(monkeypatch):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", "1")
    cache, allocator, _, _ = make_split_prefix_cache()

    class AlternateRadixCache(support.RadixCache):
        pass

    cache.__class__ = AlternateRadixCache
    assert len(common["alloc_token_slots"](cache, 5)) == 5
    assert cache.total_size() == 0 and allocator.available_size() == 8


def test_paged_allocator_retains_existing_eviction_request(monkeypatch):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", "1")
    calls = []
    cache = SimpleNamespace(
        is_chunk_cache=lambda: False,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=2, available_size=lambda: 4),
        evict=lambda params: calls.append(params),
    )
    common["evict_from_tree_cache"](cache, 5)
    assert calls == [EvictParams(num_tokens=5)]


def test_hybrid_allocator_keeps_its_two_existing_shortfalls(monkeypatch):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", "1")
    allocator = SWATokenToKVPoolAllocator()
    allocator.full_available_size = lambda: 4
    allocator.swa_available_size = lambda: 2
    calls = []
    cache = SimpleNamespace(is_chunk_cache=lambda: False,
                            token_to_kv_pool_allocator=allocator,
                            evict=lambda params: calls.append(params))
    common["evict_from_tree_cache"](cache, 5)
    assert calls == [EvictParams(num_tokens=1, swa_num_tokens=3)]
