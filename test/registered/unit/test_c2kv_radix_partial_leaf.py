"""CPU checks for opt-in, page-one partial RadixCache leaf eviction."""

import abc
import hashlib
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
    "partial_leaf_test_support", Path(__file__).with_name("test_c2kv_raw_prefix_cache.py")
)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)

allocator_defs = support._source_defs(
    support.MEM_CACHE / "allocator.py",
    {"abc": abc, "torch": torch,
     "paper_telemetry": SimpleNamespace(sample=lambda *_: None)},
    names={"BaseTokenToKVPoolAllocator", "TokenToKVPoolAllocator"},
)
TokenAllocator = allocator_defs["TokenToKVPoolAllocator"]


class EvictParams(NamedTuple):
    num_tokens: int


class BlockStored(NamedTuple):
    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None
    medium: str


class BlockRemoved(NamedTuple):
    block_hashes: list[int]
    medium: str


def hash_string(tokens, prior_hash=None):
    return hashlib.sha256(repr((tokens, prior_hash)).encode()).hexdigest()


@pytest.fixture(autouse=True)
def partial_leaf_opt_in(monkeypatch):
    monkeypatch.setenv("C2KV_RADIX_EVICT_PARTIAL_LEAF", "1")
    monkeypatch.setitem(support.radix, "TokenToKVPoolAllocator", TokenAllocator)


def make_cache(*, need_sort=False, size=24, events=False):
    cache = support.make_cache()
    allocator = TokenAllocator(size, torch.bfloat16, "cpu", None, need_sort)
    cache.token_to_kv_pool_allocator = allocator
    if events:
        cache.enable_kv_cache_events = True
        cache.kv_event_queue = []
        support.radix.update(
            get_hash_str=hash_string,
            hash_str_to_int64=lambda value: int(value[:15], 16),
            BlockStored=BlockStored,
            BlockRemoved=BlockRemoved,
            MEDIUM_GPU="GPU",
        )
    return cache, allocator


def insert(cache, allocator, tokens, *, extra_key=None, priority=0):
    slots = allocator.alloc(len(tokens))
    cache.insert(
        support.InsertParams(
            support.RadixKey(tokens, extra_key=extra_key), slots, priority=priority
        )
    )
    return slots


def match(cache, tokens, *, extra_key=None):
    return cache.match_prefix(
        support.MatchPrefixParams(support.RadixKey(tokens, extra_key=extra_key))
    ).device_indices


@pytest.mark.parametrize("shortfall,partial,retained_slots", [
    (False, False, 0), (True, False, 0), (False, True, 4), (True, True, 8),
])
@pytest.mark.parametrize("need_sort", [False, True])
def test_allocation_combines_shortfall_and_partial_leaf(
    monkeypatch, shortfall, partial, retained_slots, need_sort
):
    monkeypatch.setenv("C2KV_RADIX_EVICT_SHORTFALL_ONLY", str(int(shortfall)))
    monkeypatch.setenv("C2KV_RADIX_EVICT_PARTIAL_LEAF", str(int(partial)))
    common = support._source_defs(
        support.MEM_CACHE / "common.py",
        {"os": os, "time": time, "EvictParams": EvictParams,
         "TokenToKVPoolAllocator": TokenAllocator,
         "RadixCache": support.RadixCache,
         "SessionAwareCache": support.SessionAwareCache,
         "SWATokenToKVPoolAllocator": type("UnusedSWAAllocator", (), {}),
         "logger": logging.getLogger(__name__)},
        names={"_c2kv_shortfall_eviction_supported", "evict_from_tree_cache", "alloc_token_slots"},
    )
    cache, allocator = make_cache(need_sort=need_sort, size=16)
    cache.is_chunk_cache = lambda: False
    tokens = list(range(91, 100))
    original = insert(cache, allocator, tokens)
    live = allocator.alloc(3)
    assert allocator.available_size() == 4
    allocated = common["alloc_token_slots"](cache, 5)
    retained = match(cache, tokens)
    assert retained.tolist() == original[:retained_slots].tolist()
    assert len(allocated) == 5
    occupied = torch.cat((live, allocated, retained)).tolist()
    assert len(occupied) == len(set(occupied))
    assert len(occupied) + allocator.available_size() == allocator.size


@pytest.mark.parametrize("need_sort", [False, True])
def test_partial_eviction_retains_exact_slots_and_reinserts(need_sort):
    cache, allocator = make_cache(need_sort=need_sort, size=16)
    tokens = list(range(10, 19))
    original = insert(cache, allocator, tokens, extra_key="a")
    live = allocator.alloc(3)
    free_before = allocator.available_size()

    result = cache.evict(EvictParams(1))
    assert result.num_tokens_evicted == 1
    assert allocator.available_size() == free_before + 1
    assert cache.total_size() == cache.evictable_size() == 8
    assert cache.protected_size() == 0
    prefix = next(iter(cache.root_node.children.values()))
    assert prefix in cache.evictable_leaves
    assert len(cache.evictable_leaves) == 1
    assert prefix.key.token_ids == tokens[:-1]
    assert prefix.key.extra_key == "a"
    assert prefix.value.tolist() == original[:-1].tolist()
    assert not prefix.children
    assert match(cache, tokens, extra_key="a").tolist() == original[:-1].tolist()
    assert len(match(cache, tokens, extra_key="b")) == 0
    free_slots = torch.cat((allocator.free_pages, allocator.release_pages)).tolist()
    assert original[-1].item() in free_slots

    if need_sort:
        untouched_free = allocator.alloc(len(allocator.free_pages))
        assert allocator.available_size() == 1
    recovered = allocator.alloc(1)
    if need_sort:
        assert recovered.tolist() == original[-1:].tolist()
        allocator.free(untouched_free)
    assert len(set(torch.cat((live, prefix.value, recovered)).tolist())) == 12
    assert cache.insert(
        support.InsertParams(
            support.RadixKey(tokens, extra_key="a"),
            torch.cat((prefix.value, recovered)),
        )
    ).prefix_len == 8
    assert match(cache, tokens, extra_key="a").tolist() == torch.cat(
        (original[:-1], recovered)
    ).tolist()
    assert len(set(torch.cat((live, match(cache, tokens, extra_key="a"))).tolist())) == 12


def test_repeated_partial_eviction_and_locked_ancestor():
    cache, allocator = make_cache()
    tokens = list(range(20, 30))
    original = insert(cache, allocator, tokens)
    for amount, retained in ((2, 8), (3, 5)):
        before = allocator.available_size()
        assert cache.evict(EvictParams(amount)).num_tokens_evicted == amount
        assert allocator.available_size() == before + amount
        assert cache.total_size() == cache.evictable_size() == retained
        assert match(cache, tokens).tolist() == original[:retained].tolist()

    cache, allocator = make_cache()
    original = insert(cache, allocator, tokens)
    locked = cache.match_prefix(
        support.MatchPrefixParams(support.RadixKey(tokens[:4]))
    ).last_device_node
    cache.inc_lock_ref(locked)
    before = allocator.available_size()
    assert cache.evict(EvictParams(2)).num_tokens_evicted == 2
    assert allocator.available_size() == before + 2
    assert cache.protected_size() == 4
    assert cache.evictable_size() == 4
    assert match(cache, tokens).tolist() == original[:8].tolist()
    cache.dec_lock_ref(locked)
    assert cache.protected_size() == 0


@pytest.mark.parametrize("policy", ["lru", "mru", "fifo", "filo", "lfu", "slru", "priority"])
def test_split_preserves_eviction_metadata_and_policy_order(policy):
    strategies = support._source_defs(
        support.MEM_CACHE / "evict_policy.py",
        {"EvictionStrategy": object},
        names={"LRUStrategy", "MRUStrategy", "FIFOStrategy", "FILOStrategy",
               "LFUStrategy", "SLRUStrategy", "PriorityStrategy"},
    )
    cache, allocator = make_cache()
    cache.eviction_strategy = strategies[policy.upper() + "Strategy" if policy != "priority" else "PriorityStrategy"]()
    insert(cache, allocator, [1, 2, 3, 4, 5], priority=0)
    insert(cache, allocator, [8, 9, 10], priority=1)
    first = cache.root_node.children[1]
    other = cache.root_node.children[8]
    first.last_access_time = first.creation_time = 3.0 if policy in ("mru", "filo") else 1.0
    other.last_access_time = other.creation_time = 2.0
    first.hit_count = 0
    other.hit_count = 3
    old_priority = cache.eviction_strategy.get_priority(first)
    old_creation = first.creation_time
    old_access = first.last_access_time

    assert cache.evict(EvictParams(1)).num_tokens_evicted == 1
    retained = cache.root_node.children[1]
    assert retained.last_access_time == old_access
    assert retained.creation_time == old_creation
    assert retained.hit_count == first.hit_count
    assert retained.priority == first.priority
    assert cache.eviction_strategy.get_priority(retained) == old_priority
    assert cache.evict(EvictParams(1)).num_tokens_evicted == 1
    assert cache.root_node.children[1].key.token_ids == [1, 2, 3]
    assert cache.root_node.children[8] is other


def test_events_remove_only_suffix_and_reinsert_on_existing_hash_chain():
    cache, allocator = make_cache(events=True)
    tokens = [31, 32, 33, 34]
    slots = insert(cache, allocator, tokens)
    stored = cache.take_events()
    hashes = [event.block_hashes[0] for event in stored]
    assert len(stored) == 4
    assert all(isinstance(event, BlockStored) for event in stored)
    assert [event.parent_block_hash for event in stored] == [None, *hashes[:-1]]

    assert cache.evict(EvictParams(1)).num_tokens_evicted == 1
    removed = cache.take_events()
    assert removed == [BlockRemoved([hashes[-1]], "GPU")]
    retained = cache.root_node.children[31]
    assert retained.hash_value is not None and len(retained.hash_value) == 3
    assert [int(value[:15], 16) for value in retained.hash_value] == hashes[:-1]
    assert match(cache, tokens).tolist() == slots[:-1].tolist()

    suffix = allocator.alloc(1)
    cache.insert(
        support.InsertParams(support.RadixKey(tokens), torch.cat((retained.value, suffix)))
    )
    assert cache.take_events() == [
        BlockStored([hashes[-1]], hashes[-2], [tokens[-1]], 1, None, "GPU")
    ]


def test_trace_identifies_partial_branch_only_when_enabled(monkeypatch, caplog):
    cache, allocator = make_cache()
    insert(cache, allocator, [51, 52, 53, 54])
    caplog.set_level(logging.INFO)
    cache.evict(EvictParams(1))
    assert not [record for record in caplog.records if "C2KV_RADIX_PARTIAL_LEAF" in record.message]

    monkeypatch.setenv("C2KV_RADIX_EVICT_TRACE", "1")
    cache.evict(EvictParams(1))
    rows = [record.message for record in caplog.records if "C2KV_RADIX_PARTIAL_LEAF" in record.message]
    assert rows == [
        "C2KV_RADIX_PARTIAL_LEAF original_leaf_len=3 suffix_freed=1 prefix_retained=2"
    ]


@pytest.mark.parametrize("case", ["flag_off", "cache_subclass", "allocator_subclass", "paged", "eagle", "bigram"])
def test_unsupported_pair_or_default_off_keeps_whole_leaf_eviction(monkeypatch, case):
    cache, allocator = make_cache()
    insert(cache, allocator, [41, 42, 43, 44])
    if case == "flag_off":
        monkeypatch.delenv("C2KV_RADIX_EVICT_PARTIAL_LEAF")
    elif case == "cache_subclass":
        class OtherCache(support.RadixCache):
            pass
        cache.__class__ = OtherCache
    elif case == "allocator_subclass":
        class OtherAllocator(TokenAllocator):
            pass
        allocator.__class__ = OtherAllocator
    elif case == "paged":
        cache.page_size = 2
    elif case == "eagle":
        cache.is_eagle = True
    else:
        cache.root_node.children[41].key.is_bigram = True

    assert cache.evict(EvictParams(1)).num_tokens_evicted == 4
    assert cache.total_size() == cache.evictable_size() == 0
    assert allocator.available_size() == allocator.size
