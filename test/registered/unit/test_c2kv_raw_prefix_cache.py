"""CPU ownership tests for the native first-raw-prefix RadixCache path."""

import argparse
import ast
import heapq
import logging
import os
import sys
import time
from collections import defaultdict
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
MEM_CACHE = ROOT / "python/sglang/srt/mem_cache"


class InsertParams(NamedTuple):
    key: object
    value: object = None
    priority: int = 0
    chunked: bool = False


class InsertResult(NamedTuple):
    prefix_len: int


class MatchPrefixParams(NamedTuple):
    key: object
    req: object = None


class MatchResult(NamedTuple):
    device_indices: torch.Tensor
    last_device_node: object
    last_host_node: object
    host_hit_length: int = 0
    mamba_branching_seqlen: object = None
    cache_protected_len: object = None


class LockResult(NamedTuple):
    delta: int = 0


class EvictResult(NamedTuple):
    num_tokens_evicted: int = 0


def _source_defs(path, namespace, names=None):
    """Run unmodified source definitions without importing GPU-only modules."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and (names is None or node.name in names)
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    ast.fix_missing_locations(future)
    exec(
        compile(ast.Module(body=[future, *definitions], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


radix = _source_defs(
    MEM_CACHE / "radix_cache.py",
    {
        "BasePrefixCache": object,
        "InsertParams": InsertParams,
        "InsertResult": InsertResult,
        "MatchPrefixParams": MatchPrefixParams,
        "MatchResult": MatchResult,
        "IncLockRefResult": LockResult,
        "DecLockRefResult": LockResult,
        "EvictResult": EvictResult,
        "torch": torch,
        "time": time,
        "sys": sys,
        "heapq": heapq,
        "logging": logging,
        "defaultdict": defaultdict,
        "lru_cache": lru_cache,
        "partial": partial,
    },
)
RadixCache = radix["RadixCache"]
RadixKey = radix["RadixKey"]
PriorityStrategy = _source_defs(
    MEM_CACHE / "evict_policy.py",
    {"EvictionStrategy": object},
    names={"PriorityStrategy"},
)["PriorityStrategy"]


class SessionAwareCache:
    def __init__(self, inner):
        self.inner = inner

    @staticmethod
    def owns_finished_request(req):
        return req.session is not None

    def match_prefix(self, params):
        return self.inner.match_prefix(params)

    def __getattr__(self, name):
        return getattr(self.inner, name)


raw_cache = _source_defs(
    MEM_CACHE / "c2kv_raw_prefix_cache.py",
    {
        "os": os,
        "torch": torch,
        "InsertParams": InsertParams,
        "MatchPrefixParams": MatchPrefixParams,
        "RadixCache": RadixCache,
        "RadixKey": RadixKey,
        "SessionAwareCache": SessionAwareCache,
        "get_global_server_args": lambda: SimpleNamespace(
            speculative_algorithm=None
        ),
    },
)


class HybridReqToTokenPool:
    pass


release_namespace = _source_defs(
    MEM_CACHE / "common.py",
    {
        "HybridReqToTokenPool": HybridReqToTokenPool,
        "get_global_server_args": lambda: SimpleNamespace(
            page_size=1, speculative_algorithm=None
        ),
        "ceil_align": lambda n, size: (n + size - 1) // size * size,
    },
    names={"release_kv_cache"},
)
release_kv_cache = release_namespace["release_kv_cache"]


class ReqPool:
    def __init__(self):
        self.req_to_token = torch.zeros((8, 32), dtype=torch.int64)
        self.released_rows = set()

    def write(self, index, values):
        self.req_to_token[index] = values

    def free(self, req):
        assert req.req_pool_idx not in self.released_rows
        self.released_rows.add(req.req_pool_idx)
        req.req_pool_idx = None


class Allocator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.owned = set()
        self.freed = []

    def allocate(self, *indices):
        assert not self.owned.intersection(indices)
        self.owned.update(indices)

    def free(self, values):
        for index in values.tolist():
            index = int(index)
            assert index in self.owned, f"duplicate or invalid free: {index}"
            self.owned.remove(index)
            self.freed.append(index)


def make_cache():
    cache = RadixCache.__new__(RadixCache)
    cache.disable = False
    cache.disable_finished_insert = False
    cache.page_size = 1
    cache.is_eagle = False
    cache.enable_kv_cache_events = False
    cache.device = torch.device("cpu")
    cache.key_match_fn = radix["_key_match_page_size1"]
    cache.get_child_key_fn = radix["get_child_key"]
    cache.req_to_token_pool = ReqPool()
    cache.token_to_kv_pool_allocator = Allocator()
    cache.evictable_leaves = set()
    cache.eviction_strategy = SimpleNamespace(
        get_priority=lambda node: node.last_access_time
    )
    cache.update_eviction_metrics = lambda *_: None
    cache.reset()
    return cache


def make_priority_cache():
    cache = make_cache()
    cache.eviction_strategy = PriorityStrategy()
    return cache


def make_req(cache, row, raw_ids, kv_slots, *, segment_start=None):
    cache.token_to_kv_pool_allocator.allocate(*kv_slots)
    cache.req_to_token_pool.req_to_token[row, : len(kv_slots)] = torch.tensor(
        kv_slots
    )
    return SimpleNamespace(
        req_pool_idx=row,
        c2kv_rounds=[SimpleNamespace(tokens=raw_ids)],
        c2kv_round_idx=0,
        c2kv_round_start_len=0,
        c2kv_segments=[
            SimpleNamespace(
                token_start=len(raw_ids) if segment_start is None else segment_start
            )
        ],
        origin_input_ids=list(raw_ids) + [99],
        origin_input_ids_unpadded=list(raw_ids) + [99],
        c2kv_virtual_input_ids=list(raw_ids) + [77],
        c2kv_gist_seen=False,
        c2kv_layout=[],
        c2kv_position_correction=0,
        c2kv_use_gist_projection=False,
        output_ids=[],
        history_kv_eviction=None,
        session=None,
        input_embeds=None,
        token_type_ids=None,
        multimodal_inputs=None,
        is_retracted=False,
        kv_committed_freed=False,
        kv_overallocated_freed=False,
        kv_committed_len=0,
        kv_allocated_len=0,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        last_node=cache.root_node,
        last_host_node=cache.root_node,
        extra_key="adapter-a",
        host_hit_length=0,
        c2kv_tree_cache_prefix_len=0,
        cache_protected_len=0,
        c2kv_raw_prefix_cache=None,
        c2kv_raw_prefix_lock_held=False,
        priority=0,
    )


def commit_first_round(req, cache):
    count = len(req.c2kv_rounds[0].tokens)
    req.kv_committed_len = req.kv_allocated_len = count
    return raw_cache["cache_c2kv_first_raw_prefix"](req, cache)


def test_two_requests_share_only_real_prefix_and_release_suffix(monkeypatch):
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE", "1")
    cache = make_cache()
    wrapped = SessionAwareCache(cache)
    first = make_req(cache, 0, [10, 11, 12], [1, 2, 3, 4, 5])
    assert raw_cache["match_c2kv_first_raw_prefix"](first, wrapped)
    assert first.c2kv_raw_prefix_cache["hit_tokens"] == 0
    assert commit_first_round(first, wrapped)
    assert first.c2kv_raw_prefix_cache["inserted_tokens"] == 3

    # Synthetic gist slots remain request-owned and must be freed on finish.
    first.kv_committed_len = first.kv_allocated_len = 5
    release_kv_cache(first, wrapped, is_insert=False)
    assert set(cache.token_to_kv_pool_allocator.owned) == {1, 2, 3}
    assert cache.token_to_kv_pool_allocator.freed == [4, 5]

    second = make_req(cache, 1, [10, 11, 12], [6, 7, 8])
    assert raw_cache["match_c2kv_first_raw_prefix"](second, wrapped)
    assert second.prefix_indices.tolist() == [1, 2]
    assert second.c2kv_raw_prefix_cache["hit_tokens"] == 2
    assert len(second.c2kv_rounds[0].tokens) - len(second.prefix_indices) == 1


def test_duplicate_insert_abort_and_retract_do_not_double_free(monkeypatch):
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE", "1")
    cache = make_cache()
    first = make_req(cache, 0, [10, 11, 12], [1, 2, 3])
    duplicate = make_req(cache, 1, [10, 11, 12], [4, 5, 6])
    assert commit_first_round(first, cache)
    # Both requests started from an empty tree; the second insertion loses
    # ownership of its duplicates and must point at the first request's slots.
    assert commit_first_round(duplicate, cache)
    assert duplicate.c2kv_raw_prefix_cache["inserted_tokens"] == 0
    assert cache.req_to_token_pool.req_to_token[1, :3].tolist() == [1, 2, 3]
    assert cache.token_to_kv_pool_allocator.freed == [4, 5, 6]

    release_kv_cache(duplicate, cache, is_insert=False)  # abort
    release_kv_cache(first, cache, is_insert=False)  # retract
    assert not first.c2kv_raw_prefix_lock_held
    assert cache.token_to_kv_pool_allocator.owned == {1, 2, 3}
    assert cache.protected_size() == 0
    assert cache.evictable_size() == 3
    cache.evict(SimpleNamespace(num_tokens=3))
    assert cache.token_to_kv_pool_allocator.owned == set()
    assert sorted(cache.token_to_kv_pool_allocator.freed) == [1, 2, 3, 4, 5, 6]


def test_extra_key_separates_identical_token_prefixes(monkeypatch):
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE", "1")
    cache = make_cache()
    first = make_req(cache, 0, [10, 11, 12], [1, 2, 3])
    second = make_req(cache, 1, [10, 11, 12], [4, 5, 6])
    second.extra_key = "adapter-b"
    assert commit_first_round(first, cache)
    assert raw_cache["match_c2kv_first_raw_prefix"](second, cache)
    assert second.c2kv_raw_prefix_cache["hit_tokens"] == 0
    assert commit_first_round(second, cache)
    assert second.c2kv_raw_prefix_cache["inserted_tokens"] == 3
    assert cache.match_prefix(
        MatchPrefixParams(RadixKey([10, 11, 12], "adapter-a"))
    ).device_indices.tolist() == [1, 2, 3]
    assert cache.match_prefix(
        MatchPrefixParams(RadixKey([10, 11, 12], "adapter-b"))
    ).device_indices.tolist() == [4, 5, 6]


def test_existing_matched_node_lock_moves_to_new_prefix(monkeypatch):
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE", "1")
    cache = make_cache()
    first = make_req(cache, 0, [10, 11], [1, 2])
    assert commit_first_round(first, cache)
    release_kv_cache(first, cache, is_insert=False)

    longer = make_req(cache, 1, [10, 11, 12], [3])
    assert raw_cache["match_c2kv_first_raw_prefix"](longer, cache)
    assert longer.prefix_indices.tolist() == [1, 2]
    cache.req_to_token_pool.req_to_token[1, :3] = torch.tensor([1, 2, 3])
    cache.inc_lock_ref(longer.last_node)  # PrefillAdder owns the matched node.
    assert commit_first_round(longer, cache)
    assert longer.c2kv_raw_prefix_cache["hit_tokens"] == 2
    assert longer.c2kv_raw_prefix_cache["inserted_tokens"] == 1
    assert cache.protected_size() == 3
    release_kv_cache(longer, cache, is_insert=False)
    assert cache.protected_size() == 0
    assert cache.token_to_kv_pool_allocator.owned == {1, 2, 3}


@pytest.mark.parametrize("priority_enabled", [False, True])
def test_priority_policy_eviction_prefers_native_raw_prefix_when_enabled(
    monkeypatch, priority_enabled
):
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE", "1")
    if priority_enabled:
        monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE_PRIORITY", "1")
    else:
        monkeypatch.delenv("C2KV_NATIVE_RAW_PREFIX_CACHE_PRIORITY", raising=False)
    cache = make_priority_cache()
    raw = make_req(cache, 0, [10, 11, 12], [1, 2, 3])
    assert commit_first_round(raw, cache)
    assert raw.c2kv_raw_prefix_cache["eviction_priority"] == (
        1 if priority_enabled else 0
    )
    assert raw.priority == 0
    release_kv_cache(raw, cache, is_insert=False)

    cache.token_to_kv_pool_allocator.allocate(4, 5, 6)
    cache.insert(
        InsertParams(RadixKey([20, 21, 22], "adapter-a"), torch.tensor([4, 5, 6]))
    )
    raw_node = cache.match_prefix(
        MatchPrefixParams(RadixKey([10, 11, 12], "adapter-a"))
    ).last_device_node
    ordinary_node = cache.match_prefix(
        MatchPrefixParams(RadixKey([20, 21, 22], "adapter-a"))
    ).last_device_node
    raw_node.last_access_time = 1.0
    ordinary_node.last_access_time = 2.0
    assert raw_node.priority == (1 if priority_enabled else 0)
    assert ordinary_node.priority == 0

    assert cache.evict(SimpleNamespace(num_tokens=1)).num_tokens_evicted == 3
    assert cache.token_to_kv_pool_allocator.freed == (
        [4, 5, 6] if priority_enabled else [1, 2, 3]
    )
    assert cache.evict(SimpleNamespace(num_tokens=1)).num_tokens_evicted == 3
    assert cache.token_to_kv_pool_allocator.owned == set()


def test_priority_policy_respects_explicit_priority_after_split_and_duplicate(
    monkeypatch,
):
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE", "1")
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE_PRIORITY", "1")
    cache = make_priority_cache()
    raw = make_req(cache, 0, [10, 11, 12], [1, 2, 3])
    assert commit_first_round(raw, cache)
    release_kv_cache(raw, cache, is_insert=False)

    cache.token_to_kv_pool_allocator.allocate(4)
    result = cache.insert(
        InsertParams(RadixKey([10, 11, 13], "adapter-a"), torch.tensor([1, 2, 4]))
    )
    assert result.prefix_len == 2
    raw_node = cache.match_prefix(
        MatchPrefixParams(RadixKey([10, 11, 12], "adapter-a"))
    ).last_device_node
    ordinary_node = cache.match_prefix(
        MatchPrefixParams(RadixKey([10, 11, 13], "adapter-a"))
    ).last_device_node
    shared_node = raw_node.parent
    assert (shared_node.priority, raw_node.priority, ordinary_node.priority) == (
        1, 1, 0
    )

    duplicate = make_req(cache, 1, [10, 11, 12], [5, 6, 7])
    duplicate.priority = 3
    assert commit_first_round(duplicate, cache)
    assert duplicate.c2kv_raw_prefix_cache["inserted_tokens"] == 0
    assert duplicate.c2kv_raw_prefix_cache["eviction_priority"] == 3
    assert duplicate.priority == 3
    release_kv_cache(duplicate, cache, is_insert=False)
    assert (shared_node.priority, raw_node.priority, ordinary_node.priority) == (
        3, 3, 0
    )
    assert cache.evict(SimpleNamespace(num_tokens=1)).num_tokens_evicted == 1
    assert cache.token_to_kv_pool_allocator.freed[-1] == 4
    assert cache.evict(SimpleNamespace(num_tokens=1)).num_tokens_evicted == 1
    assert cache.token_to_kv_pool_allocator.freed[-1] == 3


def test_priority_policy_does_not_evict_locked_raw_prefix(monkeypatch):
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE", "1")
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE_PRIORITY", "1")
    cache = make_priority_cache()
    raw = make_req(cache, 0, [10, 11, 12], [1, 2, 3])
    assert commit_first_round(raw, cache)
    assert cache.protected_size() == 3

    cache.token_to_kv_pool_allocator.allocate(4)
    cache.insert(InsertParams(RadixKey([20], "adapter-a"), torch.tensor([4])))
    assert cache.evict(SimpleNamespace(num_tokens=100)).num_tokens_evicted == 1
    assert cache.token_to_kv_pool_allocator.owned == {1, 2, 3}
    release_kv_cache(raw, cache, is_insert=False)
    assert cache.evict(SimpleNamespace(num_tokens=100)).num_tokens_evicted == 3
    assert cache.token_to_kv_pool_allocator.owned == set()


def test_server_cli_accepts_priority_eviction_policy():
    path = ROOT / "python/sglang/srt/server_args.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    choices_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "RADIX_EVICTION_POLICY_CHOICES"
            for target in node.targets
        )
    )
    cli_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--radix-eviction-policy"
    )
    parser = argparse.ArgumentParser()
    cli_expression = ast.fix_missing_locations(ast.Expression(body=cli_call))
    eval(
        compile(cli_expression, str(path), "eval"),
        {
            "parser": parser,
            "RADIX_EVICTION_POLICY_CHOICES": ast.literal_eval(choices_assignment.value),
            "ServerArgs": SimpleNamespace(radix_eviction_policy="lru"),
        },
    )
    assert (
        parser.parse_args(["--radix-eviction-policy", "priority"]).radix_eviction_policy
        == "priority"
    )
    assert "priority" in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--radix-eviction-policy", "fifo"])


@pytest.mark.parametrize(
    "mode",
    [
        "off", "paged", "finished_insert_disabled", "eagle", "speculative",
        "eviction", "leading_gist", "session", "projection", "multimodal",
    ],
)
def test_incompatible_or_disabled_modes_do_not_insert(monkeypatch, mode):
    monkeypatch.setenv("C2KV_NATIVE_RAW_PREFIX_CACHE", "0" if mode == "off" else "1")
    cache = make_cache()
    req = make_req(cache, 0, [10, 11, 12], [1, 2, 3])
    if mode == "paged":
        cache.page_size = 4
    elif mode == "finished_insert_disabled":
        cache.disable_finished_insert = True
    elif mode == "eagle":
        cache.is_eagle = True
    elif mode == "speculative":
        monkeypatch.setitem(
            raw_cache,
            "get_global_server_args",
            lambda: SimpleNamespace(speculative_algorithm="EAGLE"),
        )
    elif mode == "eviction":
        req.history_kv_eviction = {"method": "h2o"}
    elif mode == "leading_gist":
        req.c2kv_segments[0].token_start = 0
    elif mode == "session":
        req.session = SimpleNamespace(streaming=True)
    elif mode == "projection":
        req.c2kv_use_gist_projection = True
    elif mode == "multimodal":
        req.multimodal_inputs = object()
    assert not commit_first_round(req, SessionAwareCache(cache))
    assert cache.total_size() == 0
    if mode == "off":
        assert req.c2kv_raw_prefix_cache is None
    else:
        assert req.c2kv_raw_prefix_cache["status"] == "skipped"
        assert req.c2kv_raw_prefix_cache["reason"]
