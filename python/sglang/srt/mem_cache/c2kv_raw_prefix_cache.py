"""Share only the real-token prefix preceding a native C2KV injection."""

import os

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.session_aware_cache import SessionAwareCache
from sglang.srt.server_args import get_global_server_args


def native_raw_prefix_cache_enabled() -> bool:
    return os.environ.get("C2KV_NATIVE_RAW_PREFIX_CACHE", "false").lower() in (
        "1",
        "true",
    )


def _candidate(req, tree_cache):
    if not native_raw_prefix_cache_enabled():
        return None, None, "disabled"

    if isinstance(tree_cache, SessionAwareCache):
        if tree_cache.owns_finished_request(req):
            return None, None, "session_owned"
        radix_cache = tree_cache.inner
    else:
        radix_cache = tree_cache

    if type(radix_cache) is not RadixCache:
        return None, None, "cache_type"
    if radix_cache.disable:
        return None, None, "radix_disabled"
    if radix_cache.disable_finished_insert:
        return None, None, "finished_insert_disabled"
    if radix_cache.is_eagle:
        return None, None, "eagle_cache"
    if radix_cache.page_size != 1:
        return None, None, "page_size"
    if get_global_server_args().speculative_algorithm is not None:
        return None, None, "speculative_decoding"
    rounds = getattr(req, "c2kv_rounds", None)
    if not rounds or getattr(req, "c2kv_round_idx", 0) != 0:
        return None, None, "not_first_round"
    if (
        getattr(req, "session", None) is not None
        or getattr(req, "history_kv_eviction", None) is not None
        or getattr(req, "input_embeds", None) is not None
        or getattr(req, "multimodal_inputs", None) is not None
        or getattr(req, "token_type_ids", None) is not None
        or getattr(req, "c2kv_use_gist_projection", False)
        or getattr(req, "c2kv_gist_seen", False)
        or getattr(req, "c2kv_layout", None)
        or getattr(req, "c2kv_position_correction", 0) != 0
        or getattr(req, "c2kv_round_start_len", 0) != 0
        or getattr(req, "output_ids", None)
        or getattr(req, "session_cache_closed_during_request", False)
        or getattr(req, "is_retracted", False)
    ):
        return None, None, "modified_or_stateful_request"

    raw_ids = list(rounds[0].tokens)
    origin_ids = getattr(req, "origin_input_ids", None)
    virtual_ids = getattr(req, "c2kv_virtual_input_ids", None)
    segments = getattr(req, "c2kv_segments", None)
    if (
        not raw_ids
        or not segments
        or origin_ids is None
        or virtual_ids is None
        or list(origin_ids[: len(raw_ids)]) != raw_ids
        or list(virtual_ids[: len(raw_ids)]) != raw_ids
        or min(seg.token_start for seg in segments) < len(raw_ids)
        or list(getattr(req, "origin_input_ids_unpadded", origin_ids))
        != list(origin_ids)
    ):
        return None, None, "not_initial_raw_tokens"
    if getattr(req, "kv_committed_freed", False):
        return None, None, "kv_already_freed"
    return raw_ids, radix_cache, None


def _report(req, *, status, reason=None, hit_tokens=None, inserted_tokens=None, prefix_tokens=None):
    report = getattr(req, "c2kv_raw_prefix_cache", None)
    if report is None:
        report = {"enabled": True, "hit_tokens": 0, "inserted_tokens": 0}
        req.c2kv_raw_prefix_cache = report
    report["status"] = status
    report["reason"] = reason
    if hit_tokens is not None:
        report["hit_tokens"] = hit_tokens
    if inserted_tokens is not None:
        report["inserted_tokens"] = inserted_tokens
    if prefix_tokens is not None:
        report["prefix_tokens"] = prefix_tokens


def match_c2kv_first_raw_prefix(req, tree_cache) -> bool:
    """Match an initial C2KV round when FCFS does not do prefix matching."""
    if not native_raw_prefix_cache_enabled():
        return False
    raw_ids, _, reason = _candidate(req, tree_cache)
    if reason is not None:
        _report(req, status="skipped", reason=reason)
        return False
    if req.kv_committed_len != 0:
        return False
    # Keep one real token for the normal prefill path. An exact full hit would
    # otherwise create a zero-token forward and bypass the injection boundary.
    result = tree_cache.match_prefix(
        MatchPrefixParams(key=RadixKey(raw_ids[:-1], req.extra_key), req=req)
    )
    req.prefix_indices = result.device_indices
    req.last_node = result.last_device_node
    req.last_host_node = result.last_host_node
    req.host_hit_length = result.host_hit_length
    req.c2kv_tree_cache_prefix_len = (
        result.cache_protected_len
        if result.cache_protected_len is not None
        else len(result.device_indices)
    )
    req.cache_protected_len = req.c2kv_tree_cache_prefix_len
    _report(
        req,
        status="matched",
        hit_tokens=len(result.device_indices),
        prefix_tokens=len(raw_ids),
    )
    return True


def cache_c2kv_first_raw_prefix(req, tree_cache) -> bool:
    """Transfer first-round KV to RadixCache before any synthetic KV appears."""
    if not native_raw_prefix_cache_enabled():
        return False
    raw_ids, radix_cache, reason = _candidate(req, tree_cache)
    if reason is not None:
        _report(req, status="skipped", reason=reason)
        return False
    raw_len = len(raw_ids)
    if (
        req.req_pool_idx is None
        or req.last_node is None
        or req.kv_committed_len != raw_len
        or req.kv_allocated_len != raw_len
    ):
        _report(req, status="skipped", reason="first_round_not_fully_committed")
        return False
    old_protected_len = getattr(req, "c2kv_tree_cache_prefix_len", 0)
    if not 0 <= old_protected_len <= raw_len:
        _report(req, status="skipped", reason="invalid_existing_prefix")
        return False

    kv_indices = tree_cache.req_to_token_pool.req_to_token[
        req.req_pool_idx, :raw_len
    ]
    key = RadixKey(raw_ids, req.extra_key)
    result = radix_cache.insert(
        InsertParams(
            key=key,
            value=kv_indices.to(dtype=torch.int64, copy=True),
            priority=getattr(req, "priority", 0) or 0,
        )
    )
    duplicate_len = result.prefix_len
    assert old_protected_len <= duplicate_len <= raw_len
    match = radix_cache.match_prefix(MatchPrefixParams(key=key))
    assert len(match.device_indices) == raw_len
    if duplicate_len > old_protected_len:
        tree_cache.token_to_kv_pool_allocator.free(
            kv_indices[old_protected_len:duplicate_len]
        )
        tree_cache.req_to_token_pool.write(
            (req.req_pool_idx, slice(old_protected_len, duplicate_len)),
            match.device_indices[old_protected_len:duplicate_len],
        )

    if match.last_device_node is not req.last_node:
        tree_cache.inc_lock_ref(match.last_device_node)
        tree_cache.dec_lock_ref(req.last_node)
    req.last_node = match.last_device_node
    req.last_host_node = match.last_host_node
    req.prefix_indices = match.device_indices
    req.cache_protected_len = raw_len
    req.c2kv_tree_cache_prefix_len = raw_len
    req.c2kv_raw_prefix_lock_held = True
    _report(
        req,
        status="cached",
        hit_tokens=old_protected_len,
        inserted_tokens=raw_len - duplicate_len,
        prefix_tokens=raw_len,
    )
    return True
