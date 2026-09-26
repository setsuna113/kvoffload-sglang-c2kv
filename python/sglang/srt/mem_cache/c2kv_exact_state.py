"""Exact quiescent snapshots for an isolated, eager, single-worker T02 engine.

Only allocated KV is model state. Completed native requests own no decoder KV;
their gist pool remains live and is copied byte-for-byte, including positions,
allocator ordering and LRU metadata. Unallocated backing bytes are not replayed.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import random
import uuid
from collections import OrderedDict

import numpy as np
import torch


SCHEMA = "c2kv-exact-backend-state-v1"
ALLOCATOR_FIELDS = ("free_pages", "release_pages", "is_not_in_free_group", "free_group")
COUNTERS = (
    "forward_ct", "forward_ct_decode", "batch_record_ct", "num_generated_tokens",
    "num_retracted_reqs", "num_paused_reqs", "new_token_ratio",
    "spec_num_accepted_tokens", "spec_num_forward_ct",
    "spec_total_num_accepted_tokens", "spec_total_num_forward_ct",
    "last_decode_stats_tic", "last_prefill_stats_tic", "last_prefill_tokens",
    "last_gen_throughput", "last_input_throughput", "step_time_dict", "stats",
    "_c2kv_runtime_peak_main_kv_slots", "_c2kv_runtime_peak_c2kv_pool_slots",
    "_c2kv_runtime_peak_total_gpu_kv_bytes",
)


def _plain(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "bytes_sha256": hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported exact-state value {type(value).__name__}")


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if dataclasses.is_dataclass(value):
        return sum(_tensor_bytes(getattr(value, f.name)) for f in dataclasses.fields(value))
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _clone(value, device="cpu"):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return type(value)((k, _clone(v, device)) for k, v in value.items()) if type(value) in (dict, OrderedDict) else copy.deepcopy(value)
    if isinstance(value, list):
        return [_clone(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone(v, device) for v in value)
    return copy.deepcopy(value)


def _allocator_state(allocator):
    return {name: _clone(getattr(allocator, name)) for name in ALLOCATOR_FIELDS}


def _restore_allocator(allocator, state):
    for name, value in state.items():
        setattr(allocator, name, _clone(value, allocator.device))


class ExactStateStore:
    """Bounded checkpoints; immutable host copies never alias live pools."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.snapshots = {}
        self.max_snapshots = int(os.environ.get("C2KV_EXACT_MAX_SNAPSHOTS", "128"))
        self.max_host_bytes = int(os.environ.get("C2KV_EXACT_MAX_HOST_BYTES", str(4 * 1024**3)))
        if self.max_snapshots <= 0 or self.max_host_bytes <= 0:
            raise ValueError("Exact snapshot resource caps must be positive")

    def _device_api(self):
        device_type = torch.device(self.scheduler.c2kv_pool.device).type
        if device_type == "cpu":
            return None
        api = getattr(torch, device_type, None)
        if api is None or not all(hasattr(api, name) for name in ("synchronize", "get_rng_state", "set_rng_state")):
            raise ValueError(f"Exact snapshot lacks RNG API for {device_type}")
        return api

    def _validate(self):
        s = self.scheduler
        args = s.server_args
        if os.environ.get("C2KV_ENABLE_EXACT_STATE") != "1":
            raise ValueError("Exact state requires C2KV_ENABLE_EXACT_STATE=1 on an isolated engine")
        if s.c2kv_pool is None:
            raise ValueError("Exact state requires a C2KV pool")
        for name in ("tp_size", "dp_size", "pp_size"):
            if getattr(args, name, 1) != 1:
                raise ValueError(f"Exact state requires {name}=1")
        for name in ("disable_radix_cache", "disable_overlap_schedule", "disable_cuda_graph"):
            if getattr(args, name, False) is not True:
                raise ValueError(f"Exact state requires {name}=True")
        for name in ("enable_hierarchical_cache", "enable_hisparse", "enable_metrics", "enable_lora"):
            if getattr(args, name, False):
                raise ValueError(f"Exact state does not support {name}")
        if getattr(args, "speculative_algorithm", None) is not None:
            raise ValueError("Exact state requires ordinary autoregressive decoding")
        if getattr(args, "disaggregation_mode", "null") not in (None, "null"):
            raise ValueError("Exact state requires no disaggregation")
        if not s.is_fully_idle():
            raise ValueError("Exact snapshot/restore requires a fully idle scheduler")
        if type(s.tree_cache).__name__ != "ChunkCache":
            raise ValueError("Exact state supports only the no-prefix ChunkCache")
        if s.token_to_kv_pool_allocator.available_size() != s.token_to_kv_pool_allocator.size:
            raise ValueError("Completed actor still owns decoder KV; refusing an empty-state claim")
        if len(s.req_to_token_pool.free_slots) != s.req_to_token_pool.size:
            raise ValueError("Completed actor still owns a request slot")
        if s.c2kv_pool._pin_counts:
            raise ValueError("Completed actor still owns live gist pins")
        if getattr(s.session_controller, "sessions", {}):
            raise ValueError("Exact state does not support native persistent sessions")
        api = self._device_api()
        if api is not None:
            api.synchronize()

    def _read(self):
        s = self.scheduler
        pool = s.c2kv_pool
        entries = []
        for key, entry in pool._cache.items():
            cloned_entry = copy.deepcopy(entry)
            cloned_entry.token_indices = _clone(entry.token_indices)
            entries.append((key, cloned_entry))
        indices = torch.cat([entry.token_indices for _, entry in entries]) if entries else torch.empty(0, dtype=torch.int64)
        device_indices = indices.to(pool.device)
        kv = [(_clone(pool.kv_cache.get_key_buffer(pool.start_layer + layer)[device_indices]),
               _clone(pool.kv_cache.get_value_buffer(pool.start_layer + layer)[device_indices]))
              for layer in range(pool.num_layers)]
        api = self._device_api()
        return {
            "actor_kv": {"decoder_kv": None, "entries": entries, "kv": kv},
            "actor_positions": {"decoder_positions": None,
                "indices": indices, "gist_positions": _clone(pool.position_buffer[device_indices])},
            "backend_stats": {
                "gist_allocator": _allocator_state(pool.allocator),
                "main_allocator": _allocator_state(s.token_to_kv_pool_allocator),
                "request_free_slots": copy.deepcopy(s.req_to_token_pool.free_slots),
                "current_tokens": pool._current_tokens, "pin_counts": dict(pool._pin_counts),
                "counters": {name: copy.deepcopy(getattr(s, name)) for name in COUNTERS if hasattr(s, name)},
            },
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state().clone(),
                "torch_device": None if api is None else api.get_rng_state().cpu().clone()},
        }

    def _receipt(self, snapshot_id, state, operation):
        return {"schema": SCHEMA, "snapshot_id": snapshot_id, "operation": operation,
            "exact": True, "component_digests": {name: _digest(value) for name, value in state.items()},
            "decoder_kv": None, "decoder_positions": None,
            "gist_entries": len(state["actor_kv"]["entries"]),
            "gist_tokens": state["backend_stats"]["current_tokens"],
            "snapshot_tensor_bytes": _tensor_bytes(state),
            "storage": "independent_cpu_tensor_copies",
            "stats_scope": "scheduler_counters_and_allocated_kv_pools",
            "excluded_telemetry": ["wall_clock", "framework_reserved_memory", "unallocated_backing_bytes"]}

    def execute(self, operation, snapshot_id=None):
        self._validate()
        if operation == "capture":
            if len(self.snapshots) >= self.max_snapshots:
                raise ValueError("Exact snapshot count cap exhausted; release saved states")
            snapshot_id = uuid.uuid4().hex
            state = self._read()
            receipt = self._receipt(snapshot_id, state, operation)
            retained_bytes = sum(item[1]["snapshot_tensor_bytes"] for item in self.snapshots.values())
            if retained_bytes + receipt["snapshot_tensor_bytes"] > self.max_host_bytes:
                raise ValueError("Exact snapshot host tensor byte cap exhausted; release saved states")
            self.snapshots[snapshot_id] = (state, receipt)
            return receipt
        if snapshot_id not in self.snapshots:
            raise ValueError("Unknown exact snapshot ID")
        state, receipt = self.snapshots[snapshot_id]
        if operation == "release":
            del self.snapshots[snapshot_id]
            return {"schema": SCHEMA, "snapshot_id": snapshot_id, "operation": "release", "released": True}
        if operation != "restore":
            raise ValueError("Exact state operation must be capture, restore, or release")
        s = self.scheduler
        pool = s.c2kv_pool
        indices = state["actor_positions"]["indices"].to(pool.device)
        for layer, (key, value) in enumerate(state["actor_kv"]["kv"]):
            pool.kv_cache.get_key_buffer(pool.start_layer + layer)[indices] = key.to(pool.device)
            pool.kv_cache.get_value_buffer(pool.start_layer + layer)[indices] = value.to(pool.device)
        pool.position_buffer[indices] = state["actor_positions"]["gist_positions"].to(pool.device)
        pool._cache = OrderedDict()
        for key, entry in state["actor_kv"]["entries"]:
            restored_entry = copy.deepcopy(entry)
            restored_entry.token_indices = entry.token_indices.to(device=pool.device, copy=True)
            pool._cache[key] = restored_entry
        stats = state["backend_stats"]
        pool._current_tokens = stats["current_tokens"]
        pool._pin_counts.clear()
        pool._pin_counts.update(stats["pin_counts"])
        _restore_allocator(pool.allocator, stats["gist_allocator"])
        _restore_allocator(s.token_to_kv_pool_allocator, stats["main_allocator"])
        s.req_to_token_pool.free_slots = copy.deepcopy(stats["request_free_slots"])
        for name in COUNTERS:
            if name in stats["counters"]:
                setattr(s, name, copy.deepcopy(stats["counters"][name]))
            elif hasattr(s, name):
                delattr(s, name)
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch_cpu"])
        api = self._device_api()
        if api is not None:
            api.set_rng_state(state["rng"]["torch_device"])
            api.synchronize()
        verified = self._receipt(snapshot_id, self._read(), operation)
        if verified["component_digests"] != receipt["component_digests"]:
            raise RuntimeError("Live state differs after exact backend restoration")
        verified["verified_from_live_state"] = True
        return verified
