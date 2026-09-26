"""CPU tests exercise actual tensor restoration without importing a GPU server."""

import dataclasses
import importlib.util
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


_PATH = Path(__file__).resolve().parents[3] / "python/sglang/srt/mem_cache/c2kv_exact_state.py"
_SPEC = importlib.util.spec_from_file_location("_c2kv_exact_state_test", _PATH)
module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = module
_SPEC.loader.exec_module(module)


@dataclasses.dataclass
class Entry:
    key_hash: str
    token_indices: torch.Tensor
    gist_len: int


class Allocator:
    device = "cpu"
    size = 6

    def __init__(self, free):
        self.free_pages = torch.tensor(free)
        self.release_pages = torch.tensor([], dtype=torch.int64)
        self.is_not_in_free_group = True
        self.free_group = []

    def available_size(self):
        return len(self.free_pages) + len(self.release_pages)


def scheduler():
    keys = [torch.arange(14, dtype=torch.float32).reshape(7, 1, 2)]
    values = [keys[0] + 50]
    pool = SimpleNamespace(
        device="cpu", start_layer=0, num_layers=1,
        kv_cache=SimpleNamespace(get_key_buffer=lambda i: keys[i], get_value_buffer=lambda i: values[i]),
        position_buffer=torch.arange(7, dtype=torch.int64) * 3,
        _cache=OrderedDict((name, Entry(name, torch.tensor([index]), 1)) for name, index in (("first", 2), ("second", 4))),
        _current_tokens=2, _pin_counts=Counter(), allocator=Allocator([1, 3, 5, 6]),
    )
    return SimpleNamespace(
        server_args=SimpleNamespace(disable_radix_cache=True, disable_overlap_schedule=True, disable_cuda_graph=True),
        c2kv_pool=pool, tree_cache=type("ChunkCache", (), {})(),
        req_to_token_pool=SimpleNamespace(size=2, free_slots=[1, 0]),
        token_to_kv_pool_allocator=Allocator([6, 5, 4, 3, 2, 1]),
        session_controller=SimpleNamespace(sessions={}), is_fully_idle=lambda: True,
        forward_ct=8, _c2kv_runtime_peak_total_gpu_kv_bytes=200,
    )


def test_exact_restore_replaces_changed_kv_positions_lru_allocator_counters_rng(monkeypatch):
    monkeypatch.setenv("C2KV_ENABLE_EXACT_STATE", "1")
    s = scheduler()
    store = module.ExactStateStore(s)
    before = store.execute("capture")
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(3))
    s.c2kv_pool.kv_cache.get_key_buffer(0).fill_(-11)
    s.c2kv_pool.kv_cache.get_value_buffer(0).fill_(-12)
    s.c2kv_pool.position_buffer.fill_(99)
    s.c2kv_pool._cache.clear()
    s.c2kv_pool._current_tokens = 0
    s.c2kv_pool.allocator.free_pages = torch.arange(1, 7)
    s.token_to_kv_pool_allocator.free_pages = torch.arange(1, 7)
    s.req_to_token_pool.free_slots.reverse()
    s.forward_ct = 90
    s._c2kv_runtime_peak_total_gpu_kv_bytes = 400
    after = store.execute("restore", before["snapshot_id"])
    assert after["component_digests"] == before["component_digests"]
    assert after["verified_from_live_state"] is True
    assert after["decoder_kv"] is None
    assert list(s.c2kv_pool._cache) == ["first", "second"]
    assert s.c2kv_pool.kv_cache.get_key_buffer(0)[2].flatten().tolist() == [4, 5]
    assert s.c2kv_pool.kv_cache.get_value_buffer(0)[4].flatten().tolist() == [58, 59]
    assert s.c2kv_pool.position_buffer[4] == 12
    assert s.c2kv_pool.allocator.free_pages.tolist() == [1, 3, 5, 6]
    assert s.token_to_kv_pool_allocator.free_pages.tolist() == [6, 5, 4, 3, 2, 1]
    assert s.forward_ct == 8
    assert random.random() == expected_rng[0]
    assert np.random.rand() == expected_rng[1]
    assert torch.equal(torch.rand(3), expected_rng[2])
    # Repeated restoration uses the original immutable CPU snapshot.
    again = store.execute("restore", before["snapshot_id"])
    assert again["component_digests"] == before["component_digests"]
    store.execute("release", before["snapshot_id"])
    with pytest.raises(ValueError, match="Unknown"):
        store.execute("restore", before["snapshot_id"])


@pytest.mark.parametrize("change,pattern", [
    (lambda s: setattr(s.server_args, "disable_radix_cache", False), "disable_radix"),
    (lambda s: setattr(s.server_args, "tp_size", 2), "tp_size"),
    (lambda s: setattr(s, "is_fully_idle", lambda: False), "idle"),
    (lambda s: setattr(s.token_to_kv_pool_allocator, "free_pages", torch.tensor([1])), "decoder KV"),
    (lambda s: s.c2kv_pool._pin_counts.update({"first": 1}), "pins"),
])
def test_rejects_unrepresented_live_state(monkeypatch, change, pattern):
    monkeypatch.setenv("C2KV_ENABLE_EXACT_STATE", "1")
    s = scheduler()
    change(s)
    with pytest.raises(ValueError, match=pattern):
        module.ExactStateStore(s).execute("capture")


def test_snapshot_is_opt_in(monkeypatch):
    monkeypatch.delenv("C2KV_ENABLE_EXACT_STATE", raising=False)
    with pytest.raises(ValueError, match="C2KV_ENABLE_EXACT_STATE"):
        module.ExactStateStore(scheduler()).execute("capture")


def test_multiple_snapshots_restore_distinct_states_with_explicit_caps(monkeypatch):
    monkeypatch.setenv("C2KV_ENABLE_EXACT_STATE", "1")
    monkeypatch.setenv("C2KV_EXACT_MAX_SNAPSHOTS", "2")
    s = scheduler()
    store = module.ExactStateStore(s)
    first = store.execute("capture")
    s.c2kv_pool.kv_cache.get_key_buffer(0)[2].fill_(99)
    second = store.execute("capture")
    assert first["component_digests"]["actor_kv"] != second["component_digests"]["actor_kv"]
    assert first["snapshot_tensor_bytes"] > 0
    with pytest.raises(ValueError, match="count cap"):
        store.execute("capture")
    store.execute("restore", first["snapshot_id"])
    assert s.c2kv_pool.kv_cache.get_key_buffer(0)[2].flatten().tolist() == [4, 5]
    store.execute("restore", second["snapshot_id"])
    assert s.c2kv_pool.kv_cache.get_key_buffer(0)[2].flatten().tolist() == [99, 99]
    monkeypatch.setenv("C2KV_EXACT_MAX_HOST_BYTES", "1")
    limited = module.ExactStateStore(s)
    with pytest.raises(ValueError, match="host tensor byte cap"):
        limited.execute("capture")
    assert not limited.snapshots
