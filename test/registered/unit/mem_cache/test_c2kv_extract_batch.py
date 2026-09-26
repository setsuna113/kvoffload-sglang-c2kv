"""CPU contracts for the opt-in C2KV packed extraction scheduler."""

import ast
import logging
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/scheduler.py"


def _output(**kwargs):
    return SimpleNamespace(
        **{
            "items": [],
            "key_hash": "",
            "gist_len": 0,
            "cache_hit": False,
            "success": True,
            "error": "",
            "gist_generation_duration_ns": None,
            "extraction_batch_id": None,
            "extraction_batch_size": None,
            "shared_gist_generation_duration_ns": None,
            **kwargs,
        }
    )


def _methods():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {
        "_c2kv_extract_cache_key",
        "handle_c2kv_bulk_cache_lookup",
        "handle_c2kv_extract_batch",
        "_run_c2kv_extract_group",
        "handle_extract_request",
    }
    methods = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "Scheduler"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    for method in methods:
        method.decorator_list = []
        if method.name == "handle_extract_request":
            method.body = [
                node for node in method.body if not isinstance(node, ast.ImportFrom)
            ]
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))

    class FakeTorch:
        long = int
        bool = bool
        distributed = SimpleNamespace(is_initialized=lambda: False)

        @staticmethod
        def tensor(values, **_kwargs):
            return SimpleNamespace(shape=(len(values), len(values[0])))

        @staticmethod
        def ones_like(value, **_kwargs):
            return SimpleNamespace(shape=value.shape)

    class C2KVBulkCacheLookupReqInput:
        def __init__(self, items, *, rid, materialize_first_miss=True):
            self.items = items
            self.rid = rid
            self.materialize_first_miss = materialize_first_miss

    namespace = {
        "C2KVExtractBatchReqInput": lambda **kwargs: SimpleNamespace(**kwargs),
        "C2KVExtractBatchReqOutput": _output,
        "C2KVExtractReqOutput": _output,
        "C2KVBulkCacheLookupReqInput": C2KVBulkCacheLookupReqInput,
        "C2KVBulkCacheLookupReqOutput": _output,
        "TokenizedExtractReqInput": object,
        "torch": FakeTorch,
        "time": time,
        "uuid": uuid,
        "logger": logging.getLogger(__name__),
        "paper_telemetry": SimpleNamespace(enabled=lambda: False),
        "_is_npu": False,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102
    return namespace


class Qwen3ForCausalLM:
    full_length_pic = False
    c2kv_tool_gist_identity = "tool-checkpoint"


class Pool:
    def __init__(self, capacity=16, per_entry=16, hits=(), pinned=()):
        self.max_total_tokens = capacity
        self.max_entry_tokens = per_entry
        self._cache = OrderedDict(
            (key, SimpleNamespace(gist_len=1, original_seq_len=4)) for key in hits
        )
        self.pinned = set(pinned)
        self.events = []
        self.allocator = SimpleNamespace(available_size=self.available_size)

    def available_size(self):
        return self.max_total_tokens - sum(x.gist_len for x in self._cache.values())

    def compute_hash(self, ids, *, compression_ratio, extractor_config):
        projection = extractor_config.get("projection_set", "history")
        return f"{projection}:{compression_ratio}:{ids[0]}"

    def get(self, key):
        self.events.append(("get", key))
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
        return entry

    def can_allocate(self, count, *, existing_key):
        available = self.available_size() + sum(
            entry.gist_len
            for key, entry in self._cache.items()
            if key not in self.pinned and key != existing_key
        )
        return count <= available

    def store(self, *, key_hash, gist_mask, original_seq_len, **_kwargs):
        gist_len = gist_mask.shape[1]
        while self.available_size() < gist_len:
            for key in self._cache:
                if key not in self.pinned:
                    self.events.append(("evict", key))
                    del self._cache[key]
                    break
            else:
                raise ValueError("no unpinned space")
        entry = SimpleNamespace(gist_len=gist_len, original_seq_len=original_seq_len)
        self._cache[key_hash] = entry
        self.events.append(("store", key_hash))
        return entry


def _item(token, *, ratio=4, projection="history", allowed=True):
    return SimpleNamespace(
        rid=f"r{token}",
        input_ids=[token] * 4,
        compression_ratio=ratio,
        projection_set=projection,
        allow_cache_miss=allowed,
        c2kv_outer_request_id="outer",
        c2kv_measurement_phase="prewarm",
    )


def _scheduler(pool, *, model=None, fail=False, tp_size=1):
    namespace = _methods()
    model = model or Qwen3ForCausalLM()
    forwards = []
    logs = []

    def result(item, ratio):
        return (
            object(),
            SimpleNamespace(shape=(1, (len(item) + ratio - 1) // ratio)),
            object(),
        )

    def many(ids, ratio, *, projection_set):
        forwards.append(("many", [part[0] for part in ids], ratio, projection_set))
        if fail:
            raise RuntimeError("packed failed")
        return [result(item, ratio) for item in ids]

    def one(ids, _mask, ratio, *, projection_set):
        token = int(ids.token)
        forwards.append(("one", [token], ratio, projection_set))
        return result([token] * 4, ratio)

    class FakeTorch:
        long = int
        bool = bool
        distributed = SimpleNamespace(is_initialized=lambda: False)

        @staticmethod
        def tensor(values, **_kwargs):
            return SimpleNamespace(token=values[0][0], shape=(1, len(values[0])))

        @staticmethod
        def ones_like(value, **_kwargs):
            return SimpleNamespace(shape=value.shape)

    namespace["handle_extract_request"].__globals__["torch"] = FakeTorch
    scheduler = SimpleNamespace(
        c2kv_pool=pool,
        tp_size=tp_size,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=model,
                get_c2kv_compression_ratio=int,
                forward_c2kv_extract_many=many,
                forward_c2kv_extract=one,
            )
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        server_args=SimpleNamespace(),
        enable_overlap=False,
        forward_stream=None,
        _log_c2kv_token_usage=lambda name, **kwargs: logs.append((name, kwargs)),
    )
    for name in (
        "_c2kv_extract_cache_key",
        "handle_c2kv_bulk_cache_lookup",
        "handle_c2kv_extract_batch",
        "_run_c2kv_extract_group",
        "handle_extract_request",
    ):
        setattr(scheduler, name, MethodType(namespace[name], scheduler))
    return scheduler, forwards, logs


def _bulk(scheduler, rid, *tokens, materialize=True):
    cls = scheduler.handle_c2kv_extract_batch.__func__.__globals__["C2KVBulkCacheLookupReqInput"]
    return cls(
        [_item(token) for token in tokens],
        rid=rid,
        materialize_first_miss=materialize,
    )


def test_packed_misses_keep_fifo_hit_touch_duplicate_and_logical_calls():
    pool = Pool(hits=("history:4:9",))
    scheduler, forwards, logs = _scheduler(pool)
    items = [_item(1), _item(2), _item(9), _item(3), _item(3), _item(4)]
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=items))
    assert out.success
    assert [part.key_hash for part in out.items] == [
        "history:4:1",
        "history:4:2",
        "history:4:9",
        "history:4:3",
        "history:4:3",
        "history:4:4",
    ]
    assert forwards == [
        ("many", [1, 2], 4, "history"),
        ("one", [3], 4, "history"),
        ("one", [4], 4, "history"),
    ]
    assert out.items[2].cache_hit and out.items[4].cache_hit
    assert [name for name, _ in logs].count("extract_request") == len(items)
    assert [name for name, _ in logs].count("extract_store") == 4
    first, second = out.items[:2]
    assert first.extraction_batch_id == second.extraction_batch_id
    assert first.extraction_batch_id and first.extraction_batch_size == 2
    assert first.gist_generation_duration_ns is None
    assert second.gist_generation_duration_ns is None
    assert out.items[2].extraction_batch_id is None
    assert pool.events[:5] == [
        ("get", "history:4:1"),
        ("store", "history:4:1"),
        ("get", "history:4:2"),
        ("store", "history:4:2"),
        ("get", "history:4:9"),
    ]


def test_barriers_split_ratio_projection_and_forbidden_miss():
    scheduler, forwards, _ = _scheduler(Pool())
    items = [
        _item(1),
        _item(2),
        _item(3, ratio=8),
        _item(4, ratio=8),
        _item(5, ratio=8, allowed=False),
        _item(6, projection="tool"),
        _item(7, projection="tool"),
    ]
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=items))
    assert out.success and not out.error
    assert "C2KV_EXTRACTION_BUDGET_EXHAUSTED" in out.items[4].error
    assert all(part.success for i, part in enumerate(out.items) if i != 4)
    assert forwards == [
        ("many", [1, 2], 4, "history"),
        ("many", [3, 4], 8, "history"),
        ("many", [6, 7], 4, "tool"),
    ]
    assert len({out.items[i].extraction_batch_id for i in (0, 2, 5)}) == 3


@pytest.mark.parametrize("capacity,per_entry,pinned", [(2, 16, ()), (16, 0, ())])
def test_capacity_or_entry_cap_uses_single_item_path(capacity, per_entry, pinned):
    pool = Pool(
        capacity=capacity, per_entry=per_entry, hits=("history:4:9",), pinned=pinned
    )
    scheduler, forwards, _ = _scheduler(pool)
    out = scheduler.handle_c2kv_extract_batch(
        SimpleNamespace(items=[_item(1), _item(2)])
    )
    assert all(kind == "one" for kind, *_ in forwards) or not forwards
    assert all(part.extraction_batch_id is None for part in out.items)


def test_pinned_capacity_fails_without_packed_forward_or_eviction():
    pool = Pool(capacity=1, hits=("history:4:9",), pinned=("history:4:9",))
    scheduler, forwards, _ = _scheduler(pool)
    out = scheduler.handle_c2kv_extract_batch(
        SimpleNamespace(items=[_item(1), _item(2)])
    )
    assert out.success and not forwards
    assert all(not part.success for part in out.items)
    assert all("unpinned space" in part.error for part in out.items)
    assert not any(event[0] == "evict" for event in pool.events)


def test_packed_model_error_fails_each_item_without_single_retry():
    scheduler, forwards, _ = _scheduler(Pool(), fail=True)
    out = scheduler.handle_c2kv_extract_batch(
        SimpleNamespace(items=[_item(1), _item(2)])
    )
    assert out.success and [part.error for part in out.items] == [
        "packed failed",
        "packed failed",
    ]
    assert forwards == [("many", [1, 2], 4, "history")]
    assert all(part.extraction_batch_size == 2 for part in out.items)
    assert all(part.gist_generation_duration_ns is None for part in out.items)


def test_tp_or_non_qwen_falls_back_to_fifo_singles():
    for tp_size, model in [(2, Qwen3ForCausalLM()), (1, object())]:
        scheduler, forwards, _ = _scheduler(Pool(), tp_size=tp_size, model=model)
        out = scheduler.handle_c2kv_extract_batch(
            SimpleNamespace(items=[_item(1), _item(2)])
        )
        assert out.success
        assert [kind for kind, *_ in forwards] == ["one", "one"]


def test_document_and_raw_token_limits_split_groups():
    scheduler, forwards, _ = _scheduler(Pool(capacity=10000, per_entry=10000))
    items = [_item(token) for token in range(1, 6)]
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=items))
    assert out.success
    assert forwards == [
        ("many", [1, 2, 3, 4], 4, "history"),
        ("one", [5], 4, "history"),
    ]

    scheduler, forwards, _ = _scheduler(Pool(capacity=10000, per_entry=10000))
    items = [_item(6), _item(7)]
    items[0].input_ids = [6] * 2049
    items[1].input_ids = [7] * 2049
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=items))
    assert out.success
    assert [kind for kind, *_ in forwards] == ["one", "one"]


def test_shared_timing_and_overlap_wait_are_group_scoped():
    scheduler, forwards, _ = _scheduler(Pool())
    events = []
    scheduler.enable_overlap = True
    scheduler.forward_stream = object()
    scheduler.schedule_stream = SimpleNamespace(
        wait_stream=lambda stream: events.append(("wait", stream)),
        synchronize=lambda: events.append(("sync", None)),
    )
    scheduler._run_c2kv_extract_group.__func__.__globals__["paper_telemetry"] = (
        SimpleNamespace(enabled=lambda: True)
    )
    out = scheduler.handle_c2kv_extract_batch(
        SimpleNamespace(items=[_item(1), _item(2)])
    )
    assert out.success and forwards == [("many", [1, 2], 4, "history")]
    assert events == [
        ("wait", scheduler.forward_stream),
        ("sync", None),
        ("sync", None),
    ]
    assert out.items[0].shared_gist_generation_duration_ns is not None
    assert (
        out.items[0].shared_gist_generation_duration_ns
        == out.items[1].shared_gist_generation_duration_ns
    )
    assert all(part.gist_generation_duration_ns is None for part in out.items)


def test_two_bulk_first_misses_and_ordinary_miss_share_one_forward():
    pool = Pool(hits=("history:4:9", "history:4:8"))
    scheduler, forwards, logs = _scheduler(pool)
    requests = [
        _bulk(scheduler, "bulk-a", 9, 1, 7),
        _bulk(scheduler, "bulk-b", 8, 2, 6),
        _item(3),
    ]
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=requests))
    assert out.success and len(out.items) == 3
    assert forwards == [("many", [1, 2, 3], 4, "history")]
    first, second, ordinary = out.items
    assert [first.rid, second.rid] == ["bulk-a", "bulk-b"]
    assert [first.first_miss_index, second.first_miss_index] == [1, 1]
    assert [x.key_hash for x in first.hits] == ["history:4:9"]
    assert [x.key_hash for x in second.hits] == ["history:4:8"]
    assert all(x.cache_hit for x in first.hits + second.hits)
    assert [first.first_miss_result.key_hash, second.first_miss_result.key_hash] == [
        "history:4:1",
        "history:4:2",
    ]
    assert ordinary.key_hash == "history:4:3"
    assert first.first_miss_result.extraction_batch_size == 3
    assert second.first_miss_result.extraction_batch_size == 3
    assert pool.events == [
        ("get", "history:4:9"),
        ("get", "history:4:1"),
        ("store", "history:4:1"),
        ("get", "history:4:8"),
        ("get", "history:4:2"),
        ("store", "history:4:2"),
        ("get", "history:4:3"),
        ("store", "history:4:3"),
    ]
    assert [name for name, _ in logs].count("extract_request") == 5
    assert "history:4:7" not in pool._cache
    assert "history:4:6" not in pool._cache


def test_isolated_and_all_hit_bulk_use_original_fusion_path():
    pool = Pool(hits=("history:4:9",))
    scheduler, forwards, _ = _scheduler(pool)
    first = _bulk(scheduler, "first", 9, 1, 2)
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=[first]))
    assert out.items[0].rid == "first"
    assert out.items[0].first_miss_index == 1
    assert out.items[0].first_miss_result.key_hash == "history:4:1"
    assert forwards == [("one", [1], 4, "history")]

    all_hit = _bulk(scheduler, "all-hit", 9, 1)
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=[all_hit]))
    assert out.items[0].rid == "all-hit"
    assert out.items[0].first_miss_index == 2
    assert out.items[0].first_miss_result is None
    assert len(out.items[0].hits) == 2
    assert forwards == [("one", [1], 4, "history")]


def test_bulk_budget_rejection_is_per_item_and_does_not_block_next_bulk():
    pool = Pool(hits=("history:4:9", "history:4:8"))
    scheduler, forwards, _ = _scheduler(pool)
    denied = _bulk(scheduler, "denied", 9, 1)
    denied.items[1].allow_cache_miss = False
    allowed = _bulk(scheduler, "allowed", 8, 2)
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=[denied, allowed]))
    assert out.items[0].rid == "denied"
    assert out.items[0].hits[0].cache_hit
    assert not out.items[0].first_miss_result.success
    assert "C2KV_EXTRACTION_BUDGET_EXHAUSTED" in out.items[0].first_miss_result.error
    assert out.items[1].rid == "allowed"
    assert out.items[1].first_miss_result.success
    assert forwards == [("one", [2], 4, "history")]


def test_duplicate_first_miss_key_falls_back_in_original_order():
    pool = Pool(hits=("history:4:9", "history:4:8"))
    scheduler, forwards, _ = _scheduler(pool)
    first = _bulk(scheduler, "first", 9, 1)
    second = _bulk(scheduler, "second", 8, 1)
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=[first, second]))
    assert forwards == [("one", [1], 4, "history")]
    assert out.items[0].first_miss_index == 1
    assert out.items[1].first_miss_index == 2
    assert out.items[1].first_miss_result is None
    assert [hit.key_hash for hit in out.items[1].hits] == [
        "history:4:8",
        "history:4:1",
    ]


def test_no_early_prefix_touch_when_packed_capacity_is_insufficient():
    pool = Pool(capacity=2, hits=("history:4:9", "history:4:8"))
    scheduler, forwards, _ = _scheduler(pool)
    first = _bulk(scheduler, "first", 9, 1)
    second = _bulk(scheduler, "second", 8, 2)
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=[first, second]))
    assert [kind for kind, *_ in forwards] == ["one", "one"]
    assert out.items[0].first_miss_index == 1
    assert out.items[1].first_miss_index == 0
    assert pool.events.index(("evict", "history:4:8")) < pool.events.index(
        ("get", "history:4:8")
    )


def test_packed_bulk_failure_preserves_prefix_hits_and_isolates_outputs():
    pool = Pool(hits=("history:4:9", "history:4:8"))
    scheduler, forwards, _ = _scheduler(pool, fail=True)
    first = _bulk(scheduler, "first", 9, 1)
    second = _bulk(scheduler, "second", 8, 2)
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=[first, second]))
    assert forwards == [("many", [1, 2], 4, "history")]
    assert [x.rid for x in out.items] == ["first", "second"]
    assert all(x.hits[0].cache_hit for x in out.items)
    assert [x.first_miss_result.error for x in out.items] == [
        "packed failed",
        "packed failed",
    ]
    assert all(x.success for x in out.items)


def test_one_packed_bulk_store_failure_does_not_fail_peer():
    pool = Pool(hits=("history:4:9", "history:4:8"))
    original_store = pool.store

    def store(**kwargs):
        if kwargs["key_hash"] == "history:4:1":
            raise ValueError("first store failed")
        return original_store(**kwargs)

    pool.store = store
    scheduler, forwards, _ = _scheduler(pool)
    out = scheduler.handle_c2kv_extract_batch(
        SimpleNamespace(
            items=[
                _bulk(scheduler, "first", 9, 1),
                _bulk(scheduler, "second", 8, 2),
            ]
        )
    )
    assert forwards == [("many", [1, 2], 4, "history")]
    assert out.items[0].first_miss_result.error == "first store failed"
    assert out.items[1].first_miss_result.success
    assert out.items[1].first_miss_result.key_hash == "history:4:2"


def test_bulk_ratio_mismatch_and_tp_use_original_single_paths():
    pool = Pool(hits=("history:4:9",))
    scheduler, forwards, _ = _scheduler(pool)
    first = _bulk(scheduler, "first", 9, 1)
    second = _bulk(scheduler, "second", 2)
    second.items[0].compression_ratio = 8
    out = scheduler.handle_c2kv_extract_batch(SimpleNamespace(items=[first, second]))
    assert [kind for kind, *_ in forwards] == ["one", "one"]
    assert [x.rid for x in out.items] == ["first", "second"]

    scheduler, forwards, _ = _scheduler(Pool(), tp_size=2)
    out = scheduler.handle_c2kv_extract_batch(
        SimpleNamespace(items=[_bulk(scheduler, "a", 1), _bulk(scheduler, "b", 2)])
    )
    assert [kind for kind, *_ in forwards] == ["one", "one"]
    assert [x.rid for x in out.items] == ["a", "b"]
