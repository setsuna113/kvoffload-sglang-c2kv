"""CPU contracts for the optional native cache-hit lookup RPC."""

import ast
import asyncio
import importlib.util
from collections import OrderedDict
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Optional

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCHEDULER = ROOT / "python/sglang/srt/managers/scheduler.py"
COMMUNICATOR = ROOT / "python/sglang/srt/managers/tokenizer_communicator_mixin.py"


def _scheduler_methods():
    tree = ast.parse(SCHEDULER.read_text(encoding="utf-8"))
    methods = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "Scheduler"
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_c2kv_extract_cache_key",
            "handle_c2kv_bulk_cache_lookup",
            "handle_extract_request",
        }
    ]
    for method in methods:
        method.decorator_list = []
        if method.name == "handle_extract_request":
            method.body = [
                node for node in method.body if not isinstance(node, ast.ImportFrom)
            ]
    module = ast.Module(body=methods, type_ignores=[])
    ast.fix_missing_locations(module)

    def output(**kwargs):
        return SimpleNamespace(
            **{
                "hits": [],
                "first_miss_index": 0,
                "success": True,
                "error": "",
                **kwargs,
            }
        )

    namespace = {
        "TokenizedExtractReqInput": object,
        "C2KVBulkCacheLookupReqInput": object,
        "C2KVBulkCacheLookupReqOutput": output,
        "C2KVExtractReqOutput": output,
    }
    exec(compile(module, str(SCHEDULER), "exec"), namespace)
    return namespace


def _native_helpers():
    path = HERE / "test_c2kv_native_background_extras.py"
    spec = importlib.util.spec_from_file_location(
        "c2kv_native_background_test_helpers", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _communicator_bulk_method():
    tree = ast.parse(COMMUNICATOR.read_text(encoding="utf-8"))
    method = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "TokenizerCommunicatorMixin"
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "c2kv_bulk_cache_lookup"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "asyncio": asyncio,
        "TokenizerManager": object,
        "List": list,
        "Dict": dict,
        "Any": object,
        "Optional": Optional,
        "C2KVBulkCacheLookupReqOutput": object,
        "C2KVBulkCacheLookupReqInput": lambda **kwargs: SimpleNamespace(**kwargs),
        "TokenizedExtractReqInput": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    exec(compile(module, str(COMMUNICATOR), "exec"), namespace)
    return namespace["c2kv_bulk_cache_lookup"]


def test_scheduler_returns_only_hit_prefix_with_exact_projection_keys():
    methods = _scheduler_methods()

    class Pool:
        def __init__(self):
            self.seen = []
            self._cache = {
                "history:4:11": object(),
                "tool:tool-checkpoint:4:12": object(),
                "history:4:14": object(),
            }

        def compute_hash(self, ids, *, compression_ratio, extractor_config):
            self.seen.append((list(ids), compression_ratio, dict(extractor_config)))
            projection = extractor_config.get("projection_set", "history")
            identity = extractor_config.get("projection_identity")
            prefix = f"{projection}:{identity}:" if identity else f"{projection}:"
            return f"{prefix}{compression_ratio}:{ids[0]}"

    pool = Pool()
    calls = []
    scheduler = SimpleNamespace(
        c2kv_pool=pool,
        tp_size=1,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                get_c2kv_compression_ratio=int,
                model=SimpleNamespace(c2kv_tool_gist_identity="tool-checkpoint"),
            )
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        server_args=SimpleNamespace(),
    )
    scheduler._c2kv_extract_cache_key = MethodType(
        methods["_c2kv_extract_cache_key"], scheduler
    )

    def extract(item):
        calls.append(item.rid)
        return SimpleNamespace(success=True, cache_hit=True, key_hash=item.rid)

    scheduler.handle_extract_request = extract
    items = [
        SimpleNamespace(
            rid=f"r{i}",
            input_ids=[token],
            compression_ratio=4,
            projection_set=projection,
        )
        for i, (token, projection) in enumerate(
            [(11, "history"), (12, "tool"), (13, "history"), (14, "history")]
        )
    ]
    reply = methods["handle_c2kv_bulk_cache_lookup"](
        scheduler, SimpleNamespace(items=items)
    )

    assert reply.success and reply.first_miss_index == 2
    assert [hit.key_hash for hit in reply.hits] == ["r0", "r1"]
    assert calls == ["r0", "r1"]
    assert [ids for ids, _, _ in pool.seen] == [[11], [12], [13]]
    assert "projection_set" not in pool.seen[0][2]
    assert pool.seen[1][2]["projection_identity"] == "tool-checkpoint"


@pytest.mark.parametrize("allowed", [False, True])
def test_scheduler_fuses_hit_prefix_and_only_first_miss(allowed):
    methods = _scheduler_methods()

    class Pool:
        _cache = {"history:4:11": object(), "history:4:14": object()}

        def compute_hash(self, ids, *, compression_ratio, extractor_config):
            projection = extractor_config.get("projection_set", "history")
            identity = extractor_config.get("projection_identity")
            prefix = f"{projection}:{identity}:" if identity else f"{projection}:"
            return f"{prefix}{compression_ratio}:{ids[0]}"

    scheduler = SimpleNamespace(
        c2kv_pool=Pool(),
        tp_size=1,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                get_c2kv_compression_ratio=int,
                model=SimpleNamespace(c2kv_tool_gist_identity="tool-checkpoint"),
            )
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        server_args=SimpleNamespace(),
    )
    scheduler._c2kv_extract_cache_key = MethodType(
        methods["_c2kv_extract_cache_key"], scheduler
    )
    calls = []

    def extract(item):
        calls.append((item.rid, item.compression_ratio, item.projection_set,
                      item.allow_cache_miss))
        hit = item.input_ids[0] in (11, 14)
        return SimpleNamespace(
            success=hit or item.allow_cache_miss,
            cache_hit=hit,
            key_hash=item.rid,
            error="C2KV_EXTRACTION_BUDGET_EXHAUSTED" if not hit and not allowed else "",
        )

    scheduler.handle_extract_request = extract
    items = [
        SimpleNamespace(rid="hit", input_ids=[11], compression_ratio=4,
                        projection_set="history", allow_cache_miss=False),
        SimpleNamespace(rid="miss", input_ids=[12], compression_ratio=8,
                        projection_set="tool", allow_cache_miss=allowed),
        SimpleNamespace(rid="trailing", input_ids=[14], compression_ratio=4,
                        projection_set="history", allow_cache_miss=False),
    ]
    reply = methods["handle_c2kv_bulk_cache_lookup"](
        scheduler, SimpleNamespace(items=items, materialize_first_miss=True)
    )
    assert reply.success and reply.first_miss_index == 1
    assert [result.key_hash for result in reply.hits] == ["hit"]
    assert reply.first_miss_result.key_hash == "miss"
    assert reply.first_miss_result.success is allowed
    assert calls == [
        ("hit", 4, "history", False),
        ("miss", 8, "tool", allowed),
    ]


def test_fused_miss_preserves_hit_touch_then_unpinned_eviction_order():
    methods = _scheduler_methods()
    items = [
        SimpleNamespace(rid=str(token), input_ids=[token], compression_ratio=4,
                        projection_set="history", allow_cache_miss=True)
        for token in (11, 12, 13)
    ]

    def run(materialize_first_miss):
        class Pool:
            def __init__(self):
                self._cache = OrderedDict((str(token), object()) for token in (11, 12))

        pool = Pool()
        scheduler = SimpleNamespace(c2kv_pool=pool, tp_size=1)
        scheduler._c2kv_extract_cache_key = lambda item: (
            str(item.input_ids[0]), item.compression_ratio, None
        )
        calls = []

        def extract(item):
            key = str(item.input_ids[0])
            hit = key in pool._cache
            calls.append((key, hit))
            if hit:
                pool._cache.move_to_end(key)
            else:
                assert item.allow_cache_miss
                pool._cache.popitem(last=False)
                pool._cache[key] = object()
            return SimpleNamespace(success=True, cache_hit=hit, key_hash=key)

        scheduler.handle_extract_request = extract
        reply = methods["handle_c2kv_bulk_cache_lookup"](
            scheduler,
            SimpleNamespace(items=items, materialize_first_miss=materialize_first_miss),
        )
        if not materialize_first_miss:
            scheduler.handle_extract_request(items[reply.first_miss_index])
        return calls, list(pool._cache)

    expected = ([("11", True), ("12", True), ("13", False)], ["12", "13"])
    assert run(False) == expected
    assert run(True) == expected


def test_real_extract_miss_keeps_projection_set_for_model_forward():
    methods = _scheduler_methods()
    forward_projections = []

    class Pool:
        max_entry_tokens = 64
        max_total_tokens = 64

        def compute_hash(self, ids, *, compression_ratio, extractor_config):
            assert extractor_config["projection_identity"] == "tool-checkpoint"
            return "tool-key"

        def get(self, key_hash):
            return None

        def can_allocate(self, gist_len, *, existing_key):
            return True

        def store(self, **kwargs):
            assert kwargs["key_hash"] == "tool-key"
            return SimpleNamespace(gist_len=1)

    def forward(input_ids, attention_mask, ratio, *, projection_set):
        forward_projections.append(projection_set)
        return object(), SimpleNamespace(shape=(1, 1)), object()

    scheduler = SimpleNamespace(
        c2kv_pool=Pool(),
        tp_size=1,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                get_c2kv_compression_ratio=int,
                model=SimpleNamespace(c2kv_tool_gist_identity="tool-checkpoint"),
                forward_c2kv_extract=forward,
            )
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        server_args=SimpleNamespace(),
        enable_overlap=False,
        forward_stream=None,
        _log_c2kv_token_usage=lambda *args, **kwargs: None,
    )
    scheduler._c2kv_extract_cache_key = MethodType(
        methods["_c2kv_extract_cache_key"], scheduler
    )
    fake_torch = SimpleNamespace(
        tensor=lambda value, **kwargs: SimpleNamespace(shape=(1, len(value[0]))),
        ones_like=lambda value, **kwargs: SimpleNamespace(shape=value.shape),
        long=object(),
        bool=object(),
    )
    methods["handle_extract_request"].__globals__.update(
        torch=fake_torch,
        paper_telemetry=SimpleNamespace(enabled=lambda: False),
        _is_npu=False,
    )
    result = methods["handle_extract_request"](
        scheduler,
        SimpleNamespace(
            input_ids=[12] * 8,
            compression_ratio=4,
            projection_set="tool",
            allow_cache_miss=True,
        ),
    )
    assert result.success and result.key_hash == "tool-key"
    assert forward_projections == ["tool"]


@pytest.mark.asyncio
async def test_cancelled_bulk_lookup_drains_reply_before_next_request():
    method = _communicator_bulk_method()
    started = asyncio.Event()
    release = asyncio.Event()
    completed = []

    async def communicator(req):
        started.set()
        await release.wait()
        completed.append(req.items[0].rid)
        return [SimpleNamespace(first_miss_index=0)]

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        c2kv_bulk_cache_lookup_communicator=communicator,
    )

    def item(rid):
        return {
            "rid": rid,
            "input_ids": [11],
            "compression_ratio": 4,
            "projection_set": "history",
        }

    first = asyncio.create_task(method(manager, [item("first")]))
    await started.wait()
    first.cancel()
    await asyncio.sleep(0)
    assert not first.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert completed == ["first"]

    await method(manager, [item("second")])
    assert completed == ["first", "second"]
    with pytest.raises(ValueError, match="1 to 32 items"):
        await method(manager, [item("oversize")] * 33)


@pytest.mark.asyncio
async def test_communicator_only_forwards_miss_permission_in_fused_mode():
    method = _communicator_bulk_method()
    sent = []

    async def communicator(req):
        sent.append(req)
        return [SimpleNamespace(success=True)]

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        c2kv_bulk_cache_lookup_communicator=communicator,
    )
    item = {
        "rid": "tool-miss", "input_ids": [12], "compression_ratio": 8,
        "projection_set": "tool", "allow_cache_miss": True,
    }
    await method(manager, [item])
    await method(manager, [item], materialize_first_miss=True)
    assert sent[0].materialize_first_miss is False
    assert sent[0].items[0].allow_cache_miss is False
    assert sent[1].materialize_first_miss is True
    assert sent[1].items[0].allow_cache_miss is True


@pytest.mark.asyncio
async def test_native_lookup_rechecks_after_eviction_and_preserves_receipts():
    helpers = _native_helpers()
    plan = helpers._plan()
    plan.unique_chunks = [
        {"handle": name, "chunk_id": name, "token_ids": [token] * 8}
        for name, token in [
            ("a", 11),
            ("b", 12),
            ("c", 13),
            ("d", 14),
            ("selected", 22),
        ]
    ]
    plan.compression_handles = ["a", "b", "c", "d"]
    request = helpers._request(budget=2)

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        def __init__(self):
            self.cached = {11, 12, 14, 22}
            self.calls = []

        async def c2kv_bulk_cache_lookup(self, items, **kwargs):
            self.calls.append(("lookup", [item["rid"] for item in items]))
            hits = []
            for item in items:
                token = item["input_ids"][0]
                if token not in self.cached:
                    break
                result = helpers._extract_result(item["rid"].split(":")[-1])
                result.cache_hit = True
                result.key_hash = f"key-{token}"
                hits.append(result)
            return SimpleNamespace(
                success=True, error="", hits=hits, first_miss_index=len(hits)
            )

        async def c2kv_extract(self, **kwargs):
            token = kwargs["input_ids"][0]
            self.calls.append(("extract", kwargs["rid"], kwargs["allow_cache_miss"]))
            result = helpers._extract_result(str(token))
            result.cache_hit = token in self.cached
            result.key_hash = f"key-{token}"
            if token == 13:
                self.cached.remove(14)
            self.cached.add(token)
            return result

        async def generate_request(self, request, raw_request):
            yield helpers._generation_output([41, 42], True)

    manager = Manager()
    namespace = helpers._namespace(manager, plan, enabled=False)
    namespace["get_bool_env_var"] = lambda name: name == "C2KV_NATIVE_BULK_CACHE_LOOKUP"
    response = await namespace["v1_c2kv_native_generate"](
        request, SimpleNamespace(headers={})
    )

    assert manager.calls == [
        ("lookup", [f"native-1:extract:{i}" for i in range(5)]),
        ("extract", "native-1:extract:2", True),
        ("lookup", ["native-1:extract:3", "native-1:extract:4"]),
        ("extract", "native-1:extract:3", True),
        ("extract", "native-1:extract:4", False),
    ]
    assert response["extraction"]["cache_hits"] == 3
    assert response["extraction"]["cache_misses"] == 2
    assert response["request_ids"]["native_extraction_request_ids"] == [
        f"native-1:extract:{i}" for i in range(5)
    ]
    assert response["costs"]["materialized_encoder_tokens"] == 16
    assert response["costs"]["scope_reused_encoder_tokens"] == 24
    assert response["serving_execution"]["bulk_cache_lookup_calls"] == 2
    assert response["serving_execution"]["bulk_cache_hit_chunks"] == 2


@pytest.mark.asyncio
async def test_bulk_hits_do_not_charge_projection_miss_budgets():
    helpers = _native_helpers()
    plan = helpers._plan()
    plan.unique_chunks = [
        {"handle": "history", "chunk_id": "history", "token_ids": [11] * 8},
        {
            "handle": "tool",
            "chunk_id": "tool",
            "token_ids": [12] * 8,
            "projection_set": "tool",
        },
        {"handle": "selected", "chunk_id": "selected", "token_ids": [22] * 8},
    ]
    plan.compression_handles = ["history", "tool"]
    request = helpers._request(budget=0)
    request.max_tool_extraction_calls = 1

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        def __init__(self):
            self.extract_calls = []

        async def c2kv_bulk_cache_lookup(self, items, **kwargs):
            hit = helpers._extract_result("history")
            hit.cache_hit = True
            return SimpleNamespace(
                success=True, error="", hits=[hit], first_miss_index=1
            )

        async def c2kv_extract(self, **kwargs):
            self.extract_calls.append(
                (kwargs["projection_set"], kwargs["allow_cache_miss"])
            )
            result = helpers._extract_result("selected")
            result.cache_hit = kwargs["projection_set"] == "history"
            return result

        async def generate_request(self, request, raw_request):
            yield helpers._generation_output([41, 42], True)

    manager = Manager()
    namespace = helpers._namespace(manager, plan, enabled=False)
    namespace["get_bool_env_var"] = lambda name: name == "C2KV_NATIVE_BULK_CACHE_LOOKUP"
    response = await namespace["v1_c2kv_native_generate"](
        request, SimpleNamespace(headers={})
    )
    assert manager.extract_calls == [("tool", True), ("history", False)]
    assert response["extraction"]["history_model_calls"] == 0
    assert response["extraction"]["tool_model_calls"] == 1
    assert response["serving_execution"]["bulk_cache_hit_chunks"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "history_budget,tool_budget,expect_success",
    [(0, 1, True), (0, None, False)],
)
async def test_native_fused_first_miss_keeps_projection_and_budget_semantics(
    history_budget, tool_budget, expect_success
):
    helpers = _native_helpers()
    plan = helpers._plan(with_extra=False)
    plan.unique_chunks = [
        {"handle": "history-hit", "chunk_id": "history-hit", "token_ids": [11] * 8},
        {"handle": "tool-miss", "chunk_id": "tool-miss", "token_ids": [12] * 8,
         "projection_set": "tool", "compression_ratio": 4},
        {"handle": "history-tail", "chunk_id": "history-tail", "token_ids": [14] * 8},
    ]
    plan.selected_handles = [chunk["handle"] for chunk in plan.unique_chunks]
    plan.segment_boundaries = [(1, 2), (2, 3), (3, 4)]
    plan.compression_handles = []
    plan.logical_input_ids = [1, 11, 12, 14, 2]
    request = helpers._request(budget=history_budget)
    request.max_tool_extraction_calls = tool_budget
    request.compression_chunks = []

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        def __init__(self):
            self.calls = []
            self.generated = False

        async def c2kv_bulk_cache_lookup(self, items, **kwargs):
            self.calls.append(("bulk", [item["rid"] for item in items], kwargs))
            assert kwargs["materialize_first_miss"] is True
            assert [item["allow_cache_miss"] for item in items] == [
                False, expect_success, False
            ]
            hit = helpers._extract_result("history-hit")
            hit.cache_hit = True
            miss = helpers._extract_result("tool-miss", success=expect_success)
            miss.gist_len = 2
            return SimpleNamespace(
                success=True, error="", hits=[hit], first_miss_index=1,
                first_miss_result=miss,
            )

        async def c2kv_extract(self, **kwargs):
            self.calls.append(("extract", kwargs))
            assert kwargs["rid"] == "native-1:extract:2"
            assert kwargs["allow_cache_miss"] is False
            result = helpers._extract_result("history-tail")
            result.cache_hit = True
            return result

        async def generate_request(self, request, raw_request):
            self.generated = True
            yield helpers._generation_output([41, 42], True)

    manager = Manager()
    namespace = helpers._namespace(manager, plan, enabled=False)
    namespace["get_bool_env_var"] = lambda name: name in {
        "C2KV_NATIVE_BULK_CACHE_LOOKUP", "C2KV_NATIVE_BULK_FIRST_MISS"
    }
    response = await namespace["v1_c2kv_native_generate"](
        request, SimpleNamespace(headers={})
    )
    assert len(manager.calls) == (2 if expect_success else 1)
    assert manager.generated is expect_success
    if expect_success:
        assert response["extraction"]["history_model_calls"] == 0
        assert response["extraction"]["tool_model_calls"] == 1
        assert response["costs"]["materialized_encoder_tokens"] == 8
        assert response["costs"]["scope_reused_encoder_tokens"] == 16
        assert response["serving_execution"]["bulk_cache_lookup_calls"] == 1
        assert response["serving_execution"]["bulk_cache_hit_chunks"] == 1
        assert response["serving_execution"]["bulk_first_miss_calls"] == 1
        assert [chunk["cache_hit"] for chunk in response["encoder_chunks"]] == [
            True, False, True
        ]
        assert response["encoder_chunks"][1]["gist_len"] == 2
    else:
        assert "C2KV_EXTRACTION_BUDGET_EXHAUSTED" in response["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "history_budget,tool_budget,trailing_projection",
    [(1, None, "history"), (0, 1, "tool")],
)
async def test_fused_miss_consumes_the_finite_budget_before_trailing_miss(
    history_budget, tool_budget, trailing_projection
):
    helpers = _native_helpers()
    plan = helpers._plan(with_extra=False)
    plan.unique_chunks = [
        {"handle": "hit", "chunk_id": "hit", "token_ids": [11] * 8},
        {"handle": "first-miss", "chunk_id": "first-miss", "token_ids": [12] * 8,
         "projection_set": "tool"},
        {"handle": "trailing-miss", "chunk_id": "trailing-miss",
         "token_ids": [13] * 8, "projection_set": trailing_projection},
    ]
    plan.selected_handles = [chunk["handle"] for chunk in plan.unique_chunks]
    plan.segment_boundaries = [(1, 2), (2, 3), (3, 4)]
    plan.compression_handles = []
    plan.logical_input_ids = [1, 11, 12, 13, 2]
    request = helpers._request(budget=history_budget)
    request.max_tool_extraction_calls = tool_budget
    request.compression_chunks = []

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        def __init__(self):
            self.trailing_allow = None

        async def c2kv_bulk_cache_lookup(self, items, **kwargs):
            assert kwargs["materialize_first_miss"] is True
            assert items[1]["allow_cache_miss"] is True
            hit = helpers._extract_result("hit")
            hit.cache_hit = True
            return SimpleNamespace(
                success=True, error="", hits=[hit], first_miss_index=1,
                first_miss_result=helpers._extract_result("first-miss"),
            )

        async def c2kv_extract(self, **kwargs):
            self.trailing_allow = kwargs["allow_cache_miss"]
            return helpers._extract_result("trailing-miss", success=False)

        async def generate_request(self, request, raw_request):
            raise AssertionError("Generation must not run after budget exhaustion")
            yield

    manager = Manager()
    namespace = helpers._namespace(manager, plan, enabled=False)
    namespace["get_bool_env_var"] = lambda name: name in {
        "C2KV_NATIVE_BULK_CACHE_LOOKUP", "C2KV_NATIVE_BULK_FIRST_MISS"
    }
    response = await namespace["v1_c2kv_native_generate"](
        request, SimpleNamespace(headers={})
    )
    assert manager.trailing_allow is False
    assert "C2KV_EXTRACTION_BUDGET_EXHAUSTED" in response["error"]


@pytest.mark.asyncio
async def test_native_bulk_lookup_is_off_by_default():
    helpers = _native_helpers()

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        def __init__(self):
            self.extract_ids = []

        async def c2kv_bulk_cache_lookup(self, items, **kwargs):
            raise AssertionError("Bulk lookup ran while the feature flag was off")

        async def c2kv_extract(self, **kwargs):
            self.extract_ids.append(kwargs["rid"])
            token = kwargs["input_ids"][0]
            return helpers._extract_result("extra" if token == 11 else "selected")

        async def generate_request(self, request, raw_request):
            yield helpers._generation_output([41, 42], True)

    manager = Manager()
    namespace = helpers._namespace(manager, helpers._plan(), enabled=False)
    response = await namespace["v1_c2kv_native_generate"](
        helpers._request(), SimpleNamespace(headers={})
    )
    assert response["extraction"]["cache_misses"] == 2
    assert manager.extract_ids == ["native-1:extract:0", "native-1:extract:1"]
    assert response["serving_execution"]["bulk_cache_lookup_calls"] == 0
    assert response["serving_execution"]["bulk_cache_hit_chunks"] == 0
