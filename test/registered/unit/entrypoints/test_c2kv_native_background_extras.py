"""CPU lifecycle contracts for optional native background extras."""

import ast
import asyncio
from collections import Counter
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


class HTTPException(Exception):
    def __init__(self, *, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code


HTTP_SERVER = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/entrypoints/http_server.py"
)


def _load(*names, **bindings):
    tree = ast.parse(HTTP_SERVER.read_text(encoding="utf-8"), filename=str(HTTP_SERVER))
    functions = []
    for name in names:
        node = next(
            item for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        )
        node.decorator_list = []
        functions.append(node)
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "asyncio": asyncio,
        "uuid": uuid,
        "HTTPException": HTTPException,
        "C2KVNativePackedGenerateRequest": object,
        "Request": object,
        **bindings,
    }
    exec(compile(module, str(HTTP_SERVER), "exec"), namespace)
    return namespace


def _load_scheduler_lease_handler():
    path = HTTP_SERVER.parents[1] / "managers/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "Scheduler"
        for method in cls.body
        if isinstance(method, ast.FunctionDef)
        and method.name == "handle_c2kv_pin_lease"
    )
    namespace = {
        "C2KVPinLeaseReqInput": object,
        "C2KVPinLeaseReqOutput": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["handle_c2kv_pin_lease"]


def _load_communicator_lease_method():
    path = HTTP_SERVER.parents[1] / "managers/tokenizer_communicator_mixin.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "TokenizerCommunicatorMixin"
        for method in cls.body
        if isinstance(method, ast.AsyncFunctionDef)
        and method.name == "c2kv_pin_lease"
    )
    namespace = {
        "asyncio": asyncio,
        "TokenizerManager": object,
        "Optional": Optional,
        "List": List,
        "C2KVPinLeaseReqInput": lambda **kwargs: SimpleNamespace(**kwargs),
        "C2KVPinLeaseReqOutput": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["c2kv_pin_lease"]


def _plan(with_extra=True):
    chunks = [
        {"handle": "extra", "chunk_id": "extra", "token_ids": [11] * 8}
    ] if with_extra else []
    chunks.append(
        {"handle": "selected", "chunk_id": "selected", "token_ids": [22] * 8}
    )
    return SimpleNamespace(
        logical_input_ids=[1, 22, 2],
        selected_handles=["selected"],
        segment_boundaries=[(1, 2)],
        compression_handles=["extra"] if with_extra else [],
        unique_chunks=chunks,
        costs={
            "system_tokens": 1,
            "gist_prefix_kv_tokens": 1,
            "raw_workspace_kv_tokens": 1,
            "resident_kv_tokens": 3,
        },
    )


def _request(budget=2):
    class Chunk:
        projection_set = "history"

        def __init__(self, token_ids):
            self.token_ids = token_ids

        def model_dump(self):
            return {"token_ids": self.token_ids, "projection_set": self.projection_set}

    chunk = Chunk([22] * 8)
    extra = Chunk([11] * 8)
    return SimpleNamespace(
        sampling_profile="greedy-v1",
        sampling_params={"max_new_tokens": 2},
        shadow_features=None,
        max_extraction_calls=budget,
        max_tool_extraction_calls=None,
        encoder_chunks=[chunk],
        compression_chunks=[extra],
        raw_tool_segments=[],
        tool_gist_segments=[],
        paper_whole_full_kv_tokens=None,
        system_input_ids=[1],
        workspace_input_ids=[2],
        packing_version="history-event-v1",
        raw_layout_profile="event-native-evidence-v1",
        encoding_scope="current",
        compression_ratio=8,
        rid="native-1",
        session_id="session-1",
        generation_id="generation-1",
    )


def _namespace(manager, plan, enabled, shadow_layer=None):
    return _load(
        "_c2kv_native_sampling_params",
        "_c2kv_native_whole_full_measurement",
        "_c2kv_native_background_extras_fallback_reason",
        "_c2kv_native_background_extras_eligible",
        "_c2kv_native_generate_with_background_extras",
        "v1_c2kv_native_generate",
        _global_state=SimpleNamespace(tokenizer_manager=manager),
        _c2kv_native_capability=lambda: {
            "enabled": True,
            "model_binding": {"pic_enabled": False},
            "shadow_feature_layer": shadow_layer,
            "num_hidden_layers": 4,
            "kv_bytes_per_token": 4,
        },
        get_bool_env_var=lambda name: enabled,
        plan_native_packed_request=lambda **kwargs: plan,
        GenerateReqInput=lambda **kwargs: SimpleNamespace(**kwargs),
        C2KVSegmentInfo=lambda **kwargs: SimpleNamespace(**kwargs),
        orjson_response=lambda value: value,
        NATIVE_PACKED_RESPONSE_SCHEMA="c2kv-native-packed-response-v1",
        logger=SimpleNamespace(error=lambda *args, **kwargs: None),
        _create_error_response=lambda error: {"error": str(error)},
        float16_roundtrip=lambda values: list(values),
    )


def _extract_result(chunk_id, *, success=True):
    return SimpleNamespace(
        success=success,
        error="C2KV_EXTRACTION_BUDGET_EXHAUSTED" if not success else None,
        original_seq_len=8,
        gist_len=1,
        cache_hit=False,
        key_hash=f"key-{chunk_id}",
        extraction_duration_ns=10,
        gist_generation_duration_ns=8,
        paper_measurement=None,
    )


def _generation_output(ids, finished):
    return {
        "output_ids": ids,
        "text": "answer" if finished else "a",
        "meta_info": {
            "finish_reason": "length" if finished else None,
            "output_token_logprobs": [(-0.5, token) for token in ids],
        },
    }


@pytest.mark.asyncio
async def test_selected_admitted_before_extras_and_response_waits_for_both():
    allow_first = asyncio.Event()
    extra_started = asyncio.Event()
    release_extra = asyncio.Event()
    finish_generation = asyncio.Event()

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        def __init__(self):
            self.extract_calls = []
            self.generation_input = None
            self.leased = False

        async def c2kv_pin_lease(self, *, owner_id, action, key_hashes=None):
            if action == "acquire":
                assert key_hashes == ["key-selected"]
                self.leased = True
            else:
                assert self.leased
                self.leased = False
            return SimpleNamespace(success=True, error="")

        async def c2kv_extract(self, **kwargs):
            self.extract_calls.append(kwargs["rid"])
            if kwargs["input_ids"][0] == 11:
                assert self.leased
                extra_started.set()
                await release_extra.wait()
                return _extract_result("extra")
            return _extract_result("selected")

        async def generate_request(self, request, _raw_request):
            self.generation_input = request
            assert request.stream is True
            assert self.extract_calls == ["native-1:extract:1"]
            await allow_first.wait()
            yield _generation_output([41], False)
            await finish_generation.wait()
            yield _generation_output([41, 42], True)

        async def abort_request_and_wait(self, rid):
            raise AssertionError(f"Unexpected abort: {rid}")

    manager = Manager()
    namespace = _namespace(manager, _plan(), enabled=True)
    task = asyncio.create_task(
        namespace["v1_c2kv_native_generate"](_request(), SimpleNamespace(headers={}))
    )
    await asyncio.sleep(0)
    assert not extra_started.is_set()
    allow_first.set()
    await asyncio.wait_for(extra_started.wait(), 1)
    assert not task.done()
    finish_generation.set()
    await asyncio.sleep(0)
    assert not task.done()
    release_extra.set()
    response = await asyncio.wait_for(task, 1)
    assert manager.extract_calls == ["native-1:extract:1", "native-1:extract:0"]
    assert manager.generation_input.c2kv_segments[0].key_hash == "key-selected"
    assert response["output_ids"] == [41, 42]
    assert response["token_logprobs"] == [-0.5, -0.5]
    assert response["request_ids"]["native_extraction_request_ids"] == [
        "native-1:extract:0", "native-1:extract:1"
    ]
    assert response["extraction"]["cache_misses"] == 2
    assert response["costs"]["materialized_encoder_tokens"] == 16
    assert response["serving_execution"] == {
        "mode": "selected-first-response-barrier-v1",
        "response_barrier": "generation_and_extras",
        "extra_jobs": 1,
        "fallback_reason": None,
    }
    assert not manager.leased


def test_tight_budget_incremental_output_and_no_extras_stay_serial():
    manager = SimpleNamespace(server_args=SimpleNamespace(incremental_streaming_output=False))
    namespace = _namespace(manager, _plan(), enabled=True)
    eligible = namespace["_c2kv_native_background_extras_eligible"]
    assert eligible(_request(budget=2), _plan(), manager)
    assert not eligible(_request(budget=1), _plan(), manager)
    manager.server_args.incremental_streaming_output = True
    assert not eligible(_request(budget=2), _plan(), manager)
    manager.server_args.incremental_streaming_output = False
    assert not eligible(_request(budget=2), _plan(with_extra=False), manager)
    namespace["get_bool_env_var"] = lambda name: False
    assert not eligible(_request(budget=2), _plan(), manager)


@pytest.mark.asyncio
async def test_tight_budget_missing_extra_fails_before_generation():
    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        def __init__(self):
            self.extract_calls = []
            self.generated = False

        async def c2kv_extract(self, **kwargs):
            self.extract_calls.append(kwargs["rid"])
            return _extract_result("extra", success=False)

        async def generate_request(self, request, raw_request):
            self.generated = True
            yield _generation_output([41], True)

    manager = Manager()
    namespace = _namespace(manager, _plan(), enabled=True)
    response = await namespace["v1_c2kv_native_generate"](
        _request(budget=0), SimpleNamespace(headers={})
    )
    assert response == {"error": "C2KV_EXTRACTION_BUDGET_EXHAUSTED"}
    assert manager.extract_calls == ["native-1:extract:0"]
    assert not manager.generated


@pytest.mark.asyncio
async def test_background_and_serial_preserve_output_cost_and_request_ids():
    async def run(enabled):
        class Manager:
            server_args = SimpleNamespace(incremental_streaming_output=False)

            def __init__(self):
                self.generation_input = None
                self.leased = False

            async def c2kv_pin_lease(self, *, owner_id, action, key_hashes=None):
                self.leased = action == "acquire"
                return SimpleNamespace(success=True, error="")

            async def c2kv_extract(self, **kwargs):
                chunk_id = "extra" if kwargs["input_ids"][0] == 11 else "selected"
                return _extract_result(chunk_id)

            async def generate_request(self, request, raw_request):
                self.generation_input = request
                if request.stream:
                    yield _generation_output([41], False)
                yield _generation_output([41, 42], True)

            async def abort_request_and_wait(self, rid):
                raise AssertionError(f"Unexpected abort: {rid}")

        manager = Manager()
        namespace = _namespace(manager, _plan(), enabled=enabled)
        response = await namespace["v1_c2kv_native_generate"](
            _request(), SimpleNamespace(headers={})
        )
        assert not manager.leased
        return response, manager.generation_input

    background, background_request = await run(True)
    serial, serial_request = await run(False)
    for field in (
        "output_ids", "text", "token_logprobs", "finish_reason", "encoder_chunks",
        "compression_chunks", "extraction", "costs", "request_ids",
    ):
        assert background[field] == serial[field]
    assert background_request.input_ids == serial_request.input_ids
    assert background_request.sampling_params == serial_request.sampling_params
    assert background_request.c2kv_segments == serial_request.c2kv_segments
    assert background_request.stream and not serial_request.stream


@pytest.mark.asyncio
async def test_background_extra_failure_is_reported_after_generation():
    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        def __init__(self):
            self.generation_finished = False

        async def c2kv_pin_lease(self, *, owner_id, action, key_hashes=None):
            return SimpleNamespace(success=True, error="")

        async def c2kv_extract(self, **kwargs):
            if kwargs["input_ids"][0] == 11:
                return _extract_result("extra", success=False)
            return _extract_result("selected")

        async def generate_request(self, request, raw_request):
            yield _generation_output([41], False)
            self.generation_finished = True
            yield _generation_output([41, 42], True)

        async def abort_request_and_wait(self, rid):
            raise AssertionError(f"Unexpected abort: {rid}")

    manager = Manager()
    namespace = _namespace(manager, _plan(), enabled=True)
    response = await namespace["v1_c2kv_native_generate"](
        _request(), SimpleNamespace(headers={})
    )
    assert manager.generation_finished
    assert response == {"error": "C2KV_EXTRACTION_BUDGET_EXHAUSTED"}


@pytest.mark.asyncio
async def test_background_mode_keeps_final_shadow_readout():
    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        async def c2kv_extract(self, **kwargs):
            return _extract_result(
                "extra" if kwargs["input_ids"][0] == 11 else "selected"
            )

        async def c2kv_pin_lease(self, *, owner_id, action, key_hashes=None):
            return SimpleNamespace(success=True, error="")

        async def generate_request(self, request, raw_request):
            assert request.return_hidden_states
            first = _generation_output([41], False)
            first["meta_info"]["hidden_states"] = [[1.0, 2.0]]
            yield first
            final = _generation_output([41, 42], True)
            final["meta_info"]["hidden_states"] = [[1.0, 2.0], [3.0, 4.0]]
            yield final

        async def abort_request_and_wait(self, rid):
            raise AssertionError(f"Unexpected abort: {rid}")

    namespace = _namespace(Manager(), _plan(), enabled=True, shadow_layer=3)
    request = _request()
    request.shadow_features = {"enabled": True, "prefill_layer": 3}
    response = await namespace["v1_c2kv_native_generate"](
        request, SimpleNamespace(headers={})
    )
    assert response["output_ids"] == [41, 42]
    assert response["shadow_features"]["prefill"]["hidden"] == [1.0, 2.0]


@pytest.mark.asyncio
async def test_cancel_aborts_generation_and_cancels_owned_extras():
    extra_started = asyncio.Event()
    extra_cancelled = asyncio.Event()
    aborted = asyncio.Event()
    wait_forever = asyncio.Event()

    class Manager:
        async def generate_request(self, request, raw_request):
            yield _generation_output([41], False)
            await wait_forever.wait()
            yield _generation_output([41, 42], True)

        async def abort_request_and_wait(self, rid):
            assert rid == "native-1"
            aborted.set()

    async def run_extras():
        extra_started.set()
        try:
            await wait_forever.wait()
        except asyncio.CancelledError:
            extra_cancelled.set()
            raise

    helper = _load("_c2kv_native_generate_with_background_extras")
    task = asyncio.create_task(
        helper["_c2kv_native_generate_with_background_extras"](
            Manager(), SimpleNamespace(rid="native-1", stream=False), None, run_extras
        )
    )
    await asyncio.wait_for(extra_started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert extra_cancelled.is_set()
    assert aborted.is_set()


@pytest.mark.asyncio
async def test_extra_failure_aborts_generation_without_waiting_for_more_tokens():
    aborted = asyncio.Event()
    wait_forever = asyncio.Event()

    class Manager:
        async def generate_request(self, request, raw_request):
            yield _generation_output([41], False)
            await wait_forever.wait()
            yield _generation_output([41, 42], True)

        async def abort_request_and_wait(self, rid):
            assert rid == "native-1"
            aborted.set()

    async def run_extras():
        raise ValueError("extra failed")

    helper = _load("_c2kv_native_generate_with_background_extras")
    with pytest.raises(ValueError, match="extra failed"):
        await asyncio.wait_for(
            helper["_c2kv_native_generate_with_background_extras"](
                Manager(), SimpleNamespace(rid="native-1", stream=False), None,
                run_extras,
            ),
            1,
        )
    assert aborted.is_set()


@pytest.mark.asyncio
async def test_endpoint_cancellation_drains_extraction_before_lease_release():
    extra_started = asyncio.Event()
    finish_extract = asyncio.Event()
    aborted = asyncio.Event()
    wait_forever = asyncio.Event()

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)

        def __init__(self):
            self.leased = False
            self.released = False

        async def c2kv_extract(self, **kwargs):
            if kwargs["input_ids"][0] == 11:
                extra_started.set()
                await finish_extract.wait()
                assert self.leased
                return _extract_result("extra")
            return _extract_result("selected")

        async def c2kv_pin_lease(self, *, owner_id, action, key_hashes=None):
            if action == "acquire":
                self.leased = True
            else:
                assert finish_extract.is_set()
                self.leased = False
                self.released = True
            return SimpleNamespace(success=True, error="")

        async def generate_request(self, request, raw_request):
            yield _generation_output([41], False)
            await wait_forever.wait()
            yield _generation_output([41, 42], True)

        async def abort_request_and_wait(self, rid):
            aborted.set()

    manager = Manager()
    namespace = _namespace(manager, _plan(), enabled=True)
    task = asyncio.create_task(
        namespace["v1_c2kv_native_generate"](_request(), SimpleNamespace(headers={}))
    )
    await asyncio.wait_for(extra_started.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    assert manager.leased and not manager.released
    finish_extract.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert aborted.is_set() and manager.released


def test_scheduler_pin_lease_ownership_atomicity_and_shared_keys():
    handler = _load_scheduler_lease_handler()

    class Pool:
        def __init__(self):
            self._cache = {"a": object(), "b": object()}
            self.counts = Counter()

        def pin_many(self, keys):
            if any(key not in self._cache for key in keys):
                return False
            self.counts.update(set(keys))
            return True

        def unpin_many(self, keys):
            for key in set(keys):
                self.counts[key] -= 1

    pool = Pool()
    scheduler = SimpleNamespace(
        c2kv_pool=pool, c2kv_native_pin_leases={}, tp_size=1,
    )

    def lease(owner_id, action, keys=()):
        return handler(
            scheduler,
            SimpleNamespace(owner_id=owner_id, action=action, key_hashes=list(keys)),
        )

    assert not lease("missing", "acquire", ["a", "absent"]).success
    assert pool.counts == Counter()
    assert lease("owner-a", "acquire", ["a", "a"]).success
    assert lease("owner-a", "acquire", ["a"]).success
    assert pool.counts["a"] == 1
    assert not lease("owner-a", "acquire", ["b"]).success
    assert lease("owner-b", "acquire", ["a", "b"]).success
    assert pool.counts["a"] == 2 and pool.counts["b"] == 1
    assert lease("owner-a", "release").success
    assert lease("owner-a", "release").success
    assert pool.counts["a"] == 1
    assert lease("owner-b", "release").success
    assert pool.counts["a"] == pool.counts["b"] == 0


@pytest.mark.asyncio
async def test_partial_dp_lease_rollback_drains_after_cancellation():
    method = _load_communicator_lease_method()
    release_started = asyncio.Event()
    allow_release_reply = asyncio.Event()

    class Communicator:
        def __init__(self):
            self.leased_shards = 0

        async def __call__(self, req):
            if req.action == "acquire":
                self.leased_shards = 1
                return [
                    SimpleNamespace(success=True, error=""),
                    SimpleNamespace(success=False, error="missing key"),
                ]
            release_started.set()
            await allow_release_reply.wait()
            self.leased_shards = 0
            return [SimpleNamespace(success=True, error="")] * 2

    communicator = Communicator()
    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        c2kv_pin_lease_communicator=communicator,
    )
    task = asyncio.create_task(
        method(manager, owner_id="owner", action="acquire", key_hashes=["a"])
    )
    await asyncio.wait_for(release_started.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    assert communicator.leased_shards == 1 and not task.done()
    allow_release_reply.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert communicator.leased_shards == 0
