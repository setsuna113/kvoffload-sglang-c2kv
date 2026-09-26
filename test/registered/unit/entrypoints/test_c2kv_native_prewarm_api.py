"""CPU integration of the native HTTP functions with the real prewarm queue."""

import asyncio
import importlib.util
import sys
from types import SimpleNamespace

import pytest

import test_c2kv_native_background_extras as native


PATH = native.HTTP_SERVER.parents[1] / "managers/c2kv_prewarm.py"
NAME = "sglang.srt.managers.c2kv_prewarm"


def load_queue(monkeypatch):
    spec = importlib.util.spec_from_file_location(NAME, PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, NAME, module)
    spec.loader.exec_module(module)
    return module


def submission():
    return {
        "operation": "submit", "owner_id": "owner", "job_id": "job",
        "session_id": "session", "max_extraction_calls": 2,
        "chunks": [
            {"handle": "selected", "token_ids": [22] * 8, "compression_ratio": 8},
            {"handle": "extra", "token_ids": [11] * 8, "compression_ratio": 8},
        ],
    }


@pytest.mark.asyncio
async def test_submit_native_foreground_and_drain_use_one_queue(monkeypatch):
    module = load_queue(monkeypatch)
    started, release = asyncio.Event(), asyncio.Event()
    generation, finish = asyncio.Event(), asyncio.Event()

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)
        model_config = SimpleNamespace(vocab_size=100)

        def __init__(self):
            self.cache = set()
            self.calls = []

        async def c2kv_extract(self, **kwargs):
            self.calls.append(kwargs["rid"])
            if kwargs["rid"].startswith("prewarm:"):
                started.set()
                await release.wait()
            key = tuple(kwargs["input_ids"])
            result = native._extract_result("selected" if key[0] == 22 else "extra")
            result.cache_hit = key in self.cache
            self.cache.add(key)
            return result

        async def generate_request(self, request, raw_request):
            generation.set()
            await finish.wait()
            yield native._generation_output([41, 42], True)

        async def c2kv_bulk_cache_lookup(self, items, **kwargs):
            assert self.c2kv_native_prewarm_queue.foreground_count == 1
            hits = []
            for item in items:
                if tuple(item["input_ids"]) not in self.cache:
                    break
                hit = native._extract_result("selected")
                hit.cache_hit = True
                hits.append(hit)
            return SimpleNamespace(success=True, error="", hits=hits, first_miss_index=len(hits))

    manager = Manager()
    queue = module.NativePrewarmQueue(manager.c2kv_extract)
    manager.c2kv_native_prewarm_queue = queue
    api = native._load(
        "c2kv_native_prewarm", _global_state=SimpleNamespace(tokenizer_manager=manager),
        _c2kv_native_prewarm_queue=lambda manager: queue,
        _create_error_response=lambda error: {"error": str(error)},
    )["c2kv_native_prewarm"]
    ack = await api(submission())
    assert ack["status"] == "queued" and not started.is_set()
    await asyncio.wait_for(started.wait(), 1)
    namespace = native._namespace(manager, native._plan(), enabled=False)
    namespace["get_bool_env_var"] = lambda name: name in {
        "C2KV_NATIVE_CROSS_TURN_PREWARM", "C2KV_NATIVE_BULK_CACHE_LOOKUP"}
    namespace["_c2kv_native_prewarm_queue"] = lambda manager: queue
    foreground = asyncio.create_task(namespace["v1_c2kv_native_generate"](
        native._request(), SimpleNamespace(headers={})))
    await asyncio.sleep(0)
    assert queue.foreground_count == 1 and not foreground.done()
    release.set()
    await asyncio.wait_for(generation.wait(), 1)
    assert sum(rid.startswith("prewarm:") for rid in manager.calls) == 1
    receipt = await api({"operation": "drain", "owner_id": "owner", "job_id": "job",
                         "session_id": "session"})
    assert receipt["model_calls"] == 1 and receipt["cancelled_chunks"] == 1
    finish.set()
    response = await asyncio.wait_for(foreground, 1)
    assert response["extraction"]["cache_hits"] == 1
    assert response["extraction"]["cache_misses"] == 1
    assert response["serving_execution"]["bulk_cache_lookup_calls"] == 1
    assert queue.foreground_count == 0
    assert response["output_ids"] == [41, 42]
    await queue.close()


@pytest.mark.asyncio
async def test_http_rejects_bad_ids_and_cross_session_control(monkeypatch):
    module = load_queue(monkeypatch)

    async def extract(**kwargs):
        raise AssertionError("No extraction should be submitted")

    queue = module.NativePrewarmQueue(extract)
    await queue.enter_foreground()
    manager = SimpleNamespace(model_config=SimpleNamespace(vocab_size=100))
    api = native._load(
        "c2kv_native_prewarm", _global_state=SimpleNamespace(tokenizer_manager=manager),
        _c2kv_native_prewarm_queue=lambda manager: queue,
        get_bool_env_var=lambda name: False,
        _create_error_response=lambda error: {"error": str(error)},
    )["c2kv_native_prewarm"]
    invalid = submission()
    invalid["chunks"][0]["token_ids"] = [100]
    assert "vocabulary" in (await api(invalid))["error"]
    assert "error" in await api({"operation": []})
    overlap = submission()
    overlap["scheduling"] = module.OVERLAP_SCHEDULING
    assert "disabled" in (await api(overlap))["error"]
    assert not queue.jobs
    assert (await api(submission()))["status"] == "queued"
    assert "different session" in (await api({
        "operation": "cancel", "owner_id": "owner", "job_id": "job", "session_id": "other",
    }))["error"]
    await queue.close()
    queue.exit_foreground()


@pytest.mark.asyncio
async def test_async_http_poll_and_generation_do_not_wait_for_unselected_history(monkeypatch):
    module = load_queue(monkeypatch)
    first_started, release_first = asyncio.Event(), asyncio.Event()
    second_started, release_second = asyncio.Event(), asyncio.Event()
    generation_started, finish_generation = asyncio.Event(), asyncio.Event()

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)
        model_config = SimpleNamespace(vocab_size=100)

        def __init__(self):
            self.leased = False
            self.calls = []

        async def c2kv_extract(self, **kwargs):
            self.calls.append(kwargs["rid"])
            if kwargs["rid"].startswith("prewarm:"):
                if kwargs["input_ids"][0] == 11:
                    first_started.set()
                    await release_first.wait()
                else:
                    second_started.set()
                    await release_second.wait()
            return native._extract_result("selected")

        async def c2kv_pin_lease(self, *, owner_id, action, key_hashes=None):
            self.leased = action == "acquire"
            return SimpleNamespace(success=True, error="")

        async def generate_request(self, request, raw_request):
            assert request.stream and self.leased
            generation_started.set()
            yield native._generation_output([41], False)
            await finish_generation.wait()
            yield native._generation_output([41, 42], True)

        async def abort_request_and_wait(self, rid):
            raise AssertionError(f"Unexpected abort: {rid}")

    manager = Manager()
    queue = module.NativePrewarmQueue(manager.c2kv_extract)
    manager.c2kv_native_prewarm_queue = queue
    api = native._load(
        "c2kv_native_prewarm", _global_state=SimpleNamespace(tokenizer_manager=manager),
        _c2kv_native_prewarm_queue=lambda manager: queue,
        get_bool_env_var=lambda name: name == "C2KV_NATIVE_ASYNC_COMPRESSION",
        _create_error_response=lambda error: {"error": str(error)},
    )["c2kv_native_prewarm"]
    job = submission()
    job["scheduling"] = module.OVERLAP_SCHEDULING
    job["chunks"][0]["token_ids"] = [11] * 8
    job["chunks"][1]["token_ids"] = [12] * 8
    assert (await api(job))["pending_chunks"] == 2
    await asyncio.wait_for(first_started.wait(), 1)
    polled = await asyncio.wait_for(api({
        "operation": "poll", "owner_id": "owner", "job_id": "job", "session_id": "session",
    }), 1)
    assert polled["pending_chunks"] == 2 and polled["model_calls"] == 0
    assert "different session" in (await api({
        "operation": "poll", "owner_id": "owner", "job_id": "job", "session_id": "other",
    }))["error"]

    namespace = native._namespace(manager, native._plan(with_extra=False), enabled=False)
    namespace["get_bool_env_var"] = lambda name: name == "C2KV_NATIVE_ASYNC_COMPRESSION"
    namespace["_c2kv_native_prewarm_queue"] = lambda manager: queue
    request = native._request(budget=1)
    request.compression_chunks = []
    foreground = asyncio.create_task(namespace["v1_c2kv_native_generate"](
        request, SimpleNamespace(headers={})))
    await asyncio.wait_for(generation_started.wait(), 1)
    assert not release_first.is_set() and not second_started.is_set()
    release_first.set()
    await asyncio.wait_for(second_started.wait(), 1)
    assert not manager.leased and not foreground.done()
    finish_generation.set()
    response = await asyncio.wait_for(foreground, 1)
    assert response["output_ids"] == [41, 42]
    assert response["serving_execution"]["mode"] == module.OVERLAP_SCHEDULING
    assert response["serving_execution"]["response_barrier"] == "generation"
    timing = response["telemetry"]["execution_timing"]
    assert timing["generation_finished_monotonic_ns"] >= timing["generation_admitted_monotonic_ns"]
    assert timing["selected_extraction_wall_ns"] > 0
    assert timing["generation_wall_ns"] > 0
    assert not release_second.is_set()
    release_second.set()
    receipt = await asyncio.wait_for(api({
        "operation": "poll", "owner_id": "owner", "job_id": "job", "session_id": "session",
    }), 1)
    if receipt["pending_chunks"]:
        await asyncio.wait_for(queue.jobs[("owner", "job")].done.wait(), 1)
        receipt = await api({
            "operation": "poll", "owner_id": "owner", "job_id": "job", "session_id": "session",
        })
    assert receipt["status"] == "completed" and receipt["model_calls"] == 2
    assert all("extraction_started_monotonic_ns" in row for row in receipt["results"])
    await queue.close()


@pytest.mark.asyncio
async def test_http_gated_offer_starts_only_after_its_native_request_is_admitted(monkeypatch):
    module = load_queue(monkeypatch)
    prewarm_started, release_prewarm = asyncio.Event(), asyncio.Event()
    finish_generation = asyncio.Event()

    class Manager:
        server_args = SimpleNamespace(incremental_streaming_output=False)
        model_config = SimpleNamespace(vocab_size=100)

        def __init__(self):
            self.calls = []

        async def c2kv_extract(self, **kwargs):
            self.calls.append(kwargs["rid"])
            if kwargs["rid"].startswith("prewarm:"):
                prewarm_started.set()
                await release_prewarm.wait()
            return native._extract_result("selected")

        async def c2kv_pin_lease(self, *, owner_id, action, key_hashes=None):
            return SimpleNamespace(success=True, error="")

        async def generate_request(self, request, raw_request):
            yield native._generation_output([41], False)
            await finish_generation.wait()
            yield native._generation_output([41, 42], True)

        async def abort_request_and_wait(self, rid):
            raise AssertionError(f"Unexpected abort: {rid}")

    manager = Manager()
    queue = module.NativePrewarmQueue(manager.c2kv_extract)
    manager.c2kv_native_prewarm_queue = queue
    api = native._load(
        "c2kv_native_prewarm", _global_state=SimpleNamespace(tokenizer_manager=manager),
        _c2kv_native_prewarm_queue=lambda manager: queue,
        get_bool_env_var=lambda name: name == "C2KV_NATIVE_ASYNC_COMPRESSION",
        _create_error_response=lambda error: {"error": str(error)},
    )["c2kv_native_prewarm"]
    job = submission()
    job["chunks"] = job["chunks"][1:]
    job["max_extraction_calls"] = 1
    job["scheduling"] = module.OVERLAP_SCHEDULING
    job["after_native_rid"] = "native-1"
    assert (await api(job))["status"] == "queued"
    await asyncio.sleep(0)
    assert manager.calls == []

    namespace = native._namespace(manager, native._plan(with_extra=False), enabled=False)
    namespace["get_bool_env_var"] = lambda name: name == "C2KV_NATIVE_ASYNC_COMPRESSION"
    namespace["_c2kv_native_prewarm_queue"] = lambda manager: queue
    request = native._request(budget=1)
    request.compression_chunks = []
    foreground = asyncio.create_task(namespace["v1_c2kv_native_generate"](
        request, SimpleNamespace(headers={})))
    await asyncio.wait_for(prewarm_started.wait(), 1)
    assert manager.calls[0] == "native-1:extract:0"
    assert manager.calls[1].startswith("prewarm:")
    release_prewarm.set()
    finish_generation.set()
    response = await asyncio.wait_for(foreground, 1)
    assert response["output_ids"] == [41, 42]
    assert (await api({
        "operation": "poll", "owner_id": "owner", "job_id": "job", "session_id": "session",
    }))["status"] == "completed"
    await queue.close()
