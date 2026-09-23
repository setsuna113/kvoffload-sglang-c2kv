"""CPU integration of the native HTTP functions with the real prewarm queue."""

import asyncio
import importlib.util
from pathlib import Path
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
        _create_error_response=lambda error: {"error": str(error)},
    )["c2kv_native_prewarm"]
    invalid = submission()
    invalid["chunks"][0]["token_ids"] = [100]
    assert "vocabulary" in (await api(invalid))["error"]
    assert "error" in await api({"operation": []})
    assert not queue.jobs
    assert (await api(submission()))["status"] == "queued"
    assert "different session" in (await api({
        "operation": "cancel", "owner_id": "owner", "job_id": "job", "session_id": "other",
    }))["error"]
    await queue.close()
    queue.exit_foreground()
