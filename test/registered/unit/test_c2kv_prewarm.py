"""Exercise the real idle-time queue with CPU extraction events."""

import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[3] / "python/sglang/srt/managers/c2kv_prewarm.py"
SPEC = importlib.util.spec_from_file_location("c2kv_prewarm_cpu_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
NativePrewarmQueue = MODULE.NativePrewarmQueue


def payload(owner="owner", job="job", count=2):
    return {
        "operation": "submit", "owner_id": owner, "job_id": job,
        "session_id": "session", "outer_request_id": "decision",
        "max_extraction_calls": count,
        "chunks": [{"handle": f"chunk-{i}", "token_ids": [i + 1] * 8,
                    "compression_ratio": 8} for i in range(count)],
    }


def result(request, hit=False):
    return SimpleNamespace(
        success=True, error=None, original_seq_len=len(request["input_ids"]),
        gist_len=1, key_hash=f"key-{request['input_ids'][0]}", cache_hit=hit,
        extraction_duration_ns=10, gist_generation_duration_ns=0 if hit else 8,
        paper_measurement=None,
    )


@pytest.mark.asyncio
async def test_ack_does_not_wait_and_foreground_drains_only_one_chunk():
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def extract(**kwargs):
        calls.append(kwargs)
        started.set()
        await release.wait()
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    ack = queue.submit(payload(count=3))
    assert ack["status"] == "queued" and ack["model_calls"] == 0
    await asyncio.wait_for(started.wait(), 1)
    foreground = asyncio.create_task(queue.enter_foreground())
    await asyncio.sleep(0)
    assert not foreground.done()
    release.set()
    await asyncio.wait_for(foreground, 1)
    assert len(calls) == 1
    # The foreground gate holds across request admission and generation.
    await asyncio.sleep(0)
    assert len(calls) == 1
    receipt = await queue.drain("owner", "job", "session")
    assert receipt["status"] == "cancelled"
    assert receipt["model_calls"] == 1 and receipt["cancelled_chunks"] == 2
    assert receipt["budget_known"]
    assert (await queue.drain("owner", "job", "session")) == receipt
    queue.exit_foreground()
    await queue.close()


@pytest.mark.asyncio
async def test_all_foreground_requests_exit_before_background_resumes():
    called = asyncio.Event()

    async def extract(**kwargs):
        called.set()
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    await queue.enter_foreground()
    await queue.enter_foreground()
    queue.submit(payload(count=1))
    queue.exit_foreground()
    await asyncio.sleep(0)
    assert not called.is_set()
    queue.exit_foreground()
    await asyncio.wait_for(called.wait(), 1)
    receipt = await queue.drain("owner", "job", "session")
    assert receipt["status"] == "completed"
    await queue.close()


@pytest.mark.asyncio
async def test_completed_cache_result_is_reusable_and_budget_is_exact():
    cache, forward_calls = {}, []

    async def extract(**kwargs):
        key = tuple(kwargs["input_ids"])
        hit = key in cache
        if not hit:
            forward_calls.append(key)
            cache[key] = True
        return result(kwargs, hit=hit)

    queue = NativePrewarmQueue(extract)
    queue.submit(payload(count=1))
    await asyncio.wait_for(queue.jobs[("owner", "job")].done.wait(), 1)
    receipt = await queue.drain("owner", "job", "session")
    assert receipt["model_calls"] == 1
    await queue.enter_foreground()
    foreground_result = await extract(input_ids=[1] * 8)
    assert foreground_result.cache_hit and len(forward_calls) == 1
    queue.exit_foreground()
    queue.submit(payload(job="next", count=1))
    await asyncio.wait_for(queue.jobs[("owner", "next")].done.wait(), 1)
    receipt = await queue.drain("owner", "next", "session")
    assert receipt["cache_hits"] == 1 and receipt["extraction"]["model_calls"] == 0
    await queue.close()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_communicator_reply():
    started, release = asyncio.Event(), asyncio.Event()

    async def extract(**kwargs):
        started.set()
        await release.wait()
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    queue.submit(payload(count=2))
    await asyncio.wait_for(started.wait(), 1)
    drain = asyncio.create_task(queue.drain("owner", "job", "session"))
    await asyncio.sleep(0)
    drain.cancel()
    with pytest.raises(asyncio.CancelledError):
        await drain
    foreground = asyncio.create_task(queue.enter_foreground())
    await asyncio.sleep(0)
    foreground.cancel()
    with pytest.raises(asyncio.CancelledError):
        await foreground
    assert queue.foreground_count == 0
    release.set()
    receipt = await asyncio.wait_for(queue.drain("owner", "job", "session"), 1)
    assert receipt["model_calls"] == 1 and receipt["cancelled_chunks"] == 1
    await queue.close()


@pytest.mark.asyncio
async def test_other_sessions_cannot_evict_an_unacknowledged_cost_receipt():
    async def extract(**kwargs):
        return result(kwargs)

    queue = NativePrewarmQueue(extract, max_jobs=2)
    queue.submit(payload(owner="slow-tool", count=1))
    await asyncio.wait_for(queue.jobs[("slow-tool", "job")].done.wait(), 1)
    for index in range(4):
        name = f"fast-{index}"
        queue.submit(payload(owner=name, count=1))
        await asyncio.wait_for(queue.jobs[(name, "job")].done.wait(), 1)
        await queue.drain(name, "job", "session")
    receipt = await queue.drain("slow-tool", "job", "session")
    assert receipt["model_calls"] == 1 and receipt["status"] == "completed"
    await queue.close()


@pytest.mark.asyncio
async def test_failure_retains_receipt_and_does_not_hide_unknown_budget():
    async def extract(**kwargs):
        raise RuntimeError("lost scheduler response")

    queue = NativePrewarmQueue(extract)
    queue.submit(payload())
    await asyncio.wait_for(queue.jobs[("owner", "job")].done.wait(), 1)
    receipt = await queue.drain("owner", "job", "session")
    assert receipt["status"] == "failed" and receipt["budget_known"] is False
    assert "lost scheduler response" in receipt["error"]
    await queue.close()


@pytest.mark.asyncio
async def test_idempotency_and_close_cancel_only_owned_pending_work():
    async def extract(**kwargs):
        raise AssertionError("Queue must stay paused")

    queue = NativePrewarmQueue(extract, max_jobs=2)
    await queue.enter_foreground()
    first = payload()
    ack = queue.submit(first)
    assert queue.submit(first) == ack
    with pytest.raises(ValueError, match="different session"):
        await queue.drain("owner", "job", "another-session")
    assert not queue.jobs[("owner", "job")].cancelled
    with pytest.raises(ValueError, match="outstanding"):
        queue.submit(payload(job="different"))
    changed = payload()
    changed["chunks"][0]["token_ids"] = [9]
    with pytest.raises(ValueError, match="different content"):
        queue.submit(changed)
    queue.submit(payload(owner="other"))
    with pytest.raises(ValueError, match="full"):
        queue.submit(payload(owner="third"))
    await queue.close()
    assert (await queue.drain("owner", "job", "session"))["cancelled_chunks"] == 2
    assert (await queue.drain("other", "job", "session"))["model_calls"] == 0
    queue.exit_foreground()


@pytest.mark.parametrize("change", [
    {"operation": []}, {"operation": {}},
    {"max_extraction_calls": 0}, {"max_extraction_calls": True},
    {"chunks": []}, {"owner_id": ""},
    {"chunks": [{"handle": "h", "token_ids": [True], "compression_ratio": 8}]},
    {"chunks": [{"handle": "h", "token_ids": [2**63], "compression_ratio": 8}]},
    {"chunks": [{"handle": "h", "token_ids": [1], "compression_ratio": 0}]},
    {"chunks": [{"handle": "h", "token_ids": [1], "compression_ratio": 8,
                 "projection_set": "tool"}]},
])
def test_invalid_or_unbounded_submissions_are_rejected(change):
    with pytest.raises(ValueError):
        MODULE.validate_prewarm_request(dict(payload(), **change))
