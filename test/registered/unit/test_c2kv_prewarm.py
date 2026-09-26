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
async def test_overlap_foreground_skips_inflight_job_and_runs_during_generation():
    first_started, first_release = asyncio.Event(), asyncio.Event()
    second_started, second_release = asyncio.Event(), asyncio.Event()
    calls = []

    async def extract(**kwargs):
        calls.append(kwargs["rid"])
        if len(calls) == 1:
            first_started.set()
            await first_release.wait()
        else:
            second_started.set()
            await second_release.wait()
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    job = payload()
    job["scheduling"] = MODULE.OVERLAP_SCHEDULING
    queue.submit(job)
    await asyncio.wait_for(first_started.wait(), 1)
    await asyncio.wait_for(queue.enter_foreground(overlap=True), 1)
    first_release.set()
    await asyncio.sleep(0)
    assert len(calls) == 1
    queue.enter_generation()
    await asyncio.wait_for(second_started.wait(), 1)
    assert queue.foreground_count == queue.generation_count == 1
    running = queue.poll("owner", "job", "session")
    assert running["completed_chunks"] == 1 and running["pending_chunks"] == 1
    assert running["model_calls"] == 1
    second_release.set()
    await asyncio.wait_for(queue.jobs[("owner", "job")].done.wait(), 1)
    receipt = queue.poll("owner", "job", "session")
    assert receipt["status"] == "completed" and receipt["pending_chunks"] == 0
    assert receipt["model_calls"] == 2 and receipt["completed_chunks"] == 2
    assert receipt["started_monotonic_ns"] >= receipt["submitted_monotonic_ns"]
    assert receipt["finished_monotonic_ns"] >= receipt["last_completed_monotonic_ns"]
    assert receipt["extraction_wall_duration_ns"] > 0
    assert queue.poll("owner", "job", "session") == receipt
    queue.exit_generation()
    queue.exit_foreground()
    await queue.close()


@pytest.mark.asyncio
async def test_rid_gated_overlap_waits_for_matching_admission_and_foreground_preparation():
    first_started, first_release = asyncio.Event(), asyncio.Event()
    second_started = asyncio.Event()
    calls = []

    async def extract(**kwargs):
        calls.append(kwargs["rid"])
        if len(calls) == 1:
            first_started.set()
            await first_release.wait()
        else:
            second_started.set()
        return result(kwargs)

    queue = NativePrewarmQueue(extract, max_jobs=2)
    job = payload(count=2)
    job.update(scheduling=MODULE.OVERLAP_SCHEDULING, after_native_rid="selected-rid")
    assert queue.submit(job)["status"] == "queued"
    await asyncio.sleep(0)
    assert not calls

    await queue.enter_foreground(overlap=True)
    queue.enter_generation("unrelated-rid")
    await asyncio.sleep(0)
    assert not calls
    queue.exit_generation()

    queue.enter_generation("selected-rid")
    await asyncio.wait_for(first_started.wait(), 1)
    await queue.enter_foreground(overlap=True)
    first_release.set()
    async def first_chunk_completed():
        while not queue.jobs[("owner", "job")].results:
            await asyncio.sleep(0)
    await asyncio.wait_for(first_chunk_completed(), 1)
    assert not second_started.is_set()
    await queue.enter_foreground(overlap=True)
    queue.enter_generation("second-rid")
    await asyncio.sleep(0)
    assert not second_started.is_set()
    queue.enter_generation("third-rid")
    await asyncio.wait_for(second_started.wait(), 1)
    assert "selected-rid" not in queue.recent_admissions
    await asyncio.wait_for(queue.jobs[("owner", "job")].done.wait(), 1)
    assert queue.poll("owner", "job", "session")["status"] == "completed"
    queue.exit_generation()
    queue.exit_generation()
    queue.exit_generation()
    queue.exit_foreground()
    queue.exit_foreground()
    queue.exit_foreground()
    await queue.close()


@pytest.mark.asyncio
async def test_rid_gated_late_submission_and_unadmitted_cancel():
    calls = []

    async def extract(**kwargs):
        calls.append(kwargs["rid"])
        return result(kwargs)

    queue = NativePrewarmQueue(extract, max_jobs=2)
    await queue.enter_foreground(overlap=True)
    queue.enter_generation("already-admitted")
    queue.exit_generation()
    queue.exit_foreground()
    late = payload(owner="late", count=1)
    late.update(scheduling=MODULE.OVERLAP_SCHEDULING, after_native_rid="already-admitted")
    queue.submit(late)
    await asyncio.wait_for(queue.jobs[("late", "job")].done.wait(), 1)
    assert queue.poll("late", "job", "session")["status"] == "completed"

    never = payload(owner="never", count=1)
    never.update(scheduling=MODULE.OVERLAP_SCHEDULING, after_native_rid="never-admitted")
    queue.submit(never)
    receipt = await asyncio.wait_for(queue.drain("never", "job", "session"), 1)
    assert receipt["status"] == "cancelled" and receipt["model_calls"] == 0
    assert receipt["started_monotonic_ns"] is None
    assert len(calls) == 1
    for native_rid in ("later-one", "later-two"):
        await queue.enter_foreground(overlap=True)
        queue.enter_generation(native_rid)
        queue.exit_generation()
        queue.exit_foreground()
    assert list(queue.recent_admissions) == ["later-one", "later-two"]
    await queue.close()


@pytest.mark.asyncio
async def test_prepared_generation_wait_allows_only_admitted_overlap_job():
    calls = []

    async def extract(**kwargs):
        calls.append(kwargs)
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    await queue.enter_foreground(overlap=True)
    await queue.enter_foreground(overlap=True)
    queue.enter_generation("admitted")
    queue.enter_generation_wait("waiting")
    overlap = payload(owner="overlap", count=1)
    overlap.update(scheduling=MODULE.OVERLAP_SCHEDULING,
                   after_native_rid="admitted")
    queue.submit(overlap)
    queue.submit(payload(owner="legacy", count=1))
    await asyncio.wait_for(queue.jobs[("overlap", "job")].done.wait(), 1)
    assert len(calls) == 1 and calls[0]["background_extraction"] is True
    assert queue.poll("legacy", "job", "session")["status"] == "queued"

    queue.exit_generation()
    queue.exit_generation_wait("waiting")
    queue.exit_foreground()
    queue.exit_foreground()
    await asyncio.wait_for(queue.jobs[("legacy", "job")].done.wait(), 1)
    assert len(calls) == 2 and "background_extraction" not in calls[1]
    assert not queue.generation_waiting
    await queue.close()


@pytest.mark.asyncio
async def test_unprepared_foreground_keeps_admitted_overlap_job_paused():
    calls = []

    async def extract(**kwargs):
        calls.append(kwargs)
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    for _ in range(3):
        await queue.enter_foreground(overlap=True)
    queue.enter_generation("admitted")
    queue.enter_generation_wait("prepared")
    job = payload(count=1)
    job.update(scheduling=MODULE.OVERLAP_SCHEDULING,
               after_native_rid="admitted")
    queue.submit(job)
    await asyncio.sleep(0)
    assert not calls
    queue.enter_generation_wait("formerly-unprepared")
    await asyncio.wait_for(queue.jobs[("owner", "job")].done.wait(), 1)
    assert len(calls) == 1
    queue.exit_generation()
    queue.exit_generation_wait("prepared")
    queue.exit_generation_wait("formerly-unprepared")
    for _ in range(3):
        queue.exit_foreground()
    await queue.close()


@pytest.mark.asyncio
async def test_waiting_owner_does_not_unlock_own_overlap_job():
    calls = []

    async def extract(**kwargs):
        calls.append(kwargs)
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    await queue.enter_foreground(overlap=True)
    await queue.enter_foreground(overlap=True)
    queue.enter_generation("other")
    queue.enter_generation_wait("owner")
    job = payload(count=1)
    job.update(scheduling=MODULE.OVERLAP_SCHEDULING,
               after_native_rid="owner")
    queue.submit(job)
    await asyncio.sleep(0)
    assert not calls
    assert queue.poll("owner", "job", "session")["status"] == "queued"
    queue.enter_generation("owner")
    await asyncio.wait_for(queue.jobs[("owner", "job")].done.wait(), 1)
    assert len(calls) == 1 and not queue.generation_waiting
    queue.exit_generation_wait("owner")
    assert queue.generation_count == 2
    queue.exit_generation()
    queue.exit_generation()
    queue.exit_foreground()
    queue.exit_foreground()
    await queue.close()


@pytest.mark.asyncio
async def test_generation_wait_cancel_and_transition_are_request_scoped():
    async def extract(**kwargs):
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    await queue.enter_foreground(overlap=True)
    await queue.enter_foreground(overlap=True)
    queue.enter_generation_wait("cancelled")
    queue.enter_generation_wait("continued")
    with pytest.raises(RuntimeError, match="Duplicate"):
        queue.enter_generation_wait("continued")
    with pytest.raises(RuntimeError, match="foreground scope"):
        queue.enter_generation("unrelated")
    assert queue.generation_waiting == {"cancelled", "continued"}
    queue.exit_generation_wait("cancelled")
    queue.exit_generation_wait("cancelled")
    assert queue.generation_waiting == {"continued"}
    queue.enter_generation("continued")
    assert queue.generation_count == 1 and not queue.generation_waiting
    queue.exit_generation_wait("continued")
    assert queue.generation_count == 1
    queue.exit_generation()
    queue.exit_foreground()
    queue.exit_foreground()
    await queue.close()


@pytest.mark.asyncio
async def test_mixed_modes_preserve_legacy_pause_and_foreground_priority():
    overlap_started, release = asyncio.Event(), asyncio.Event()
    legacy_started = asyncio.Event()
    overlap_calls = 0
    seen_kwargs = []

    async def extract(**kwargs):
        nonlocal overlap_calls
        seen_kwargs.append(kwargs)
        if kwargs["rid"].startswith("prewarm:overlap:"):
            overlap_calls += 1
            overlap_started.set()
            await release.wait()
        else:
            legacy_started.set()
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    overlap = payload(owner="overlap", count=2)
    overlap["scheduling"] = MODULE.OVERLAP_SCHEDULING
    queue.submit(overlap)
    await asyncio.wait_for(overlap_started.wait(), 1)
    await queue.enter_foreground(overlap=True)
    queue.submit(payload(owner="legacy", count=1))
    release.set()
    await asyncio.sleep(0)
    assert overlap_calls == 1 and not legacy_started.is_set()
    queue.enter_generation()
    await asyncio.wait_for(queue.jobs[("overlap", "job")].done.wait(), 1)
    assert overlap_calls == 2 and not legacy_started.is_set()
    queue.exit_generation()
    queue.exit_foreground()
    await asyncio.wait_for(legacy_started.wait(), 1)
    assert (await queue.drain("legacy", "job", "session"))["status"] == "completed"
    assert all(call["background_extraction"] is True for call in seen_kwargs[:2])
    assert "background_extraction" not in seen_kwargs[2]
    await queue.close()


@pytest.mark.asyncio
async def test_poll_keeps_identity_and_partial_cost_before_cancel():
    started, release = asyncio.Event(), asyncio.Event()

    async def extract(**kwargs):
        started.set()
        await release.wait()
        return result(kwargs)

    queue = NativePrewarmQueue(extract)
    queued = payload(count=2)
    queued["scheduling"] = MODULE.OVERLAP_SCHEDULING
    queue.submit(queued)
    await asyncio.wait_for(started.wait(), 1)
    with pytest.raises(ValueError, match="different session"):
        queue.poll("owner", "job", "wrong-session")
    assert queue.poll("owner", "job", "session")["pending_chunks"] == 2
    draining = asyncio.create_task(queue.drain("owner", "job", "session"))
    await asyncio.sleep(0)
    assert not draining.done()
    release.set()
    receipt = await asyncio.wait_for(draining, 1)
    assert receipt["status"] == "cancelled"
    assert receipt["model_calls"] == 1 and receipt["cancelled_chunks"] == 1
    assert receipt["pending_chunks"] == 0
    assert queue.poll("owner", "job", "session") == receipt
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
    assert receipt["extraction_wall_duration_ns"] > 0
    assert receipt["finished_monotonic_ns"] >= receipt["started_monotonic_ns"]
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
    {"after_native_rid": "selected-rid"},
    {"scheduling": MODULE.OVERLAP_SCHEDULING, "after_native_rid": ""},
    {"scheduling": MODULE.OVERLAP_SCHEDULING, "after_native_rid": "x" * 4097},
])
def test_invalid_or_unbounded_submissions_are_rejected(change):
    with pytest.raises(ValueError):
        MODULE.validate_prewarm_request(dict(payload(), **change))
