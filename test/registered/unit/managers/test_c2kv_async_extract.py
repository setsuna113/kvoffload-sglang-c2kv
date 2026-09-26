"""CPU scheduling contracts for one worker-owned background extraction."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/c2kv_async_extract.py"


class Request(SimpleNamespace):
    def __init__(self, rid="bg", key="key", **kwargs):
        super().__init__(**dict(
            rid=rid, key=key, input_ids=[1, 2, 3, 4], allow_cache_miss=True,
            background_extraction=True, projection_set="history",
            c2kv_outer_request_id="outer", c2kv_measurement_phase="prewarm",
        ) | kwargs)


class Output(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**dict(success=True, error="", cache_hit=False) | kwargs)


class Bulk(SimpleNamespace):
    pass


class Stream:
    def __init__(self, **kwargs):
        self.syncs = 0
        self.waits = []
        self.wait_events = []
        self.device = kwargs.get('device')

    def wait_stream(self, stream):
        self.waits.append(stream)

    def wait_event(self, event):
        assert event.recorded_on is not None
        self.wait_events.append(event)

    def synchronize(self):
        self.syncs += 1


class Event:
    def __init__(self, **kwargs):
        self.ready = False
        self.recorded_on = None
        self.recorded_thread = None

    def record(self, stream):
        self.recorded_on = stream
        self.recorded_thread = threading.get_ident()

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        assert other.ready
        return 5.0


class Clock:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def perf_counter_ns(self):
        with self.lock:
            self.value += 10
            return self.value


class Stepper:
    def __init__(self):
        self.calls = 0
        self.gist_key_values = []
        self.fail = False
        self.fail_at = None
        self._phase = "prelude"

    def step(self):
        self.calls += 1
        if self.fail or self.calls == self.fail_at:
            raise ValueError("layer failed")
        if self.calls == 1:
            self._phase = "layers"
        if 2 <= self.calls <= 3:
            self.gist_key_values.append((object(), object()))
        if self.calls == 3:
            self._phase = "finalize"
        if self.calls == 4:
            self._phase = "done"
        return self.calls == 4

    @property
    def result(self):
        assert self.calls == 4
        return self.gist_key_values, SimpleNamespace(shape=(1, 1)), object()


def make_controller(monkeypatch, clock=None):
    monkeypatch.setenv("C2KV_GIST_ASYNC", "1")
    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    recorded = []
    pending = {}
    pending_snapshots = []

    def set_pending(rid, tensors, *, append=False):
        assert isinstance(tensors, tuple)
        assert append is True
        assert len(tensors) == 1, "Only the newly completed layer is registered"
        pending[rid] = pending.get(rid, ()) + tensors
        pending_snapshots.append(pending[rid])

    telemetry = SimpleNamespace(
        enabled=lambda: True, concurrent_enabled=lambda: True,
        start_request=lambda **kw: recorded.append(("start", kw)),
        request_scope=lambda rid: nullcontext(),
        set_pending_tensors=set_pending,
        clear_pending_tensors=lambda rid: pending.pop(rid, None),
        sample=lambda *args, **kw: recorded.append(("sample", args)),
        finish_request=lambda **kw: {"duration_ns": 99, "metrics": kw["metric_overrides"]},
    )
    worker_devices = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(Stream=Stream, Event=Event, stream=lambda stream: nullcontext(),
                             current_device=lambda: 0,
                             set_device=lambda device: worker_devices.append((device, threading.get_ident()))),
        no_grad=lambda: nullcontext(), long=int, bool=bool,
        tensor=lambda *args, **kw: object(), ones_like=lambda *args, **kw: object(),
    )
    timer = (SimpleNamespace(perf_counter_ns=clock.perf_counter_ns)
             if clock is not None else time)
    namespace = dict(dataclass=dataclass, field=field, os=os, time=timer, torch=fake_torch,
                     ThreadPoolExecutor=ThreadPoolExecutor,
                     paper_telemetry=telemetry, TokenizedExtractReqInput=Request,
                     C2KVExtractReqOutput=Output, C2KVBulkCacheLookupReqInput=Bulk,
                     C2KVExtractBatchReqInput=Bulk)
    controls = ['PauseGenerationReqInput', 'ContinueGenerationReqInput',
                'ReleaseMemoryOccupationReqInput', 'ResumeMemoryOccupationReqInput', 'RpcReqInput',
                'UpdateWeightFromDiskReqInput', 'UpdateWeightsFromDistributedReqInput',
                'UpdateWeightsFromIPCReqInput', 'UpdateWeightsFromTensorReqInput']
    namespace.update({name: type(name, (), {}) for name in controls})
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    cache = {}
    capacity = [True]
    stored = []
    store_threads = []

    def store(**kwargs):
        assert pending, "Temporary tensors must remain counted during publication"
        store_threads.append(threading.get_ident())
        entry = SimpleNamespace(gist_len=1, original_seq_len=4)
        cache[kwargs['key_hash']] = entry
        stored.append(kwargs)
        return entry

    stepper = Stepper()
    runner = SimpleNamespace(model=type('Qwen3ForCausalLM', (), {})(),
                             create_c2kv_extract_stepper=lambda *args, **kw: stepper)
    replies = []
    reply_threads = []

    def send_output(*args):
        replies.append(args)
        reply_threads.append(threading.get_ident())

    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=runner), enable_overlap=True,
        tp_size=1, pp_size=1, device="cuda", server_args=SimpleNamespace(dp_size=1),
        c2kv_pool=SimpleNamespace(_cache=cache, max_entry_tokens=10, max_total_tokens=20,
                                 can_allocate=lambda *args, **kw: capacity[0], store=store),
        _c2kv_extract_cache_key=lambda request: (request.key, 4, None),
        schedule_stream=Stream(), send_to_tokenizer=SimpleNamespace(send_output=send_output),
    )
    controller = namespace['AsyncGistExtraction'](scheduler)
    return SimpleNamespace(controller=controller, scheduler=scheduler, stepper=stepper,
                           replies=replies, stored=stored, pending=pending, capacity=capacity,
                           namespace=namespace, recorded=recorded, worker_devices=worker_devices,
                           pending_snapshots=pending_snapshots,
                           store_threads=store_threads, reply_threads=reply_threads)


def submitted_result(controller):
    return controller.job.future.result(timeout=2)


def finish_job(controller):
    result = submitted_result(controller)
    controller.advance()
    assert controller.job is not None
    result.end_event.ready = True
    controller.advance()
    return result


def test_worker_submits_all_steps_and_publishes_only_after_event(monkeypatch):
    state = make_controller(monkeypatch)
    controller = state.controller
    request = Request()
    assert controller.intercept(request)
    assert state.stepper.calls == 0
    assert controller.busy
    duplicate = Bulk(items=[Request(rid="fg", background_extraction=False)])
    assert controller.intercept(duplicate)
    assert controller.take_ready_requests() == []
    controller.advance(decode_dispatched=True)
    result = submitted_result(controller)
    assert state.stepper.calls == result.ticks == 4
    assert not state.replies and not state.stored
    controller.advance(decode_dispatched=True)
    assert not state.replies
    assert controller.job is not None
    result.end_event.ready = True
    controller.advance(decode_dispatched=True)
    assert len(state.stored) == len(state.replies) == 1
    output, owner = state.replies[0]
    assert output.success and owner is request
    assert output.paper_measurement['metrics']['gist_overlap_decode_batches'] == 3
    assert output.paper_measurement['metrics']['gist_execution_mode'] == 'tp1-worker-stream-v1'
    assert len([entry for entry in state.recorded if entry[0] == 'sample']) == 2
    assert [len(snapshot) for snapshot in state.pending_snapshots] == [1, 2]
    assert state.worker_devices == [(0, state.worker_devices[0][1])]
    assert state.worker_devices[0][1] != threading.get_ident()
    assert state.store_threads == state.reply_threads == [threading.get_ident()]
    assert state.scheduler.schedule_stream.syncs == 1
    assert not state.pending
    assert controller.take_ready_requests() == [duplicate]
    assert not controller.busy


def test_phase_timers_reconcile_and_wait_for_end_event(monkeypatch):
    state = make_controller(monkeypatch, clock=Clock())
    controller = state.controller
    assert controller.intercept(Request())
    controller.advance()
    result = submitted_result(controller)
    controller.advance()
    assert not state.stored and not state.replies
    assert controller.busy
    assert result.cpu_step_ns == (
        result.cpu_prelude_ns + result.cpu_layers_ns + result.cpu_finalize_ns
    )
    assert result.cpu_prelude_ns > 0
    assert result.cpu_layers_ns > 0
    assert result.cpu_finalize_ns > 0
    assert 0 < result.cpu_telemetry_ns <= result.cpu_layers_ns

    result.end_event.ready = True
    controller.advance()
    metrics = state.replies[0][0].paper_measurement['metrics']
    assert metrics['gist_cpu_step_ns'] == (
        metrics['gist_cpu_prelude_ns']
        + metrics['gist_cpu_layers_ns']
        + metrics['gist_cpu_finalize_ns']
    )
    assert metrics['gist_cpu_telemetry_ns'] == result.cpu_telemetry_ns
    assert metrics['gist_pool_store_cpu_ns'] == 10
    assert metrics['gist_pool_sync_cpu_ns'] == 10
    assert state.scheduler.schedule_stream.syncs == 1
    assert len(state.stored) == len(state.replies) == 1


@pytest.mark.parametrize('reason', ['foreground', 'disabled', 'tp2', 'no_budget', 'hit', 'full'])
def test_ineligible_jobs_keep_synchronous_path(monkeypatch, reason):
    state = make_controller(monkeypatch)
    request = Request()
    if reason == 'foreground':
        request.background_extraction = False
    elif reason == 'disabled':
        monkeypatch.delenv('C2KV_GIST_ASYNC')
    elif reason == 'tp2':
        state.scheduler.tp_size = 2
    elif reason == 'no_budget':
        request.allow_cache_miss = False
    elif reason == 'hit':
        state.scheduler.c2kv_pool._cache['key'] = object()
    else:
        state.capacity[0] = False
    assert not state.controller.intercept(request)
    assert state.controller.job is None
    assert not state.recorded


def test_capacity_rechecked_after_decode_has_run(monkeypatch):
    state = make_controller(monkeypatch)
    assert state.controller.intercept(Request())
    state.controller.advance()
    result = submitted_result(state.controller)
    state.capacity[0] = False
    result.end_event.ready = True
    state.controller.advance()
    assert not state.stored
    assert not state.replies[0][0].success
    assert 'capacity' in state.replies[0][0].error
    assert not state.pending and not state.controller.busy


def test_failed_step_retains_job_until_submitted_work_finishes(monkeypatch):
    state = make_controller(monkeypatch)
    state.controller.intercept(Request())
    state.stepper.fail_at = 3
    state.controller.advance()
    result = submitted_result(state.controller)
    assert 'layer failed' in result.error
    assert state.controller.job is not None and not state.replies
    state.controller.advance()
    assert state.controller.job is not None and len(state.pending['bg']) == 1
    result.end_event.ready = True
    state.controller.advance()
    assert not state.stored
    assert 'layer failed' in state.replies[0][0].error
    metrics = state.replies[0][0].paper_measurement['metrics']
    assert metrics['gist_cpu_step_ns'] == (
        metrics['gist_cpu_prelude_ns']
        + metrics['gist_cpu_layers_ns']
        + metrics['gist_cpu_finalize_ns']
    )
    assert metrics['gist_cpu_layers_ns'] > 0
    assert not state.controller.busy


def test_blocked_worker_does_not_block_scheduler_advance(monkeypatch):
    state = make_controller(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    original_step = state.stepper.step

    def blocked_step():
        entered.set()
        assert release.wait(timeout=5)
        return original_step()

    state.stepper.step = blocked_step
    assert state.controller.intercept(Request())
    try:
        state.controller.advance()
        assert entered.wait(timeout=2)
        started = time.perf_counter()
        state.controller.advance(decode_dispatched=True)
        assert time.perf_counter() - started < 0.2
        assert state.controller.busy and not state.replies
    finally:
        release.set()
    result = submitted_result(state.controller)
    assert result.ticks == 4
    state.controller.advance()
    assert state.controller.busy and not state.replies
    result.end_event.ready = True
    state.controller.advance()
    assert state.replies[0][0].success


def test_weight_change_waits_for_async_job(monkeypatch):
    state = make_controller(monkeypatch)
    state.controller.intercept(Request())
    mutation = state.namespace['UpdateWeightsFromTensorReqInput']()
    assert state.controller.intercept(mutation)
    assert state.controller.take_ready_requests() == []


def test_each_job_waits_on_pre_submission_scheduler_event(monkeypatch):
    state = make_controller(monkeypatch)
    state.scheduler.enable_overlap = False
    for index in range(2):
        assert state.controller.intercept(Request(rid=f'bg{index}', key=f'key{index}'))
        state.stepper.calls = 0
        state.controller.advance()
        result = submitted_result(state.controller)
        result.end_event.ready = True
        state.controller.advance()
    assert len(state.controller._worker_stream.wait_events) == 2
    assert all(event.recorded_on is state.scheduler.schedule_stream
               for event in state.controller._worker_stream.wait_events)
    assert all(event.recorded_thread == threading.get_ident()
               for event in state.controller._worker_stream.wait_events)


def test_pending_control_pairs_preserve_fifo(monkeypatch):
    state = make_controller(monkeypatch)
    state.controller.intercept(Request())
    names = ['PauseGenerationReqInput', 'ContinueGenerationReqInput',
             'ReleaseMemoryOccupationReqInput', 'ResumeMemoryOccupationReqInput']
    requests = [state.namespace[name]() for name in names]
    for request in requests:
        assert state.controller.intercept(request)
    state.controller.advance()
    finish_job(state.controller)
    assert state.controller.take_ready_requests() == requests
