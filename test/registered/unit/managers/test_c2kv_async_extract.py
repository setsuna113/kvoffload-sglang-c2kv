"""CPU scheduling contracts for worker-owned background extraction."""

import ast
import copy
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
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


class BulkLookup(SimpleNamespace):
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
        assert tensors, "Only newly completed layer tensors are registered"
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
                     copy=copy,
                     ThreadPoolExecutor=ThreadPoolExecutor,
                     paper_telemetry=telemetry, TokenizedExtractReqInput=Request,
                     C2KVExtractReqOutput=Output, C2KVBulkCacheLookupReqInput=BulkLookup,
                     C2KVBulkCacheLookupReqOutput=Output,
                     C2KVExtractBatchReqInput=Bulk, C2KVExtractBatchReqOutput=Output)
    controls = ['PauseGenerationReqInput', 'ContinueGenerationReqInput',
                'ReleaseMemoryOccupationReqInput', 'ResumeMemoryOccupationReqInput', 'RpcReqInput',
                'UpdateWeightFromDiskReqInput', 'UpdateWeightsFromDistributedReqInput',
                'UpdateWeightsFromIPCReqInput', 'UpdateWeightsFromTensorReqInput',
                'FlushCacheReqInput', 'TokenizedRepairExtractReqInput',
                'TokenizedGenerateReqInput', 'TokenizedEmbeddingReqInput',
                'BatchTokenizedGenerateReqInput', 'BatchTokenizedEmbeddingReqInput']
    namespace.update({name: type(name, (), {}) for name in controls})
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    cache = {}
    capacity = [True]
    free_tokens = [20]
    stored = []
    store_threads = []
    sync_calls = []

    def store(**kwargs):
        assert pending, "Temporary tensors must remain counted during publication"
        store_threads.append(threading.get_ident())
        entry = SimpleNamespace(gist_len=kwargs['gist_mask'].shape[1],
                                original_seq_len=kwargs['original_seq_len'])
        cache[kwargs['key_hash']] = entry
        stored.append(kwargs)
        return entry

    stepper = Stepper()

    def create_stepper(*args, **kwargs):
        nonlocal stepper
        if stepper.calls:
            stepper = Stepper()
        return stepper

    runner = SimpleNamespace(model=type('Qwen3ForCausalLM', (), {})(),
                             create_c2kv_extract_stepper=create_stepper)
    replies = []
    reply_threads = []

    def send_output(*args):
        replies.append(args)
        reply_threads.append(threading.get_ident())

    def sync_extract(request):
        sync_calls.append(request.rid)
        existing = cache.get(request.key)
        if existing is not None:
            return Output(key_hash=request.key, cache_hit=True,
                          gist_len=existing.gist_len,
                          original_seq_len=existing.original_seq_len)
        if not request.allow_cache_miss:
            return Output(key_hash=request.key, success=False, error='budget exhausted')
        if not capacity[0]:
            return Output(key_hash=request.key, success=False, error='capacity unavailable')
        raise AssertionError('An eligible foreground miss ran synchronously')

    def sync_bulk(request):
        hits = []
        for item in request.items:
            if item.key not in cache:
                break
            hits.append(sync_extract(item))
        first_miss = None
        if request.materialize_first_miss and len(hits) < len(request.items):
            first_miss = sync_extract(request.items[len(hits)])
        return Output(rid=request.rid, hits=hits, first_miss_index=len(hits),
                      first_miss_result=first_miss)

    scheduler = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=runner), enable_overlap=True,
        tp_size=1, pp_size=1, device="cuda", server_args=SimpleNamespace(dp_size=1),
        c2kv_pool=SimpleNamespace(_cache=cache, max_entry_tokens=10, max_total_tokens=20,
                                 allocator=SimpleNamespace(available_size=lambda: free_tokens[0]),
                                 can_allocate=lambda *args, **kw: capacity[0], store=store),
        _c2kv_extract_cache_key=lambda request: (request.key, getattr(request, 'ratio', 4), None),
        schedule_stream=Stream(), send_to_tokenizer=SimpleNamespace(send_output=send_output),
        handle_extract_request=sync_extract,
        handle_c2kv_bulk_cache_lookup=sync_bulk,
    )
    controller = namespace['AsyncGistExtraction'](scheduler)
    return SimpleNamespace(controller=controller, scheduler=scheduler, stepper=stepper,
                           replies=replies, stored=stored, pending=pending, capacity=capacity,
                           free_tokens=free_tokens,
                           namespace=namespace, recorded=recorded, worker_devices=worker_devices,
                           pending_snapshots=pending_snapshots,
                           store_threads=store_threads, reply_threads=reply_threads,
                           sync_calls=sync_calls)


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


def packed_group(lengths=(4, 7, 2)):
    items = [Request(rid=f'item{i}', key=f'key{i}', input_ids=list(range(length)))
             for i, length in enumerate(lengths)]
    return Bulk(rid='batch', items=items, background_extraction=True,
                c2kv_outer_request_id='outer')


def attach_packed_forward(state, *, error=None):
    calls = []
    layers = []

    def forward(documents, ratio, *, projection_set, on_layer_kv):
        calls.append((documents, ratio, projection_set, threading.get_ident()))
        per_document_kv = [[] for _ in documents]
        for _ in range(2):
            layer = tuple((object(), object()) for _ in documents)
            for document, kv in zip(per_document_kv, layer):
                document.append(kv)
            on_layer_kv(per_document_kv)
            layers.append(layer)
            if error is not None:
                raise RuntimeError(error)
        return [(document, SimpleNamespace(shape=(1, (len(ids) + ratio - 1) // ratio)),
                 object()) for document, ids in zip(per_document_kv, documents)]

    state.scheduler.tp_worker.model_runner.forward_c2kv_extract_many = forward
    return calls, layers


def test_packed_worker_publishes_ordered_mixed_lengths_after_event(monkeypatch):
    state = make_controller(monkeypatch)
    calls, layers = attach_packed_forward(state)
    group = packed_group()
    assert state.controller.intercept(group)
    state.controller.advance(decode_dispatched=True)
    result = submitted_result(state.controller)
    assert len(calls) == 1
    assert calls[0][:3] == ([item.input_ids for item in group.items], 4, 'history')
    assert calls[0][3] != threading.get_ident()
    assert result.model_calls == len(group.items)
    assert state.pending_snapshots == [layers[0], layers[0] + layers[1]]
    assert list(state.pending) == [group.rid]

    state.controller.advance(decode_dispatched=True)
    assert state.controller.busy and not state.stored and not state.replies
    result.end_event.ready = True
    state.controller.advance(decode_dispatched=True)
    assert [entry['key_hash'] for entry in state.stored] == [item.key for item in group.items]
    assert [entry['original_seq_len'] for entry in state.stored] == [4, 7, 2]
    assert [entry['gist_mask'].shape[1] for entry in state.stored] == [1, 2, 1]
    assert state.store_threads == [threading.get_ident()] * len(group.items)
    assert state.reply_threads == [threading.get_ident()]
    assert state.scheduler.schedule_stream.syncs == 1
    assert not state.pending and not state.controller.busy

    reply, owner = state.replies[0]
    assert owner is group and reply.success
    assert reply.attempted_model_calls == len(group.items)
    assert reply.paper_measurement['metrics']['model_calls'] == len(group.items)
    assert reply.paper_measurement['metrics']['packed_forward_calls'] == 1
    assert [item.rid for item in reply.items] == [item.rid for item in group.items]
    assert [item.gist_len for item in reply.items] == [1, 2, 1]
    assert all(item.success and item.extraction_batch_id == group.rid
               and item.extraction_batch_size == len(group.items)
               and item.shared_gist_generation_duration_ns > 0
               and getattr(item, 'gist_generation_duration_ns', None) is None
               for item in reply.items)


@pytest.mark.parametrize('reason', [
    'single', 'five', 'duplicate', 'cache_hit', 'ratio', 'projection',
    'raw_tokens', 'gist_capacity', 'entry_capacity', 'foreground_item', 'no_packed_api',
])
def test_unsafe_packed_group_is_not_admitted(monkeypatch, reason):
    state = make_controller(monkeypatch)
    group = packed_group((4, 4))
    if reason != 'no_packed_api':
        attach_packed_forward(state)
    if reason == 'single':
        group.items.pop()
    elif reason == 'five':
        group = packed_group((4, 4, 4, 4, 4))
    elif reason == 'duplicate':
        group.items[1].key = group.items[0].key
    elif reason == 'cache_hit':
        state.scheduler.c2kv_pool._cache[group.items[1].key] = object()
    elif reason == 'ratio':
        group.items[1].ratio = 8
    elif reason == 'projection':
        group.items[1].projection_set = 'tool'
    elif reason == 'raw_tokens':
        group = packed_group((2049, 2048))
        for item in group.items:
            item.ratio = 1024
    elif reason == 'gist_capacity':
        state.free_tokens[0] = 1
    elif reason == 'entry_capacity':
        group.items[1].input_ids = list(range(8))
        state.scheduler.c2kv_pool.max_entry_tokens = 1
    elif reason == 'foreground_item':
        group.items[1].background_extraction = False
    assert not state.controller.intercept(group)
    assert state.controller.job is None and not state.recorded


def test_packed_job_defers_every_owned_key_and_control_barrier(monkeypatch):
    state = make_controller(monkeypatch)
    attach_packed_forward(state)
    group = packed_group()
    assert state.controller.intercept(group)
    second_key = Request(rid='foreground', key=group.items[1].key,
                         background_extraction=False)
    control = state.namespace['UpdateWeightsFromTensorReqInput']()
    assert state.controller.intercept(second_key)
    assert state.controller.intercept(control)
    assert not state.controller.intercept(Request(rid='unrelated', key='other',
                                                  background_extraction=False))
    assert state.controller.take_ready_requests() == []
    state.controller.advance()
    finish_job(state.controller)
    assert state.controller.take_ready_requests() == [second_key, control]


def test_packed_capacity_lost_after_compute_fails_every_item(monkeypatch):
    state = make_controller(monkeypatch)
    calls, _ = attach_packed_forward(state)
    group = packed_group()
    assert state.controller.intercept(group)
    state.controller.advance()
    result = submitted_result(state.controller)
    state.free_tokens[0] = 0
    result.end_event.ready = True
    state.controller.advance()
    reply = state.replies[0][0]
    assert [item.rid for item in reply.items] == [item.rid for item in group.items]
    assert all(not item.success and 'capacity' in item.error for item in reply.items)
    assert not reply.success and reply.attempted_model_calls == len(group.items)
    assert len(calls) == 1 and not state.stored and not state.pending


def test_packed_partial_store_error_keeps_success_prefix_and_attempt_count(monkeypatch):
    state = make_controller(monkeypatch)
    calls, _ = attach_packed_forward(state)
    group = packed_group()
    original_store = state.scheduler.c2kv_pool.store

    def store(**kwargs):
        if kwargs['key_hash'] == group.items[1].key:
            raise ValueError('copy failed')
        return original_store(**kwargs)

    state.scheduler.c2kv_pool.store = store
    assert state.controller.intercept(group)
    state.controller.advance()
    finish_job(state.controller)
    reply = state.replies[0][0]
    assert [item.success for item in reply.items] == [True, False, False]
    assert [item.rid for item in reply.items] == [item.rid for item in group.items]
    assert all('copy failed' in item.error for item in reply.items[1:])
    assert not reply.success and reply.attempted_model_calls == len(group.items)
    assert len(calls) == 1 and [item['key_hash'] for item in state.stored] == ['key0']
    assert state.scheduler.schedule_stream.syncs == 1 and not state.pending


def test_packed_forward_exception_replies_once_without_singleton_retry(monkeypatch):
    state = make_controller(monkeypatch)
    calls, _ = attach_packed_forward(state, error='packed kernel failed')
    group = packed_group()
    assert state.controller.intercept(group)
    state.controller.advance()
    result = submitted_result(state.controller)
    assert 'packed kernel failed' in result.error
    assert state.controller.busy and not state.replies
    result.end_event.ready = True
    state.controller.advance()
    reply = state.replies[0][0]
    assert [item.rid for item in reply.items] == [item.rid for item in group.items]
    assert all(not item.success and 'packed kernel failed' in item.error
               for item in reply.items)
    assert reply.attempted_model_calls == len(group.items)
    assert len(calls) == 1 and state.stepper.calls == 0 and not state.stored
    assert not state.pending and not state.controller.busy


def test_packed_worker_does_not_block_foreground_decode_advance(monkeypatch):
    state = make_controller(monkeypatch)
    group = packed_group((4, 4))
    entered = threading.Event()
    release = threading.Event()

    def forward(documents, ratio, *, projection_set, on_layer_kv):
        entered.set()
        assert release.wait(timeout=5)
        per_document_kv = [[(object(), object())] for _ in documents]
        on_layer_kv(per_document_kv)
        return [(document, SimpleNamespace(shape=(1, 1)), object())
                for document in per_document_kv]

    state.scheduler.tp_worker.model_runner.forward_c2kv_extract_many = forward
    assert state.controller.intercept(group)
    try:
        state.controller.advance(decode_dispatched=True)
        assert entered.wait(timeout=2)
        started = time.perf_counter()
        for _ in range(3):
            state.controller.advance(decode_dispatched=True)
        assert time.perf_counter() - started < 0.2
        assert state.controller.busy and not state.replies
    finally:
        release.set()
    result = submitted_result(state.controller)
    result.end_event.ready = True
    state.controller.advance()
    assert state.replies[0][0].success
    assert state.replies[0][0].paper_measurement['metrics']['gist_overlap_decode_batches'] == 4


def test_unsupported_background_batch_scheduler_guard_requests_singleton_retry():
    source = SOURCE.with_name('scheduler.py')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    scheduler_class = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == 'Scheduler')
    handler = next(node for node in scheduler_class.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == 'handle_c2kv_extract_batch')
    namespace = {'C2KVExtractBatchReqInput': Bulk, 'C2KVExtractBatchReqOutput': Output}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[handler], type_ignores=[])),
                 str(source), 'exec'), namespace)
    group = packed_group()
    reply = namespace['handle_c2kv_extract_batch'](SimpleNamespace(), group)
    assert reply.rid == group.rid and not reply.success
    assert reply.retry_individually and reply.attempted_model_calls == 0
    assert getattr(reply, 'items', []) == []


def test_model_runner_passes_packed_layer_callback_to_qwen3():
    source = SOURCE.parents[1] / 'model_executor/model_runner.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    runner_class = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == 'ModelRunner')
    method = next(node for node in runner_class.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == 'forward_c2kv_extract_many')
    namespace = {'List': list}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(source), 'exec'), namespace)
    observed = []
    layer_callback = lambda values: observed.append(values)

    def generate(documents, *, ratio, projection_set, on_layer_kv):
        on_layer_kv(('layer0', 'layer1'))
        return documents, ratio, projection_set

    model = type('Qwen3ForCausalLM', (), {'generate_gist_many': staticmethod(generate)})()
    runner = SimpleNamespace(model=model, get_c2kv_compression_ratio=lambda ratio: ratio)
    forward = namespace['forward_c2kv_extract_many']
    assert forward(runner, [[1], [2, 3]], 4, projection_set='history',
                   on_layer_kv=layer_callback) == ([[1], [2, 3]], 4, 'history')
    assert observed == [('layer0', 'layer1')]


def test_foreground_flag_off_preserves_dispatch_to_sync_handlers(monkeypatch):
    state = make_controller(monkeypatch)
    foreground = Request(rid='selected', background_extraction=False)
    lookup = BulkLookup(rid='lookup', items=[foreground], materialize_first_miss=True)
    envelope = Bulk(rid='envelope', items=[foreground], background_extraction=False)
    assert not state.controller.intercept(foreground)
    assert not state.controller.intercept(lookup)
    assert not state.controller.intercept(envelope)
    assert state.controller.job is None and not state.replies


def test_foreground_miss_keeps_scheduler_free_and_replies_after_publication(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    entered = threading.Event()
    release = threading.Event()
    original_step = state.stepper.step

    def blocked_step():
        entered.set()
        assert release.wait(timeout=5)
        return original_step()

    state.stepper.step = blocked_step
    selected = Request(rid='selected', background_extraction=False)
    assert state.controller.intercept(selected)
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
    state.controller.advance()
    assert not state.replies
    result.end_event.ready = True
    state.controller.advance()
    assert len(state.replies) == 1
    output, owner = state.replies[0]
    assert owner is selected and output.success and output.key_hash == selected.key
    assert state.store_threads == state.reply_threads == [threading.get_ident()]
    assert not state.sync_calls


def test_foreground_same_key_success_waiter_uses_published_cache(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    selected = Request(rid='selected', key='shared', background_extraction=False)
    waiter = Request(rid='waiter', key='shared', background_extraction=False)
    assert state.controller.intercept(selected)
    assert state.controller.intercept(waiter)
    state.controller.advance()
    finish_job(state.controller)
    assert state.controller.take_ready_requests() == [waiter]
    assert not state.controller.intercept(waiter)
    cached = state.scheduler.handle_extract_request(waiter)
    assert cached.success and cached.cache_hit and cached.key_hash == 'shared'
    assert state.sync_calls == ['waiter']
    assert [item['key_hash'] for item in state.stored] == ['shared']


def test_fused_bulk_preserves_hit_prefix_and_original_reply_owner(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    state.scheduler.c2kv_pool._cache['hit'] = SimpleNamespace(
        gist_len=2, original_seq_len=4)
    hit = Request(rid='hit-rid', key='hit', background_extraction=False)
    miss = Request(rid='miss-rid', key='miss', background_extraction=False)
    tail = Request(rid='tail-rid', key='tail', background_extraction=False)
    lookup = BulkLookup(rid='lookup', items=[hit, miss, tail],
                        materialize_first_miss=True)
    assert state.controller.intercept(lookup)
    assert state.sync_calls == ['hit-rid']
    state.controller.advance()
    finish_job(state.controller)
    assert len(state.replies) == 1
    output, owner = state.replies[0]
    assert owner is lookup and output.rid == lookup.rid
    assert output.first_miss_index == 1
    assert [item.key_hash for item in output.hits] == ['hit']
    assert output.first_miss_result.rid == miss.rid
    assert output.first_miss_result.success and output.first_miss_result.key_hash == 'miss'
    assert state.sync_calls == ['hit-rid']
    assert [item['key_hash'] for item in state.stored] == ['miss']


def test_foreground_batch_keeps_mixed_reply_order_and_item_error(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    state.scheduler.c2kv_pool._cache['hit'] = SimpleNamespace(
        gist_len=1, original_seq_len=4)
    hit = Request(rid='hit-rid', key='hit', background_extraction=False)
    miss = Request(rid='miss-rid', key='miss', background_extraction=False)
    lookup = BulkLookup(rid='lookup', items=[hit, miss], materialize_first_miss=True)
    denied = Request(rid='denied', key='denied', background_extraction=False,
                     allow_cache_miss=False)
    envelope = Bulk(rid='envelope', items=[lookup, denied],
                    background_extraction=False)
    assert state.controller.intercept(envelope)
    assert not state.replies and state.sync_calls == ['hit-rid']
    state.controller.advance()
    finish_job(state.controller)
    assert len(state.replies) == 1
    output, owner = state.replies[0]
    assert owner is envelope and output.success
    assert [item.rid for item in output.items] == ['lookup', 'denied']
    assert output.items[0].first_miss_result.success
    assert not output.items[1].success and output.items[1].error == 'budget exhausted'
    assert state.sync_calls == ['hit-rid', 'denied']


def test_foreground_batch_cursor_runs_successive_fused_misses_in_order(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    first = BulkLookup(rid='first', items=[Request(rid='a', key='a')],
                       materialize_first_miss=True)
    second = BulkLookup(rid='second', items=[Request(rid='b', key='b')],
                        materialize_first_miss=True)
    envelope = Bulk(rid='envelope', items=[first, second],
                    background_extraction=False)
    assert state.controller.intercept(envelope)
    state.controller.advance()
    finish_job(state.controller)
    assert state.controller.job is not None and not state.replies
    state.controller.advance()
    finish_job(state.controller)
    assert len(state.replies) == 1
    output, owner = state.replies[0]
    assert owner is envelope and output.success
    assert [item.rid for item in output.items] == ['first', 'second']
    assert [item.first_miss_result.key_hash for item in output.items] == ['a', 'b']
    assert [item['key_hash'] for item in state.stored] == ['a', 'b']


def test_foreground_batch_packs_independent_misses_without_changing_envelope_contract(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    calls, _ = attach_packed_forward(state)
    items = [Request(rid=f'selected-{i}', key=f'key-{i}',
                     background_extraction=False) for i in range(2)]
    envelope = Bulk(rid=None, items=items,
                    background_extraction=False,
                    c2kv_outer_request_id='outer',
                    c2kv_measurement_phase='selected:extraction_batch')
    assert state.controller.intercept(envelope)
    state.controller.advance()
    finish_job(state.controller)
    assert len(calls) == 1 and calls[0][3] != threading.get_ident()
    assert len(state.replies) == 1
    output, owner = state.replies[0]
    assert owner is envelope and output.success and output.rid is None
    assert [item.rid for item in output.items] == [item.rid for item in items]
    assert [item.key_hash for item in output.items] == [item.key for item in items]
    assert output.attempted_model_calls == 2
    batch_id = output.items[0].extraction_batch_id
    assert batch_id.startswith('c2kv-foreground-batch:')
    assert all(item.extraction_batch_id == batch_id for item in output.items)
    assert any(kind == 'start' and fields['server_request_id'] == batch_id
               for kind, fields in state.recorded if kind == 'start')


def test_failed_same_key_waiters_share_completed_error(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    selected = Request(rid='selected', key='shared', background_extraction=False)
    waiter = Request(rid='waiter', key='shared', background_extraction=False)
    bulk = BulkLookup(rid='bulk', items=[Request(rid='nested', key='shared')],
                      materialize_first_miss=True)
    limited = Request(rid='limited', key='shared', background_extraction=False,
                      allow_cache_miss=False)
    assert state.controller.intercept(selected)
    assert state.controller.intercept(waiter)
    assert state.controller.intercept(bulk)
    assert state.controller.intercept(limited)
    state.stepper.fail_at = 2
    state.controller.advance()
    finish_job(state.controller)
    assert [owner for _, owner in state.replies] == [selected]
    assert state.controller.take_ready_requests() == [waiter, bulk, limited]
    assert state.controller.intercept(waiter)
    assert state.controller.intercept(bulk)
    assert not state.controller.intercept(limited)
    assert id(limited) not in state.controller._replay_outcomes
    budget_result = state.scheduler.handle_extract_request(limited)
    assert not budget_result.success and budget_result.error == 'budget exhausted'
    assert state.controller.job is None
    assert [owner for _, owner in state.replies] == [selected, waiter, bulk]
    assert all(not output.success for output, _ in state.replies[:2])
    assert not state.replies[2][0].first_miss_result.success
    assert 'layer failed' in state.replies[2][0].first_miss_result.error
    assert state.sync_calls == ['limited'] and not state.stored


def test_failed_waiter_uses_later_cache_hit_and_releases_failed_memo(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    active = Request(rid='active', key='shared', background_extraction=False)
    waiter = Request(rid='waiter', key='shared', background_extraction=False)
    assert state.controller.intercept(active)
    assert state.controller.intercept(waiter)
    state.stepper.fail_at = 2
    state.controller.advance()
    finish_job(state.controller)
    assert id(waiter) in state.controller._replay_outcomes
    state.scheduler.c2kv_pool._cache['shared'] = SimpleNamespace(
        gist_len=1, original_seq_len=4)
    assert state.controller.take_ready_requests() == [waiter]
    assert not state.controller.intercept(waiter)
    assert id(waiter) not in state.controller._replay_outcomes
    assert state.scheduler.handle_extract_request(waiter).cache_hit


def test_failed_job_is_not_reused_across_mutation_barrier(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    active = Request(rid='active', key='shared', background_extraction=False)
    waiter = Request(rid='waiter', key='shared', background_extraction=False)
    mutation = state.namespace['UpdateWeightsFromTensorReqInput']()
    assert state.controller.intercept(active)
    assert state.controller.intercept(mutation)
    assert state.controller.intercept(waiter)
    state.stepper.fail_at = 2
    state.controller.advance()
    finish_job(state.controller)
    assert id(waiter) not in state.controller._replay_outcomes
    assert state.controller.take_ready_requests() == [mutation, waiter]
    assert state.controller.intercept(waiter)
    assert state.controller.job.request is waiter


def test_failed_packed_background_key_does_not_rerun_foreground_waiter(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    calls, _ = attach_packed_forward(state, error='packed kernel failed')
    group = packed_group((4, 4))
    waiter = Request(rid='selected', key=group.items[1].key,
                     background_extraction=False)
    assert state.controller.intercept(group)
    assert state.controller.intercept(waiter)
    state.controller.advance()
    finish_job(state.controller)
    assert state.controller.take_ready_requests() == [waiter]
    assert state.controller.intercept(waiter)
    assert len(calls) == 1 and state.controller.job is None
    assert len(state.replies) == 2
    output, owner = state.replies[1]
    assert owner is waiter and not output.success
    assert 'packed kernel failed' in output.error
    assert output.paper_measurement['metrics']['gist_execution_mode'] == (
        'tp1-worker-shared-failure')
    assert output.extraction_batch_id is None
    assert output.shared_gist_generation_duration_ns is None


def test_foreground_priority_respects_mutation_barrier(monkeypatch):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    assert state.controller.intercept(Request(rid='active', key='active'))
    background_before = Request(rid='bg-before', key='bg-before')
    selected_before = Request(rid='selected-before', key='selected-before',
                              background_extraction=False)
    mutation = state.namespace['FlushCacheReqInput']()
    background_after = Request(rid='bg-after', key='bg-after')
    selected_after = Request(rid='selected-after', key='selected-after',
                             background_extraction=False)
    later_generation = state.namespace['TokenizedGenerateReqInput']()
    for request in (background_before, selected_before, mutation,
                    background_after, selected_after, later_generation):
        assert state.controller.intercept(request)
    state.controller.advance()
    finish_job(state.controller)
    assert state.controller.take_ready_requests() == [
        selected_before, background_before, mutation, selected_after,
        later_generation, background_after]


@pytest.mark.parametrize('restriction', ['tp2', 'pic', 'lora', 'capacity'])
def test_foreground_unsupported_configuration_uses_existing_fallback(monkeypatch, restriction):
    state = make_controller(monkeypatch)
    monkeypatch.setenv('C2KV_GIST_ASYNC_FOREGROUND', '1')
    if restriction == 'tp2':
        state.scheduler.tp_size = 2
    elif restriction == 'pic':
        state.scheduler.tp_worker.model_runner.model.full_length_pic = True
    elif restriction == 'lora':
        state.scheduler.server_args.enable_lora = True
    else:
        state.capacity[0] = False
    selected = Request(rid='selected', background_extraction=False)
    lookup = BulkLookup(rid='lookup', items=[selected], materialize_first_miss=True)
    assert not state.controller.intercept(selected)
    assert not state.controller.intercept(lookup)
    assert state.controller.job is None and not state.replies
