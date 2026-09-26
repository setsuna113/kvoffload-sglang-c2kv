"""One TP1 gist job on a bounded worker beside the scheduler's decode stream.

Only the scheduler thread mutates jobs and the C2KV pool. The worker owns the
stepper and its dedicated CUDA stream; cache publication remains on the scheduler.
"""

from __future__ import annotations

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch
from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    C2KVBulkCacheLookupReqInput,
    C2KVBulkCacheLookupReqOutput,
    C2KVExtractBatchReqInput,
    C2KVExtractBatchReqOutput,
    C2KVExtractReqOutput,
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedExtractReqInput,
    TokenizedGenerateReqInput,
    TokenizedRepairExtractReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.observability import paper_telemetry


@dataclass
class _Job:
    request: object
    key: str
    ratio: int
    started_ns: int = field(default_factory=time.perf_counter_ns)
    future: object = None
    result: object = None
    decode_batches: int = 0
    scheduler_ticks: int = 0
    keys: tuple = ()
    packed: bool = False
    reply_owner: object = None
    bulk_hits: object = None
    bulk_index: int = 0


@dataclass
class _ForegroundBatch:
    request: object
    outputs: list = field(default_factory=list)
    index: int = 0


@dataclass
class _WorkerResult:
    # Published to the scheduler only after Future.done() is true.
    stepper: object = None
    inputs: object = None
    start_event: object = None
    end_event: object = None
    error: str | None = None
    ticks: int = 0
    # Host elapsed: these phases partition cpu_step_ns; telemetry is inside layers.
    cpu_step_ns: int = 0
    cpu_step_max_ns: int = 0
    cpu_prelude_ns: int = 0
    cpu_layers_ns: int = 0
    cpu_finalize_ns: int = 0
    cpu_telemetry_ns: int = 0
    packed_results: object = None
    model_calls: int = 0


class AsyncGistExtraction:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._executor = None
        self._worker_stream = None
        self.job = None
        self.deferred = []
        self.foreground_batch = None
        self._replay_outcomes = {}

    @property
    def busy(self):
        return self.job is not None or self.foreground_batch is not None or bool(self.deferred)

    def foreground_enabled(self):
        return os.environ.get("C2KV_GIST_ASYNC_FOREGROUND", "").lower() in {
            "1", "true", "yes", "on"
        }

    def _batch_id(self, request):
        return request.rid or f"c2kv-foreground-batch:{id(request)}"

    def supported(self):
        scheduler = self.scheduler
        runner = scheduler.tp_worker.model_runner
        model = runner.model
        return (
            os.environ.get("C2KV_GIST_ASYNC", "").lower() in {"1", "true", "yes", "on"}
            and scheduler.tp_size == 1
            and scheduler.pp_size == 1
            and getattr(scheduler.server_args, "dp_size", 1) == 1
            and scheduler.device == "cuda"
            and scheduler.c2kv_pool is not None
            and not getattr(scheduler, "_engine_paused", False)
            and model.__class__.__name__ == "Qwen3ForCausalLM"
            and not getattr(model, "full_length_pic", False)
            and not getattr(scheduler.server_args, "enable_lora", False)
            and not os.environ.get("C2KV_DEBUG_GIST_DUMP")
            and (not paper_telemetry.enabled() or paper_telemetry.concurrent_enabled())
        )

    def _contains_key(self, request, key):
        if isinstance(request, TokenizedExtractReqInput):
            observed, _, error = self.scheduler._c2kv_extract_cache_key(request)
            return error is None and observed == key
        if isinstance(request, (C2KVBulkCacheLookupReqInput, C2KVExtractBatchReqInput)):
            return any(self._contains_key(item, key) for item in request.items)
        return False

    def _foreground_barrier(self, request):
        return not self._foreground_work(request) and not isinstance(request, (
            TokenizedGenerateReqInput, TokenizedEmbeddingReqInput,
            BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput,
        ))

    def _foreground_work(self, request):
        return isinstance(request, (
            TokenizedExtractReqInput, C2KVBulkCacheLookupReqInput,
            C2KVExtractBatchReqInput, TokenizedRepairExtractReqInput,
        ))

    def _admission(self, request):
        if not self.supported() or not request.input_ids or not request.allow_cache_miss:
            return None
        key, ratio, error = self.scheduler._c2kv_extract_cache_key(request)
        if error is not None:
            return None
        pool = self.scheduler.c2kv_pool
        if key in pool._cache:
            return None
        gist_len = (len(request.input_ids) + ratio - 1) // ratio
        if gist_len > min(pool.max_entry_tokens, pool.max_total_tokens):
            return None
        if not pool.can_allocate(gist_len, existing_key=key):
            return None
        return key, ratio

    def _start_single(self, request):
        admission = self._admission(request)
        if admission is None:
            return False
        key, ratio = admission
        self.job = _Job(request, key, ratio)
        paper_telemetry.start_request(
            server_request_id=request.rid,
            outer_request_id=request.c2kv_outer_request_id,
            phase=request.c2kv_measurement_phase or "extraction",
            kind="c2kv_extract",
            whole_full_kv_tokens=len(request.input_ids),
        )
        return True

    def _failed_replay(self, request, key, *, item=None):
        item = item or request
        if not item.allow_cache_miss or key in self.scheduler.c2kv_pool._cache:
            return None
        for owner in (request, getattr(self.foreground_batch, "request", None)):
            if owner is None:
                continue
            record = self._replay_outcomes.get(id(owner))
            if record is not None and record[0] is owner and key in record[1]:
                original = record[1][key]
                replay = copy.copy(original)
                replay.rid = item.rid
                replay.paper_measurement = None
                replay.extraction_duration_ns = None
                replay.gist_generation_duration_ns = 0
                replay.extraction_batch_id = None
                replay.extraction_batch_size = None
                replay.shared_gist_generation_duration_ns = None
                paper_telemetry.start_request(
                    server_request_id=item.rid,
                    outer_request_id=item.c2kv_outer_request_id,
                    phase=item.c2kv_measurement_phase or "extraction",
                    kind="c2kv_extract",
                    whole_full_kv_tokens=len(item.input_ids),
                )
                measurement = paper_telemetry.finish_request(
                    server_request_id=item.rid, success=False,
                    error=replay.error,
                    metric_overrides={
                        "cache_hit": False,
                        "gist_execution_mode": "tp1-worker-shared-failure",
                    },
                )
                if measurement is not None:
                    replay.paper_measurement = measurement
                    replay.extraction_duration_ns = measurement["duration_ns"]
                return replay
        return None

    def _remember_failure(self, job, output, *, key=None):
        if output.success or not self.foreground_enabled():
            return
        key = key or job.key
        before_barrier = []
        for request in self.deferred:
            if self._foreground_barrier(request):
                break
            before_barrier.append(request)
        if self.foreground_batch is not None:
            before_barrier.append(self.foreground_batch.request)
        for request in before_barrier:
            if self._contains_key(request, key):
                record = self._replay_outcomes.setdefault(id(request), (request, {}))
                record[1][key] = output

    def _start_bulk(self, request):
        if not getattr(request, "materialize_first_miss", False):
            return False
        if not 1 <= len(request.items) <= 32:
            return False
        hit_count = 0
        for item in request.items:
            if not item.input_ids:
                return False
            key, _, error = self.scheduler._c2kv_extract_cache_key(item)
            if error is not None:
                return False
            if key not in self.scheduler.c2kv_pool._cache:
                break
            hit_count += 1
        if hit_count == len(request.items):
            return False
        miss = request.items[hit_count]
        key, _, _ = self.scheduler._c2kv_extract_cache_key(miss)
        failed = self._failed_replay(request, key, item=miss)
        if failed is None and self._admission(miss) is None:
            return False
        # Match the synchronous handler's hit-prefix LRU touches before the miss.
        hits = [self.scheduler.handle_extract_request(item)
                for item in request.items[:hit_count]]
        if any(not hit.success or not hit.cache_hit for hit in hits):
            return C2KVBulkCacheLookupReqOutput(
                rid=request.rid, success=False,
                error="C2KV bulk cache lookup lost a cache hit",
            )
        if failed is not None:
            return C2KVBulkCacheLookupReqOutput(
                rid=request.rid, hits=hits, first_miss_index=hit_count,
                first_miss_result=failed,
            )
        if not self._start_single(miss):
            # Only the scheduler thread can change pool admission; no other
            # scheduler operation runs between the two admission checks.
            raise RuntimeError("C2KV foreground admission changed during bulk lookup")
        self.job.reply_owner = request
        self.job.bulk_hits = hits
        self.job.bulk_index = hit_count
        return True

    def _deliver_foreground(self, output, owner):
        batch = self.foreground_batch
        if batch is None:
            self.scheduler.send_to_tokenizer.send_output(output, owner)
            self._replay_outcomes.pop(id(owner), None)
            return
        output.rid = owner.rid
        batch.outputs.append(output)
        batch.index += 1
        self._replay_outcomes.pop(id(owner), None)
        self._drive_foreground_batch()

    def _drive_foreground_batch(self):
        batch = self.foreground_batch
        while batch is not None and self.job is None and batch.index < len(batch.request.items):
            item = batch.request.items[batch.index]
            if isinstance(item, C2KVBulkCacheLookupReqInput):
                bulk_result = self._start_bulk(item)
                if bulk_result is True:
                    return
                output = (self.scheduler.handle_c2kv_bulk_cache_lookup(item)
                          if bulk_result is False else bulk_result)
            else:
                key, _, error = self.scheduler._c2kv_extract_cache_key(item)
                failed = (None if error is not None else
                          self._failed_replay(batch.request, key, item=item))
                if failed is not None:
                    output = failed
                elif self._start_single(item):
                    return
                else:
                    output = self.scheduler.handle_extract_request(item)
            output.rid = item.rid
            batch.outputs.append(output)
            batch.index += 1
        if batch is not None and self.job is None and batch.index == len(batch.request.items):
            self.foreground_batch = None
            self.scheduler.send_to_tokenizer.send_output(
                C2KVExtractBatchReqOutput(items=batch.outputs, success=True), batch.request
            )
            self._replay_outcomes.pop(id(batch.request), None)

    def _intercept_foreground(self, request):
        if self.job is not None or self.foreground_batch is not None:
            barrier_pending = any(self._foreground_barrier(queued)
                                  for queued in self.deferred)
            if (self._foreground_work(request) or self._foreground_barrier(request)
                    or barrier_pending):
                self.deferred.append(request)
                return True
        if isinstance(request, TokenizedExtractReqInput):
            key, _, error = self.scheduler._c2kv_extract_cache_key(request)
            failed = None if error is not None else self._failed_replay(request, key)
            if failed is not None:
                self.scheduler.send_to_tokenizer.send_output(failed, request)
                self._replay_outcomes.pop(id(request), None)
                return True
            accepted = self._start_single(request)
            if not accepted:
                self._replay_outcomes.pop(id(request), None)
            return accepted
        if isinstance(request, C2KVBulkCacheLookupReqInput):
            if not self.supported():
                self._replay_outcomes.pop(id(request), None)
                return False
            bulk_result = self._start_bulk(request)
            if bulk_result is True or bulk_result is False:
                if bulk_result is False:
                    self._replay_outcomes.pop(id(request), None)
                return bulk_result
            self.scheduler.send_to_tokenizer.send_output(bulk_result, request)
            self._replay_outcomes.pop(id(request), None)
            return True
        if isinstance(request, C2KVExtractBatchReqInput):
            if getattr(request, "background_extraction", False):
                accepted = self._intercept_batch(request)
                if not accepted:
                    self._replay_outcomes.pop(id(request), None)
                return accepted
            if not self.supported():
                self._replay_outcomes.pop(id(request), None)
                return False
            if self._intercept_batch(request, foreground=True):
                return True
            self.foreground_batch = _ForegroundBatch(request)
            self._drive_foreground_batch()
            return True
        return False

    def intercept(self, request):
        """Return True only when ownership of the eventual reply is retained."""
        if self.foreground_enabled():
            return self._intercept_foreground(request)
        if self.job is not None and (
            any(self._contains_key(request, key)
                for key in (self.job.keys or (self.job.key,)))
            or isinstance(request, (
                PauseGenerationReqInput, ContinueGenerationReqInput,
                ReleaseMemoryOccupationReqInput, ResumeMemoryOccupationReqInput, RpcReqInput,
                UpdateWeightFromDiskReqInput, UpdateWeightsFromDistributedReqInput,
                UpdateWeightsFromIPCReqInput, UpdateWeightsFromTensorReqInput,
            ))
            or (isinstance(request, TokenizedExtractReqInput)
                and getattr(request, "background_extraction", False))
            or (isinstance(request, C2KVExtractBatchReqInput)
                and getattr(request, "background_extraction", False))
        ):
            self.deferred.append(request)
            return True
        if isinstance(request, C2KVExtractBatchReqInput) and getattr(
            request, "background_extraction", False
        ):
            return self._intercept_batch(request)
        if not isinstance(request, TokenizedExtractReqInput):
            return False
        if not getattr(request, "background_extraction", False) or not self.supported():
            return False
        return self._start_single(request)

    def _intercept_batch(self, request, *, foreground=False):
        """Accept only an immediately runnable group of independent misses."""
        if not self.supported() or not 2 <= len(request.items) <= 4:
            return False
        runner = self.scheduler.tp_worker.model_runner
        if not callable(getattr(runner, "forward_c2kv_extract_many", None)):
            return False
        pool = self.scheduler.c2kv_pool
        keys = []
        ratio = projection = None
        raw_tokens = gist_tokens = 0
        for item in request.items:
            if (not isinstance(item, TokenizedExtractReqInput)
                    or not item.input_ids or not item.allow_cache_miss
                    or bool(getattr(item, "background_extraction", False)) == foreground):
                return False
            key, item_ratio, error = self.scheduler._c2kv_extract_cache_key(item)
            item_projection = item.projection_set or "history"
            if (error is not None or key in keys or key in pool._cache
                    or (keys and (item_ratio != ratio or item_projection != projection))):
                return False
            gist_len = (len(item.input_ids) + item_ratio - 1) // item_ratio
            raw_tokens += len(item.input_ids)
            gist_tokens += gist_len
            if (raw_tokens > 4096
                    or gist_len > min(pool.max_entry_tokens, pool.max_total_tokens)
                    or gist_tokens > pool.allocator.available_size()):
                return False
            keys.append(key)
            ratio, projection = item_ratio, item_projection
        self.job = _Job(request, keys[0], ratio, keys=tuple(keys), packed=True)
        paper_telemetry.start_request(
            server_request_id=self._batch_id(request),
            outer_request_id=request.c2kv_outer_request_id,
            phase=(getattr(request, "c2kv_measurement_phase", None) or "extraction_batch")
                  if foreground else "c2kv_cross_turn_prewarm:extraction_batch",
            kind="c2kv_extract_batch", whole_full_kv_tokens=raw_tokens,
        )
        for item in request.items:
            paper_telemetry.start_request(
                server_request_id=item.rid, outer_request_id=item.c2kv_outer_request_id,
                phase=item.c2kv_measurement_phase or "extraction", kind="c2kv_extract",
                whole_full_kv_tokens=len(item.input_ids),
            )
        return True

    def take_ready_requests(self):
        if self.job is not None or self.foreground_batch is not None:
            return []
        requests, self.deferred = self.deferred, []
        if self.foreground_enabled() and requests:
            ordered = []
            span = []

            def flush_span():
                ordered.extend(request for request in span if not getattr(
                    request, "background_extraction", False
                ))
                ordered.extend(request for request in span if getattr(
                    request, "background_extraction", False
                ))
                span.clear()

            for request in requests:
                if self._foreground_barrier(request):
                    flush_span()
                    ordered.append(request)
                else:
                    span.append(request)
            flush_span()
            return ordered
        return requests

    def advance(self, *, decode_dispatched=False):
        """Submit once, then poll without waiting for the worker or its GPU work."""
        job = self.job
        if job is None:
            return
        job.scheduler_ticks += 1
        if decode_dispatched:
            job.decode_batches += 1
        if job.future is None and job.result is None:
            try:
                # Record before submission so the worker observes preceding
                # scheduler-stream weight and pool writes.
                device = torch.cuda.current_device()
                dependency = torch.cuda.Event()
                dependency.record(self.scheduler.schedule_stream)
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="c2kv-gist"
                    )
                job.future = self._executor.submit(
                    self._run_batch if job.packed else self._run_job,
                    job.request, job.ratio, dependency, device
                )
            except Exception as exc:
                job.result = _WorkerResult(error=f"{type(exc).__name__}: {exc}")
            return

        if job.result is None:
            if not job.future.done():
                return
            try:
                job.result = job.future.result()
            except Exception as exc:
                job.result = _WorkerResult(error=f"{type(exc).__name__}: {exc}")
        if job.result.end_event is None or job.result.end_event.query():
            if job.packed:
                self._complete_batch(job)
            else:
                self._complete(job)

    def _run_batch(self, request, ratio, dependency, device):
        """Keep a packed forward and its temporary KV on the existing worker."""
        result = _WorkerResult()
        stream = None
        try:
            torch.cuda.set_device(device)
            if self._worker_stream is None:
                self._worker_stream = torch.cuda.Stream(device=device, priority=0)
            stream = self._worker_stream
            batch_id = self._batch_id(request)
            with torch.cuda.stream(stream), torch.no_grad(), paper_telemetry.request_scope(batch_id):
                stream.wait_event(dependency)
                result.start_event = torch.cuda.Event(enable_timing=True)
                result.end_event = torch.cuda.Event(enable_timing=True)
                result.start_event.record(stream)

                def register_layer(per_document_kv):
                    started = time.perf_counter_ns()
                    paper_telemetry.set_pending_tensors(
                        batch_id,
                        tuple(document[-1] for document in per_document_kv),
                        append=True,
                    )
                    result.cpu_telemetry_ns += time.perf_counter_ns() - started
                    result.ticks += 1

                started = time.perf_counter_ns()
                result.model_calls = len(request.items)
                try:
                    result.packed_results = self.scheduler.tp_worker.model_runner.forward_c2kv_extract_many(
                        [item.input_ids for item in request.items], ratio,
                        projection_set=request.items[0].projection_set or "history",
                        on_layer_kv=register_layer,
                    )
                    if len(result.packed_results) != len(request.items):
                        raise ValueError("C2KV async packed extraction returned the wrong item count")
                finally:
                    result.cpu_step_ns = time.perf_counter_ns() - started
                result.end_event.record(stream)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            if stream is not None:
                try:
                    with torch.cuda.stream(stream):
                        result.end_event = torch.cuda.Event()
                        result.end_event.record(stream)
                except Exception:
                    stream.synchronize()
                    result.end_event = None
        return result

    def _complete_batch(self, job):
        """Publish ordered results on the scheduler after the worker event."""
        scheduler, result = self.scheduler, job.result
        request = job.request
        batch_id = self._batch_id(request)
        duration_ns = time.perf_counter_ns() - job.started_ns
        device_interval_ns = None
        if result.error is None and result.start_event is not None and result.end_event is not None:
            device_interval_ns = int(result.start_event.elapsed_time(result.end_event) * 1e6)
        error = result.error
        outputs = []
        store_cpu_ns = sync_cpu_ns = 0
        try:
            if error is None:
                lengths = [packed[1].shape[1] for packed in result.packed_results]
                pool = scheduler.c2kv_pool
                if (any(length > min(pool.max_entry_tokens, pool.max_total_tokens)
                        for length in lengths)
                        or sum(lengths) > pool.allocator.available_size()):
                    raise ValueError("C2KV async packed result has no remaining unpinned capacity")
            with torch.cuda.stream(scheduler.schedule_stream):
                for index, (item, key) in enumerate(zip(request.items, job.keys)):
                    output = C2KVExtractReqOutput(
                        rid=item.rid, key_hash=key, success=False, error=error or "",
                        original_seq_len=len(item.input_ids), cache_hit=False,
                        extraction_batch_id=batch_id, extraction_batch_size=len(request.items),
                        shared_gist_generation_duration_ns=duration_ns,
                    )
                    if error is None:
                        try:
                            key_values, mask, positions = result.packed_results[index]
                            started = time.perf_counter_ns()
                            with paper_telemetry.request_scope(item.rid):
                                entry = scheduler.c2kv_pool.store(
                                    key_hash=key, gist_key_values=key_values, gist_mask=mask,
                                    gist_position_ids=positions, original_seq_len=len(item.input_ids),
                                )
                            store_cpu_ns += time.perf_counter_ns() - started
                            output.gist_len = entry.gist_len
                            output.success = True
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                            output.error = error
                    outputs.append(output)
                started = time.perf_counter_ns()
                scheduler.schedule_stream.synchronize()
                sync_cpu_ns = time.perf_counter_ns() - started
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            # A failed store barrier makes any earlier copy unconfirmed too.
            for output in outputs:
                output.success = False
                output.error = error
            for item, key in zip(request.items[len(outputs):], job.keys[len(outputs):]):
                outputs.append(C2KVExtractReqOutput(
                    rid=item.rid, key_hash=key, success=False, error=error,
                    original_seq_len=len(item.input_ids), cache_hit=False,
                    extraction_batch_id=batch_id, extraction_batch_size=len(request.items),
                    shared_gist_generation_duration_ns=duration_ns,
                ))
        finally:
            paper_telemetry.clear_pending_tensors(batch_id)
            result.packed_results = None
            key_values = mask = positions = None
        for item, output in zip(request.items, outputs):
            measurement = paper_telemetry.finish_request(
                server_request_id=item.rid, success=output.success, error=output.error or None,
                metric_overrides={
                    "cache_hit": False, "gist_execution_mode": "tp1-packed-worker-stream-v1",
                    "gist_duration_scope": "shared_batch_elapsed_with_decode_interleaving",
                    "gist_overlap_decode_batches": job.decode_batches,
                    "extraction_batch_id": batch_id,
                    "extraction_batch_size": len(request.items),
                    "shared_gist_generation_duration_ns": duration_ns,
                    "temporary_owner_rid": batch_id,
                },
            )
            if measurement is not None:
                output.paper_measurement = measurement
                output.extraction_duration_ns = measurement["duration_ns"]
        success = all(output.success for output in outputs)
        measurement = paper_telemetry.finish_request(
            server_request_id=batch_id, success=success, error=error,
            metric_overrides={
                "gist_execution_mode": "tp1-packed-worker-stream-v1",
                "gist_duration_scope": "elapsed_with_decode_interleaving",
                "gist_device_interval_ns": device_interval_ns,
                "gist_generation_duration_ns": duration_ns,
                "gist_overlap_decode_batches": job.decode_batches,
                "gist_worker_steps": result.ticks,
                "gist_cpu_step_ns": result.cpu_step_ns,
                "gist_cpu_telemetry_registration_ns": result.cpu_telemetry_ns,
                "gist_pool_store_cpu_ns": store_cpu_ns, "gist_pool_sync_cpu_ns": sync_cpu_ns,
                "model_calls": result.model_calls, "packed_forward_calls": int(result.model_calls > 0),
                "extraction_batch_size": len(request.items),
            },
        )
        foreground = self.foreground_enabled() and not getattr(
            request, "background_extraction", False
        )
        scheduler.send_to_tokenizer.send_output(C2KVExtractBatchReqOutput(
            rid=request.rid, items=outputs, success=True if foreground else success,
            error="" if foreground else error or "",
            attempted_model_calls=result.model_calls, paper_measurement=measurement,
        ), request)
        if self.foreground_enabled():
            for key, output in zip(job.keys, outputs):
                self._remember_failure(job, output, key=key)
        job.result = None
        job.future = None
        self.job = None

    def _run_job(self, request, ratio, dependency, device):
        """Launch all gist phases on one worker thread and CUDA stream."""
        result = _WorkerResult()
        stream = None
        try:
            torch.cuda.set_device(device)
            if self._worker_stream is None:
                self._worker_stream = torch.cuda.Stream(device=device, priority=0)
            stream = self._worker_stream
            with torch.cuda.stream(stream), torch.no_grad(), paper_telemetry.request_scope(request.rid):
                stream.wait_event(dependency)
                result.start_event = torch.cuda.Event(enable_timing=True)
                result.end_event = torch.cuda.Event(enable_timing=True)
                result.start_event.record(stream)
                input_ids = torch.tensor([request.input_ids], dtype=torch.long, device="cuda")
                result.inputs = (input_ids,)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                result.inputs = (input_ids, attention_mask)
                result.stepper = self.scheduler.tp_worker.model_runner.create_c2kv_extract_stepper(
                    input_ids, attention_mask, ratio,
                    projection_set=request.projection_set or "history",
                )
                while True:
                    started = time.perf_counter_ns()
                    first_step = result.ticks == 0
                    phase_before = getattr(result.stepper, "_phase", None)
                    prior_layers = len(result.stepper.gist_key_values)
                    try:
                        done = result.stepper.step()
                        result.ticks += 1
                        if len(result.stepper.gist_key_values) != prior_layers:
                            # Completed layers remain unchanged. Register only
                            # the new suffix, keeping every per-layer sample.
                            telemetry_started = time.perf_counter_ns()
                            try:
                                paper_telemetry.set_pending_tensors(
                                    request.rid,
                                    tuple(result.stepper.gist_key_values[prior_layers:]),
                                    append=True,
                                )
                                paper_telemetry.sample("forward_with_gist")
                            finally:
                                result.cpu_telemetry_ns += (
                                    time.perf_counter_ns() - telemetry_started
                                )
                    finally:
                        elapsed = time.perf_counter_ns() - started
                        result.cpu_step_ns += elapsed
                        result.cpu_step_max_ns = max(result.cpu_step_max_ns, elapsed)
                        if first_step:
                            result.cpu_prelude_ns += elapsed
                        elif (
                            len(result.stepper.gist_key_values) != prior_layers
                            or phase_before == "layers"
                        ):
                            result.cpu_layers_ns += elapsed
                        else:
                            result.cpu_finalize_ns += elapsed
                    if done:
                        break
                result.end_event.record(stream)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            if stream is not None:
                try:
                    # Keep intermediates in result until submitted kernels end.
                    if result.end_event is None:
                        result.end_event = torch.cuda.Event(enable_timing=True)
                    result.end_event.record(stream)
                except Exception:
                    stream.synchronize()
                    result.end_event = None
        return result

    def _complete(self, job):
        scheduler = self.scheduler
        result = job.result
        duration_ns = time.perf_counter_ns() - job.started_ns
        device_interval_ns = None
        store_cpu_ns = 0
        sync_cpu_ns = 0
        if result.start_event is not None and result.end_event is not None:
            device_interval_ns = int(result.start_event.elapsed_time(result.end_event) * 1e6)
        output = C2KVExtractReqOutput(
            key_hash=job.key, success=False, error=result.error or "",
            gist_generation_duration_ns=duration_ns,
        )
        try:
            if result.error is None:
                key_values, mask, positions = result.stepper.result
                gist_len = mask.shape[1]
                if not scheduler.c2kv_pool.can_allocate(gist_len, existing_key=job.key):
                    raise ValueError("C2KV async result has no remaining unpinned capacity")
                with torch.cuda.stream(scheduler.schedule_stream), paper_telemetry.request_scope(job.request.rid):
                    # Reuse the pool's shape, dtype, position and capacity checks.
                    # No next request observes the new key before this copy ends.
                    store_started = time.perf_counter_ns()
                    try:
                        entry = scheduler.c2kv_pool.store(
                            key_hash=job.key, gist_key_values=key_values, gist_mask=mask,
                            gist_position_ids=positions, original_seq_len=len(job.request.input_ids),
                        )
                    finally:
                        store_cpu_ns = time.perf_counter_ns() - store_started
                    sync_started = time.perf_counter_ns()
                    try:
                        scheduler.schedule_stream.synchronize()
                    finally:
                        sync_cpu_ns = time.perf_counter_ns() - sync_started
                output = C2KVExtractReqOutput(
                    key_hash=job.key, gist_len=entry.gist_len,
                    original_seq_len=entry.original_seq_len, cache_hit=False,
                    gist_generation_duration_ns=duration_ns,
                )
        except Exception as exc:
            output.error = f"{type(exc).__name__}: {exc}"
            output.success = False
        finally:
            # The pending tensors remain registered through the store snapshot.
            paper_telemetry.clear_pending_tensors(job.request.rid)
            result.inputs = None
            result.stepper = None
            key_values = mask = positions = None
        measurement = paper_telemetry.finish_request(
            server_request_id=job.request.rid, success=output.success,
            error=output.error or None,
            metric_overrides={
                "cache_hit": False,
                "gist_generation_duration_ns": duration_ns,
                "gist_execution_mode": "tp1-worker-stream-v1",
                "gist_duration_scope": "elapsed_with_decode_interleaving",
                "gist_device_interval_ns": device_interval_ns,
                "gist_scheduler_ticks": job.scheduler_ticks,
                "gist_worker_steps": result.ticks,
                "gist_overlap_decode_batches": job.decode_batches,
                "gist_cpu_step_ns": result.cpu_step_ns,
                "gist_cpu_step_max_ns": result.cpu_step_max_ns,
                "gist_cpu_prelude_ns": result.cpu_prelude_ns,
                "gist_cpu_layers_ns": result.cpu_layers_ns,
                "gist_cpu_finalize_ns": result.cpu_finalize_ns,
                "gist_cpu_telemetry_ns": result.cpu_telemetry_ns,
                # Host elapsed: store includes pool telemetry; sync is the existing wait.
                "gist_pool_store_cpu_ns": store_cpu_ns,
                "gist_pool_sync_cpu_ns": sync_cpu_ns,
            },
        )
        if measurement is not None:
            output.paper_measurement = measurement
            output.extraction_duration_ns = measurement["duration_ns"]
        if not self.foreground_enabled():
            scheduler.send_to_tokenizer.send_output(output, job.request)
            job.result = None
            job.future = None
            self.job = None
            return
        self._remember_failure(job, output)
        job.result = None
        job.future = None
        self.job = None
        owner = job.reply_owner or job.request
        if job.bulk_hits is not None:
            output.rid = job.request.rid
            output = C2KVBulkCacheLookupReqOutput(
                rid=owner.rid, hits=job.bulk_hits,
                first_miss_index=job.bulk_index, first_miss_result=output,
            )
        self._deliver_foreground(output, owner)
