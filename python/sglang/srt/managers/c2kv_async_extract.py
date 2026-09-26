"""One TP1 gist job on a bounded worker beside the scheduler's decode stream.

Only the scheduler thread mutates jobs and the C2KV pool. The worker owns the
stepper and its dedicated CUDA stream; cache publication remains on the scheduler.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
import time

import torch

from sglang.srt.managers.io_struct import (
    C2KVBulkCacheLookupReqInput,
    C2KVExtractBatchReqInput,
    C2KVExtractReqOutput,
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    TokenizedExtractReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.observability import paper_telemetry


@dataclass
class _Job:
    request: TokenizedExtractReqInput
    key: str
    ratio: int
    started_ns: int = field(default_factory=time.perf_counter_ns)
    future: object = None
    result: object = None
    decode_batches: int = 0
    scheduler_ticks: int = 0


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


class AsyncGistExtraction:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._executor = None
        self._worker_stream = None
        self.job = None
        self.deferred = []

    @property
    def busy(self):
        return self.job is not None or bool(self.deferred)

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

    def intercept(self, request):
        """Return True only when ownership of the eventual reply is retained."""
        if self.job is not None and (
            self._contains_key(request, self.job.key)
            or isinstance(request, (
                PauseGenerationReqInput, ContinueGenerationReqInput,
                ReleaseMemoryOccupationReqInput, ResumeMemoryOccupationReqInput, RpcReqInput,
                UpdateWeightFromDiskReqInput, UpdateWeightsFromDistributedReqInput,
                UpdateWeightsFromIPCReqInput, UpdateWeightsFromTensorReqInput,
            ))
            or (isinstance(request, TokenizedExtractReqInput)
                and getattr(request, "background_extraction", False))
        ):
            self.deferred.append(request)
            return True
        if not isinstance(request, TokenizedExtractReqInput):
            return False
        if not getattr(request, "background_extraction", False) or not self.supported():
            return False
        if not request.input_ids or not request.allow_cache_miss:
            return False
        key, ratio, error = self.scheduler._c2kv_extract_cache_key(request)
        if error is not None:
            return False
        pool = self.scheduler.c2kv_pool
        if key in pool._cache:
            return False
        gist_len = (len(request.input_ids) + ratio - 1) // ratio
        if gist_len > min(pool.max_entry_tokens, pool.max_total_tokens):
            return False
        if not pool.can_allocate(gist_len, existing_key=key):
            return False
        self.job = _Job(request, key, ratio)
        paper_telemetry.start_request(
            server_request_id=request.rid,
            outer_request_id=request.c2kv_outer_request_id,
            phase=request.c2kv_measurement_phase or "extraction",
            kind="c2kv_extract",
            whole_full_kv_tokens=len(request.input_ids),
        )
        return True

    def take_ready_requests(self):
        if self.job is not None:
            return []
        requests, self.deferred = self.deferred, []
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
                    self._run_job, job.request, job.ratio, dependency, device
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
            self._complete(job)

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
        scheduler.send_to_tokenizer.send_output(output, job.request)
        job.result = None
        job.future = None
        self.job = None
