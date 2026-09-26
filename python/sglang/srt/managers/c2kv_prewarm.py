"""Bounded, content-addressed history prewarming for native requests.

The queue is owned by one tokenizer frontend. It never retains pool pins or
changes foreground selection. Legacy foregrounds drain one inflight chunk.
Opt-in overlap jobs can run after foreground generation admission. Neither
mode preempts extraction work already inside the communicator.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field


SCHEMA = "c2kv-native-prewarm-response-v1"
MAX_CHUNKS = 32
MAX_CHUNK_TOKENS = 8192
MAX_JOB_TOKENS = 65536
MAX_BATCH_TOKENS = 4096
LEGACY_SCHEDULING = "idle-between-native-requests-v1"
OVERLAP_SCHEDULING = "overlap-native-generation-v1"


def validate_prewarm_request(value):
    if not isinstance(value, dict):
        raise ValueError("Prewarm request must be an object")
    operation = value.get("operation")
    if not isinstance(operation, str) or operation not in {"submit", "poll", "drain", "cancel"}:
        raise ValueError("Unknown prewarm operation")
    for name in ("owner_id", "job_id"):
        if not isinstance(value.get(name), str) or not value[name] or len(value[name]) > 256:
            raise ValueError(f"Prewarm {name} must be a nonempty bounded string")
    result = {key: value[key] for key in ("operation", "owner_id", "job_id")}
    session_id = value.get("session_id")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 1024:
        raise ValueError("Prewarm session_id must be a nonempty bounded string")
    result["session_id"] = session_id
    if operation != "submit":
        return result
    scheduling = value.get("scheduling", LEGACY_SCHEDULING)
    if scheduling not in (LEGACY_SCHEDULING, OVERLAP_SCHEDULING):
        raise ValueError("Unknown prewarm scheduling mode")
    after_native_rid = value.get("after_native_rid")
    if "after_native_rid" in value:
        if scheduling != OVERLAP_SCHEDULING:
            raise ValueError("after_native_rid requires overlap scheduling")
        if (not isinstance(after_native_rid, str) or not after_native_rid
                or len(after_native_rid) > 4096):
            raise ValueError("after_native_rid must be a nonempty bounded string")
    budget = value.get("max_extraction_calls")
    chunks = value.get("chunks")
    if type(budget) is not int or budget < 0:
        raise ValueError("Prewarm extraction budget must be a nonnegative integer")
    if not isinstance(chunks, list) or not 0 < len(chunks) <= min(MAX_CHUNKS, budget):
        raise ValueError("Prewarm chunks must fit the bounded remaining extraction budget")
    normalized, handles, total_tokens = [], set(), 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Prewarm chunk must be an object")
        handle, tokens, ratio = (chunk.get(k) for k in ("handle", "token_ids", "compression_ratio"))
        if not isinstance(handle, str) or not handle or len(handle) > 256 or handle in handles:
            raise ValueError("Prewarm chunk handles must be unique nonempty bounded strings")
        if (not isinstance(tokens, list) or not 0 < len(tokens) <= MAX_CHUNK_TOKENS
                or any(type(token) is not int or not 0 <= token < 2**63 for token in tokens)):
            raise ValueError("Prewarm token_ids must be a bounded nonempty integer list")
        if type(ratio) is not int or ratio <= 0:
            raise ValueError("Prewarm compression_ratio must be a positive integer")
        if chunk.get("projection_set", "history") not in (None, "history"):
            raise ValueError("Cross-turn prewarm currently accepts history chunks only")
        total_tokens += len(tokens)
        handles.add(handle)
        normalized.append({"handle": handle, "token_ids": list(tokens), "compression_ratio": ratio})
    if total_tokens > MAX_JOB_TOKENS:
        raise ValueError("Prewarm job exceeds the queued token limit")
    outer = value.get("outer_request_id")
    if outer is not None and (not isinstance(outer, str) or not outer or len(outer) > 4096):
        raise ValueError("Prewarm outer_request_id must be a bounded string")
    result.update(session_id=session_id, chunks=normalized,
                  max_extraction_calls=budget, outer_request_id=outer,
                  scheduling=scheduling)
    if after_native_rid is not None:
        result["after_native_rid"] = after_native_rid
    return result


@dataclass
class _Job:
    request: dict
    signature: str
    submitted_ns: int = field(default_factory=time.monotonic_ns)
    results: list = field(default_factory=list)
    status: str = "queued"
    error: str | None = None
    budget_known: bool = True
    cancelled: bool = False
    admission_ready: bool = False
    finished_ns: int | None = None
    started_ns: int | None = None
    last_completed_ns: int | None = None
    extraction_wall_ns: int = 0
    unpublished_model_calls: int = 0
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def receipt(self):
        misses = sum(not row["cache_hit"] for row in self.results)
        model_calls = misses + self.unpublished_model_calls
        return {
            "schema": SCHEMA, "owner_id": self.request["owner_id"],
            "job_id": self.request["job_id"], "session_id": self.request["session_id"],
            "status": self.status, "submitted_chunks": len(self.request["chunks"]),
            "completed_chunks": len(self.results), "cache_hits": len(self.results) - misses,
            "pending_chunks": (0 if self.done.is_set()
                               else len(self.request["chunks"]) - len(self.results)),
            "model_calls": model_calls, "budget_known": self.budget_known,
            "cancelled_chunks": (len(self.request["chunks"]) - len(self.results)
                                 if self.done.is_set() else 0),
            "extraction": {"model_calls": model_calls, "history_model_calls": model_calls,
                           "tool_model_calls": 0},
            "results": copy.deepcopy(self.results), "error": self.error,
            "submitted_monotonic_ns": self.submitted_ns,
            "started_monotonic_ns": self.started_ns,
            "last_completed_monotonic_ns": self.last_completed_ns,
            "finished_monotonic_ns": self.finished_ns,
            "extraction_wall_duration_ns": self.extraction_wall_ns,
            "scheduling": self.request["scheduling"],
        }


class NativePrewarmQueue:
    def __init__(self, extract, *, extract_batch=None, max_jobs=64):
        self.extract = extract
        self.extract_batch = extract_batch
        self.max_jobs = max_jobs
        try:
            self.batch_size = int(os.environ.get("C2KV_PREWARM_BATCH_SIZE", "1"))
        except ValueError as exc:
            raise ValueError("C2KV_PREWARM_BATCH_SIZE must be an integer from 1 to 4") from exc
        if not 1 <= self.batch_size <= 4:
            raise ValueError("C2KV_PREWARM_BATCH_SIZE must be an integer from 1 to 4")
        self.overlap_idle_only = os.environ.get(
            "C2KV_PREWARM_IDLE_ONLY", ""
        ).lower() in {"1", "true", "yes", "on"}
        self.jobs = {}
        self.finished = OrderedDict()
        self.recent_admissions = OrderedDict()
        self.pending = deque()
        self.foreground_count = 0
        self.generation_count = 0
        self.generation_waiting = set()
        self.inflight = None
        self.inflight_scheduling = None
        self.worker = None
        self.wake = asyncio.Event()
        self.idle = asyncio.Event()
        self.idle.set()
        self.legacy_idle = asyncio.Event()
        self.legacy_idle.set()
        self.closed = False

    def submit(self, payload):
        request = validate_prewarm_request(payload)
        if request["operation"] != "submit":
            raise ValueError("submit requires the submit operation")
        if self.closed:
            raise ValueError("Prewarm queue is closed")
        key = (request["owner_id"], request["job_id"])
        signature = json.dumps(request, sort_keys=True, separators=(",", ":"))
        prior = self.jobs.get(key) or self.finished.get(key)
        if prior is not None:
            if prior.signature != signature:
                raise ValueError("Prewarm job identity reused with different content")
            return prior.receipt()
        if any(owner == key[0] for owner, _ in self.jobs):
            raise ValueError("Prewarm owner already has an outstanding job")
        if len(self.jobs) >= self.max_jobs:
            raise ValueError("Prewarm queue is full")
        job = self.jobs[key] = _Job(request, signature)
        after_native_rid = request.get("after_native_rid")
        job.admission_ready = (after_native_rid is None or
                               after_native_rid in self.recent_admissions)
        self.pending.append(key)
        self.wake.set()
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._run())
        return job.receipt()

    async def enter_foreground(self, *, overlap=False):
        self.foreground_count += 1
        try:
            await (self.legacy_idle if overlap else self.idle).wait()
        except BaseException:
            self.exit_foreground()
            raise

    def exit_foreground(self):
        if self.foreground_count <= 0:
            raise RuntimeError("Unbalanced prewarm foreground scope")
        self.foreground_count -= 1
        self.wake.set()

    def enter_generation_wait(self, native_rid):
        if not isinstance(native_rid, str) or not 0 < len(native_rid) <= 4096:
            raise ValueError("Generation wait requires a bounded native_rid")
        if native_rid in self.generation_waiting:
            raise RuntimeError("Duplicate prewarm generation wait")
        if self.generation_count + len(self.generation_waiting) >= self.foreground_count:
            raise RuntimeError("Generation wait requires a foreground scope")
        self.generation_waiting.add(native_rid)
        self.wake.set()

    def exit_generation_wait(self, native_rid):
        if isinstance(native_rid, str):
            self.generation_waiting.discard(native_rid)
        self.wake.set()

    def enter_generation(self, native_rid=None):
        transitioning = (isinstance(native_rid, str) and
                         native_rid in self.generation_waiting)
        if (self.generation_count + len(self.generation_waiting) - transitioning
                >= self.foreground_count):
            raise RuntimeError("Generation admission requires a foreground scope")
        if transitioning:
            self.generation_waiting.remove(native_rid)
        self.generation_count += 1
        if isinstance(native_rid, str) and 0 < len(native_rid) <= 4096:
            self.recent_admissions[native_rid] = None
            self.recent_admissions.move_to_end(native_rid)
            while len(self.recent_admissions) > self.max_jobs:
                self.recent_admissions.popitem(last=False)
            for job in self.jobs.values():
                if job.request.get("after_native_rid") == native_rid:
                    job.admission_ready = True
        self.wake.set()

    def exit_generation(self):
        if self.generation_count <= 0:
            raise RuntimeError("Unbalanced prewarm generation scope")
        self.generation_count -= 1

    def _get_job(self, owner_id, job_id, session_id):
        key = (owner_id, job_id)
        job = self.jobs.get(key) or self.finished.get(key)
        if job is None:
            raise ValueError("Unknown prewarm job")
        if session_id != job.request["session_id"]:
            raise ValueError("Prewarm job belongs to a different session")
        return key, job

    def _acknowledge(self, key, job):
        if key in self.jobs and job.done.is_set():
            self.jobs.pop(key)
            self.finished[key] = job
            while len(self.finished) > self.max_jobs:
                self.finished.popitem(last=False)

    def poll(self, owner_id, job_id, session_id):
        key, job = self._get_job(owner_id, job_id, session_id)
        self._acknowledge(key, job)
        return job.receipt()

    async def drain(self, owner_id, job_id, session_id):
        key, job = self._get_job(owner_id, job_id, session_id)
        job.cancelled = True
        if self.inflight != key and not job.done.is_set():
            self._finish(key, job, "cancelled")
        self.wake.set()
        # The extraction communicator must consume its reply even if this HTTP
        # waiter is cancelled. Only this wait is cancelled, never the worker.
        await job.done.wait()
        # Keep unacknowledged costs until their owner reconciles them.
        self._acknowledge(key, job)
        return job.receipt()

    def _finish(self, key, job, status):
        if job.done.is_set():
            return
        job.status = status
        job.finished_ns = time.monotonic_ns()
        job.done.set()

    def _extract_kwargs(self, key, job, index):
        chunk = job.request["chunks"][index]
        kwargs = {
            "input_ids": chunk["token_ids"], "input_text": "",
            "compression_ratio": chunk["compression_ratio"],
            "rid": f"prewarm:{key[0]}:{key[1]}:{index}",
            "allow_cache_miss": True, "projection_set": "history",
            "outer_request_id": job.request["outer_request_id"],
            "measurement_phase": "c2kv_cross_turn_prewarm:extraction",
        }
        if job.request["scheduling"] == OVERLAP_SCHEDULING:
            kwargs["background_extraction"] = True
        return kwargs

    def _batch_indices(self, job, index):
        if (self.extract_batch is None or self.batch_size == 1
                or job.request["scheduling"] != OVERLAP_SCHEDULING):
            return [index]
        chunks = job.request["chunks"]
        ratio = chunks[index]["compression_ratio"]
        indices = []
        raw_tokens = 0
        for next_index in range(index, min(len(chunks), index + self.batch_size)):
            chunk = chunks[next_index]
            if (chunk["compression_ratio"] != ratio
                    or raw_tokens + len(chunk["token_ids"]) > MAX_BATCH_TOKENS):
                break
            indices.append(next_index)
            raw_tokens += len(chunk["token_ids"])
        return indices or [index]

    def _record_result(self, job, chunk, result, started_ns, finished_ns):
        if not result.success:
            raise RuntimeError(result.error or "Prewarm extraction failed")
        if result.original_seq_len != len(chunk["token_ids"]):
            raise RuntimeError("Prewarm source length mismatch")
        row = {
            "handle": chunk["handle"], "cache_key": result.key_hash,
            "cache_hit": bool(result.cache_hit), "gist_len": result.gist_len,
            "original_seq_len": result.original_seq_len,
            "extraction_duration_ns": result.extraction_duration_ns,
            "gist_generation_duration_ns": result.gist_generation_duration_ns,
            "paper_measurement": result.paper_measurement,
            "extraction_started_monotonic_ns": started_ns,
            "extraction_finished_monotonic_ns": finished_ns,
        }
        for field_name in (
            "extraction_batch_id", "extraction_batch_size",
            "shared_gist_generation_duration_ns",
        ):
            value = getattr(result, field_name, None)
            if value is not None:
                row[field_name] = value
        job.results.append(row)
        job.last_completed_ns = finished_ns

    async def _run(self):
        while not self.closed:
            await self.wake.wait()
            self.wake.clear()
            while self.pending and not self.closed:
                key = None
                for _ in range(len(self.pending)):
                    candidate = self.pending.popleft()
                    candidate_job = self.jobs.get(candidate)
                    if candidate_job is None or candidate_job.done.is_set():
                        continue
                    scheduling = candidate_job.request["scheduling"]
                    allowed = (self.foreground_count == 0 or
                               (scheduling == OVERLAP_SCHEDULING and
                                not self.overlap_idle_only and
                                self.foreground_count ==
                                self.generation_count + len(self.generation_waiting)))
                    allowed = allowed and candidate_job.admission_ready
                    if key is None and allowed:
                        key = candidate
                    else:
                        self.pending.append(candidate)
                if key is None:
                    break
                job = self.jobs.get(key)
                if job is None:
                    continue
                if job.cancelled:
                    self._finish(key, job, "cancelled")
                    continue
                index = len(job.results)
                indices = self._batch_indices(job, index)
                self.inflight = key
                self.inflight_scheduling = job.request["scheduling"]
                self.idle.clear()
                if self.inflight_scheduling == LEGACY_SCHEDULING:
                    self.legacy_idle.clear()
                if job.started_ns is None:
                    job.started_ns = time.monotonic_ns()
                job.status = "running"
                chunk_started_ns = time.monotonic_ns()
                failed = False
                batch_reply = None
                attempted_model_calls = None
                published_misses_before = sum(not row["cache_hit"] for row in job.results)
                try:
                    if len(indices) > 1:
                        batch_reply = await self.extract_batch(
                            [self._extract_kwargs(key, job, item_index)
                             for item_index in indices]
                        )
                        attempts = getattr(batch_reply, "attempted_model_calls", None)
                        if attempts is not None:
                            if type(attempts) is not int or not 0 <= attempts <= len(indices):
                                raise RuntimeError("Invalid prewarm batch model-call count")
                            attempted_model_calls = attempts
                        outputs = batch_reply.items
                        if batch_reply.retry_individually:
                            if attempted_model_calls != 0 or outputs:
                                raise RuntimeError("Invalid prewarm batch retry response")
                            result = await self.extract(**self._extract_kwargs(key, job, index))
                            self._record_result(
                                job, job.request["chunks"][index], result,
                                chunk_started_ns, time.monotonic_ns(),
                            )
                        else:
                            if len(outputs) > len(indices):
                                raise RuntimeError("Prewarm batch reply count mismatch")
                            if batch_reply.success and len(outputs) != len(indices):
                                raise RuntimeError("Prewarm batch reply count mismatch")
                            batch_finished_ns = time.monotonic_ns()
                            for item_index, output in zip(indices, outputs):
                                expected_rid = self._extract_kwargs(key, job, item_index)["rid"]
                                if getattr(output, "rid", None) != expected_rid:
                                    raise RuntimeError("Prewarm batch reply order mismatch")
                                self._record_result(
                                    job, job.request["chunks"][item_index], output,
                                    chunk_started_ns, batch_finished_ns,
                                )
                            published_misses = (
                                sum(not row["cache_hit"] for row in job.results)
                                - published_misses_before
                            )
                            if (batch_reply.success and attempted_model_calls is not None
                                    and attempted_model_calls != published_misses):
                                raise RuntimeError("Invalid prewarm batch model-call count")
                            if not batch_reply.success:
                                raise RuntimeError(batch_reply.error or "Prewarm extraction failed")
                    else:
                        result = await self.extract(**self._extract_kwargs(key, job, index))
                        self._record_result(
                            job, job.request["chunks"][index], result,
                            chunk_started_ns, time.monotonic_ns(),
                        )
                except Exception as error:
                    job.error = f"{type(error).__name__}: {error}"
                    job.budget_known = False
                    if attempted_model_calls is not None and batch_reply is not None:
                        published_misses = (
                            sum(not row["cache_hit"] for row in job.results)
                            - published_misses_before
                        )
                        if attempted_model_calls < published_misses:
                            job.error = "RuntimeError: Invalid prewarm batch model-call count"
                        else:
                            job.unpublished_model_calls += (
                                attempted_model_calls - published_misses
                            )
                    failed = True
                finally:
                    job.extraction_wall_ns += time.monotonic_ns() - chunk_started_ns
                    self.inflight = None
                    if self.inflight_scheduling == LEGACY_SCHEDULING:
                        self.legacy_idle.set()
                    self.inflight_scheduling = None
                    self.idle.set()
                if failed:
                    self._finish(key, job, "failed")
                elif not job.done.is_set():
                    if len(job.results) == len(job.request["chunks"]):
                        self._finish(key, job, "completed")
                    elif job.cancelled:
                        self._finish(key, job, "cancelled")
                    else:
                        self.pending.append(key)
                # Let waiting foreground arrivals take priority at every chunk.
                await asyncio.sleep(0)

    async def close(self):
        self.closed = True
        for key, job in list(self.jobs.items()):
            if not job.done.is_set():
                job.cancelled = True
                if self.inflight != key:
                    self._finish(key, job, "cancelled")
        self.wake.set()
        if self.worker is not None:
            await asyncio.shield(self.worker)
