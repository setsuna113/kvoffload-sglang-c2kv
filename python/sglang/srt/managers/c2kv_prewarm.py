"""Bounded, content-addressed history prewarming between native requests.

The queue is owned by one tokenizer frontend. It never retains pool pins or
changes foreground selection. A foreground arrival drains at most one already
submitted extraction and prevents further background submissions until all
foreground requests finish. This is idle-time scheduling, not concurrent CUDA
execution or kernel preemption.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field


SCHEMA = "c2kv-native-prewarm-response-v1"
MAX_CHUNKS = 32
MAX_CHUNK_TOKENS = 8192
MAX_JOB_TOKENS = 65536


def validate_prewarm_request(value):
    if not isinstance(value, dict):
        raise ValueError("Prewarm request must be an object")
    operation = value.get("operation")
    if not isinstance(operation, str) or operation not in {"submit", "drain", "cancel"}:
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
                  max_extraction_calls=budget, outer_request_id=outer)
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
    finished_ns: int | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def receipt(self):
        misses = sum(not row["cache_hit"] for row in self.results)
        return {
            "schema": SCHEMA, "owner_id": self.request["owner_id"],
            "job_id": self.request["job_id"], "session_id": self.request["session_id"],
            "status": self.status, "submitted_chunks": len(self.request["chunks"]),
            "completed_chunks": len(self.results), "cache_hits": len(self.results) - misses,
            "model_calls": misses, "budget_known": self.budget_known,
            "cancelled_chunks": (len(self.request["chunks"]) - len(self.results)
                                 if self.done.is_set() else 0),
            "extraction": {"model_calls": misses, "history_model_calls": misses,
                           "tool_model_calls": 0},
            "results": copy.deepcopy(self.results), "error": self.error,
            "submitted_monotonic_ns": self.submitted_ns,
            "finished_monotonic_ns": self.finished_ns,
            "scheduling": "idle-between-native-requests-v1",
        }


class NativePrewarmQueue:
    def __init__(self, extract, *, max_jobs=64):
        self.extract = extract
        self.max_jobs = max_jobs
        self.jobs = {}
        self.finished = OrderedDict()
        self.pending = deque()
        self.foreground_count = 0
        self.inflight = None
        self.worker = None
        self.wake = asyncio.Event()
        self.idle = asyncio.Event()
        self.idle.set()
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
        self.pending.append(key)
        self.wake.set()
        if self.worker is None or self.worker.done():
            self.worker = asyncio.create_task(self._run())
        return job.receipt()

    async def enter_foreground(self):
        self.foreground_count += 1
        try:
            await self.idle.wait()
        except BaseException:
            self.exit_foreground()
            raise

    def exit_foreground(self):
        if self.foreground_count <= 0:
            raise RuntimeError("Unbalanced prewarm foreground scope")
        self.foreground_count -= 1
        self.wake.set()

    async def drain(self, owner_id, job_id, session_id):
        key = (owner_id, job_id)
        job = self.jobs.get(key) or self.finished.get(key)
        if job is None:
            raise ValueError("Unknown prewarm job")
        if session_id != job.request["session_id"]:
            raise ValueError("Prewarm job belongs to a different session")
        job.cancelled = True
        if self.inflight != key and not job.done.is_set():
            self._finish(key, job, "cancelled")
        self.wake.set()
        # The extraction communicator must consume its reply even if this HTTP
        # waiter is cancelled. Only this wait is cancelled, never the worker.
        await job.done.wait()
        if key in self.jobs:
            # Keep unacknowledged costs until their owner reconciles them.
            # Other sessions finishing jobs must not evict this receipt.
            self.jobs.pop(key)
            self.finished[key] = job
            while len(self.finished) > self.max_jobs:
                self.finished.popitem(last=False)
        return job.receipt()

    def _finish(self, key, job, status):
        job.status = status
        job.finished_ns = time.monotonic_ns()
        job.done.set()

    async def _run(self):
        while not self.closed:
            await self.wake.wait()
            self.wake.clear()
            while self.pending and not self.foreground_count and not self.closed:
                key = self.pending.popleft()
                job = self.jobs.get(key)
                if job is None:
                    continue
                if job.cancelled:
                    self._finish(key, job, "cancelled")
                    continue
                index = len(job.results)
                chunk = job.request["chunks"][index]
                self.inflight = key
                self.idle.clear()
                try:
                    result = await self.extract(
                        input_ids=chunk["token_ids"], input_text="",
                        compression_ratio=chunk["compression_ratio"],
                        rid=f"prewarm:{key[0]}:{key[1]}:{index}",
                        allow_cache_miss=True, projection_set="history",
                        outer_request_id=job.request["outer_request_id"],
                        measurement_phase="c2kv_cross_turn_prewarm:extraction",
                    )
                    if not result.success:
                        raise RuntimeError(result.error or "Prewarm extraction failed")
                    if result.original_seq_len != len(chunk["token_ids"]):
                        raise RuntimeError("Prewarm source length mismatch")
                    job.results.append({
                        "handle": chunk["handle"], "cache_key": result.key_hash,
                        "cache_hit": bool(result.cache_hit), "gist_len": result.gist_len,
                        "original_seq_len": result.original_seq_len,
                        "extraction_duration_ns": result.extraction_duration_ns,
                        "gist_generation_duration_ns": result.gist_generation_duration_ns,
                        "paper_measurement": result.paper_measurement,
                    })
                except Exception as error:
                    job.error = f"{type(error).__name__}: {error}"
                    job.budget_known = False
                    self._finish(key, job, "failed")
                finally:
                    self.inflight = None
                    self.idle.set()
                if not job.done.is_set():
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
