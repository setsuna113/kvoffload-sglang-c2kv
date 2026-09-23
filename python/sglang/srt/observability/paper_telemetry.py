"""Opt-in, request-scoped telemetry for the C2KV paper experiments.

This module deliberately has no effect unless ``C2KV_PAPER_TELEMETRY=1``.
The paper runner is single-flight by default. Set ``C2KV_PAPER_CONCURRENT=1``
for concurrent serving; memory snapshots then describe shared process occupancy.
Persistent KV occupancy is reported separately from CUDA
allocator/NVML process memory: the former is live K/V payload, while the latter
also includes model weights, workspaces, Q tensors, and allocator reservation.
"""

from __future__ import annotations

import copy
import contextvars
import functools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


_TRUE = {"1", "true", "yes", "on"}
_SYNCHRONOUS_REQUEST_ID = contextvars.ContextVar(
    "c2kv_paper_synchronous_request_id", default=None
)


def enabled() -> bool:
    return os.environ.get("C2KV_PAPER_TELEMETRY", "").strip().lower() in _TRUE


def concurrent_enabled() -> bool:
    return os.environ.get("C2KV_PAPER_CONCURRENT", "").strip().lower() in _TRUE


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class _PaperTelemetry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._main_allocator = None
        self._c2kv_pool = None
        self._tree_cache = None
        self._bytes_per_kv_token = 0
        self._active: Optional[Dict[str, Any]] = None
        self._actives: Dict[str, Dict[str, Any]] = {}
        self._completed: Dict[str, Dict[str, Any]] = {}
        self._nvml = None
        self._nvml_handle = None

    def configure(
        self, main_allocator, c2kv_pool, bytes_per_kv_token: int, tree_cache=None
    ) -> None:
        if not enabled():
            return
        with self._lock:
            self._main_allocator = main_allocator
            self._c2kv_pool = c2kv_pool
            self._tree_cache = tree_cache
            self._bytes_per_kv_token = max(_as_int(bytes_per_kv_token), 0)

    def _tree_cache_sizes(self) -> Dict[str, int]:
        """Prefix-cache slots inside the live pool, split by whether they are
        held by a running request (protected) or merely kept for reuse
        (evictable).  A ChunkCache (radix cache disabled) reports 0 for both."""
        cache = self._tree_cache
        evictable = protected = 0
        if cache is not None:
            try:
                evictable = max(_as_int(cache.evictable_size()), 0)
            except Exception:
                evictable = 0
            try:
                protected = max(_as_int(cache.protected_size()), 0)
            except Exception:
                protected = 0
        return {"evictable": evictable, "protected": protected}

    def _torch_snapshot(self) -> Dict[str, Optional[int]]:
        values: Dict[str, Optional[int]] = {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
        if not torch.cuda.is_available():
            return values
        try:
            values.update(
                allocated_bytes=int(torch.cuda.memory_allocated()),
                reserved_bytes=int(torch.cuda.memory_reserved()),
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            )
        except Exception:
            pass
        return values

    def _init_nvml(self) -> None:
        if self._nvml is False or self._nvml_handle is not None:
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
            if visible and not visible.isdigit():
                handle = pynvml.nvmlDeviceGetHandleByUUID(visible.encode())
            else:
                index = int(visible) if visible.isdigit() else int(torch.cuda.current_device())
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._nvml = pynvml
            self._nvml_handle = handle
        except Exception:
            self._nvml = False
            self._nvml_handle = None

    def _nvml_process_bytes(self) -> Optional[int]:
        self._init_nvml()
        if not self._nvml or self._nvml_handle is None:
            return None
        try:
            processes = self._nvml.nvmlDeviceGetComputeRunningProcesses(
                self._nvml_handle
            )
            pid = os.getpid()
            values = [
                int(proc.usedGpuMemory)
                for proc in processes
                if int(proc.pid) == pid
                and proc.usedGpuMemory not in (None, self._nvml.NVML_VALUE_NOT_AVAILABLE)
            ]
            # WSL can enumerate the process while reporting its memory as
            # NVML_VALUE_NOT_AVAILABLE.  An empty numeric set is unavailable,
            # not a real zero-byte process footprint.
            return sum(values) if values else None
        except Exception:
            return None

    def bind_request(self, req: Any) -> None:
        if not enabled():
            return
        with self._lock:
            active = self._actives.get(str(req.rid))
            if active is not None:
                active["req"] = req

    def _targets(self, req: Any = None) -> list[Dict[str, Any]]:
        if req is not None:
            rid = str(getattr(req, "rid", req))
            active = self._actives.get(rid)
            return [active] if active is not None else []
        if concurrent_enabled():
            synchronous_rid = _SYNCHRONOUS_REQUEST_ID.get()
            if synchronous_rid is not None:
                active = self._actives.get(synchronous_rid)
                return [active] if active is not None else []
            return list(self._actives.values())
        return [self._active] if self._active is not None else []

    def _reference_payload(self, owners: Iterable[Any], *, include_snapshots: bool = True) -> Dict[str, int]:
        states = {}
        fields = ["history_kv_reference_state"]
        if include_snapshots:
            fields += ["reference_decode_baseline_state", "reference_decode_persistent_state"]
        for owner in owners:
            for field in fields:
                state = getattr(owner, field, None)
                if state is not None:
                    states[id(state)] = state
            held = getattr(owner, "racer_held_generation", None)
            if include_snapshots and held is not None and held.reference_state is not None:
                states[id(held.reference_state)] = held.reference_state
        kv_tensors, position_tensors = [], []
        for state in states.values():
            for layer in state.layers.values():
                kv_tensors.extend((layer.key, layer.value))
                position_tensors.append(layer.positions)
        kv_bytes = self._tensor_bytes(kv_tensors)["storage_bytes"]
        position_bytes = self._tensor_bytes(position_tensors)["storage_bytes"]
        bpt = self._bytes_per_kv_token
        return {"bytes": kv_bytes + position_bytes,
                "tokens": (kv_bytes + bpt - 1) // bpt if bpt else 0,
                "kv_bytes": kv_bytes, "position_bytes": position_bytes}

    def _kv_snapshot(self) -> Dict[str, int]:
        allocator = self._main_allocator
        main_capacity = _as_int(getattr(allocator, "size", 0))
        try:
            main_available = _as_int(allocator.available_size()) if allocator else 0
        except Exception:
            main_available = 0
        main_tokens = max(main_capacity - main_available, 0)
        c2kv_pool = self._c2kv_pool
        c2kv_capacity = _as_int(getattr(c2kv_pool, "max_total_tokens", 0))
        try:
            c2kv_tokens = _as_int(c2kv_pool.current_tokens()) if c2kv_pool else 0
        except Exception:
            c2kv_tokens = 0
        # The C2KV pool is an LRU of live KV entries. Its occupied slots are
        # resident, but unpinned entries can be evicted and must appear in the
        # cache line item even when the radix cache is disabled.
        c2kv_cache = getattr(c2kv_pool, "_cache", None)
        c2kv_pins = getattr(c2kv_pool, "_pin_counts", {})
        c2kv_evictable = c2kv_pinned = 0
        if c2kv_cache is not None:
            for key, entry in c2kv_cache.items():
                if c2kv_pins.get(key, 0) > 0:
                    c2kv_pinned += _as_int(entry.gist_len)
                else:
                    c2kv_evictable += _as_int(entry.gist_len)
        c2kv_cache_accounting_available = (
            c2kv_evictable + c2kv_pinned == c2kv_tokens
            if c2kv_cache is not None
            else c2kv_tokens == 0
        )
        resident_tokens = main_tokens + c2kv_tokens
        bpt = self._bytes_per_kv_token
        cache = self._tree_cache_sizes()
        owners = list(getattr(self._tree_cache, "slots", {}).values())
        for active in self._actives.values():
            if active.get("req") is not None:
                owners.append(active["req"])
        reference = self._reference_payload(owners)
        return {
            "bytes_per_kv_token": bpt,
            "main_live_kv_tokens": main_tokens,
            "main_live_kv_bytes": main_tokens * bpt,
            "c2kv_live_kv_tokens": c2kv_tokens,
            "c2kv_live_kv_bytes": c2kv_tokens * bpt,
            "resident_kv_tokens": resident_tokens,
            "resident_kv_bytes": resident_tokens * bpt,
            # Line items of the resident total, never subtracted from it:
            # Evictable radix-cache and C2KV LRU slots are real occupancy that
            # the server keeps for reuse; protected/pinned slots are held by a
            # running request.
            "cached_evictable_kv_tokens": cache["evictable"] + c2kv_evictable,
            "cached_evictable_kv_bytes": (cache["evictable"] + c2kv_evictable) * bpt,
            "cached_protected_kv_tokens": cache["protected"] + c2kv_pinned,
            "cached_protected_kv_bytes": (cache["protected"] + c2kv_pinned) * bpt,
            "c2kv_cached_evictable_kv_tokens": c2kv_evictable,
            "c2kv_cached_evictable_kv_bytes": c2kv_evictable * bpt,
            "c2kv_cached_pinned_kv_tokens": c2kv_pinned,
            "c2kv_cached_pinned_kv_bytes": c2kv_pinned * bpt,
            "c2kv_cache_accounting_available": c2kv_cache_accounting_available,
            # Canonical request peak includes temporary K/V payload that is
            # alive at this exact sample.  With no temporary tensors, it is
            # identical to the pooled live payload.
            "simultaneous_temporary_kv_tokens": 0,
            "simultaneous_temporary_kv_bytes": 0,
            "reference_history_resident_bytes": reference["bytes"],
            "reference_history_kv_bytes": reference["kv_bytes"],
            "reference_history_position_bytes": reference["position_bytes"],
            "reference_history_token_equivalent": reference["tokens"],
            "request_resident_kv_tokens": resident_tokens + reference["tokens"],
            "request_resident_kv_bytes": resident_tokens * bpt + reference["bytes"],
            "main_pool_capacity_tokens": main_capacity,
            "main_pool_capacity_bytes": main_capacity * bpt,
            "c2kv_pool_capacity_tokens": c2kv_capacity,
            "c2kv_pool_capacity_bytes": c2kv_capacity * bpt,
        }

    def _snapshot(self, event: str) -> Dict[str, Any]:
        return {
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "kv": self._kv_snapshot(),
            "torch": self._torch_snapshot(),
            "nvml_process_bytes": self._nvml_process_bytes(),
        }

    @staticmethod
    def _tensor_bytes(tensors: Optional[Iterable[Any]]) -> Dict[str, int]:
        logical = 0
        storage = 0
        seen = set()

        def visit(value: Any) -> None:
            nonlocal logical, storage
            if isinstance(value, (list, tuple)):
                for nested in value:
                    visit(nested)
                return
            if not isinstance(value, torch.Tensor):
                return
            logical += int(value.numel() * value.element_size())
            try:
                store = value.untyped_storage()
                key = (str(value.device), int(store.data_ptr()))
                if key not in seen:
                    seen.add(key)
                    storage += int(store.nbytes())
            except Exception:
                storage += int(value.numel() * value.element_size())

        for value in tensors or ():
            visit(value)
        return {"logical_bytes": logical, "storage_bytes": storage}

    def start(
        self,
        *,
        server_request_id: Any,
        outer_request_id: Optional[str],
        phase: Optional[str],
        kind: str,
        whole_full_kv_tokens: Optional[int] = None,
    ) -> None:
        if not enabled():
            return
        rid = str(server_request_id or "")
        with self._lock:
            if not concurrent_enabled() and self._active is not None and self._active["server_request_id"] != rid:
                self._finish_locked(success=False, error="superseded_by_next_request")
            try:
                if torch.cuda.is_available() and not self._actives:
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            baseline = self._snapshot("baseline")
            if concurrent_enabled():
                for other in self._actives.values():
                    self._update_peak(baseline, other)
            now = baseline["monotonic_ns"]
            phase_name = str(phase or ("extraction" if kind != "generation" else "prefill"))
            active = {
                "server_request_id": rid,
                "outer_request_id": str(outer_request_id or rid),
                "phase": str(phase or phase_name),
                "kind": kind,
                "started_ns": now,
                "whole_full_kv_tokens": (
                    _as_int(whole_full_kv_tokens) if whole_full_kv_tokens is not None else None
                ),
                "baseline": baseline,
                "peak": copy.deepcopy(baseline),
                "pooled_peak": copy.deepcopy(baseline),
                "generation_start": None,
                "final": None,
                "phases": [],
                "current_phase": {"name": phase_name, "start_ns": now, "start_snapshot": baseline},
                "temporary_logical_peak_bytes": 0,
                "temporary_storage_peak_bytes": 0,
                "cached_evictable_peak_tokens": _as_int(
                    baseline["kv"].get("cached_evictable_kv_tokens")
                ),
                "req": None,
                "overlapped": bool(self._actives),
            }
            if concurrent_enabled():
                for other in self._actives.values():
                    other["overlapped"] = True
            self._actives[rid] = active
            if not concurrent_enabled():
                self._active = active

    def _update_peak(self, snapshot: Dict[str, Any], active: Optional[Dict[str, Any]] = None) -> None:
        active = active or self._active
        if active is None:
            return
        active["cached_evictable_peak_tokens"] = max(
            active.get("cached_evictable_peak_tokens", 0),
            _as_int(snapshot["kv"].get("cached_evictable_kv_tokens")),
        )
        peak = active["peak"]
        # Strict comparison keeps the FIRST sample that reaches the peak. With
        # a prefix cache the resident total does not drop when the request
        # finishes (its slots move from protected to evictable), so a later
        # equal sample would misreport the whole peak as evictable cache.
        if (
            snapshot["kv"]["request_resident_kv_bytes"]
            > peak["kv"]["request_resident_kv_bytes"]
        ):
            peak["kv"] = dict(snapshot["kv"])
            peak["event"] = snapshot["event"]
            peak["monotonic_ns"] = snapshot["monotonic_ns"]
        pooled_peak = active["pooled_peak"]
        if (
            snapshot["kv"]["resident_kv_bytes"]
            > pooled_peak["kv"]["resident_kv_bytes"]
        ):
            pooled_peak["kv"] = dict(snapshot["kv"])
            pooled_peak["event"] = snapshot["event"]
            pooled_peak["monotonic_ns"] = snapshot["monotonic_ns"]
        for name in (
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            value = snapshot["torch"].get(name)
            old = peak["torch"].get(name)
            if value is not None and (old is None or value > old):
                peak["torch"][name] = value
        nvml = snapshot.get("nvml_process_bytes")
        old_nvml = peak.get("nvml_process_bytes")
        if nvml is not None and (old_nvml is None or nvml > old_nvml):
            peak["nvml_process_bytes"] = nvml

    def sample(
        self,
        event: str,
        *,
        tensors: Optional[Iterable[Any]] = None,
        temporary_kv: bool = False,
        req: Any = None,
    ) -> Optional[Dict[str, Any]]:
        if not enabled():
            return None
        with self._lock:
            targets = self._targets(req)
            if not targets:
                return None
            peak_targets = list(self._actives.values()) if concurrent_enabled() else targets
            snapshot = self._snapshot(event)
            if tensors is not None:
                tensor_bytes = self._tensor_bytes(tensors)
                snapshot["temporary_kv_tensor_bytes"] = tensor_bytes
                if temporary_kv:
                    temporary_tokens = (
                        tensor_bytes["logical_bytes"] // self._bytes_per_kv_token
                        if self._bytes_per_kv_token
                        else 0
                    )
                    snapshot["kv"]["simultaneous_temporary_kv_tokens"] = (
                        temporary_tokens
                    )
                    snapshot["kv"]["simultaneous_temporary_kv_bytes"] = (
                        tensor_bytes["logical_bytes"]
                    )
                    snapshot["kv"]["request_resident_kv_tokens"] = (
                        snapshot["kv"]["request_resident_kv_tokens"] + temporary_tokens
                    )
                    snapshot["kv"]["request_resident_kv_bytes"] = (
                        snapshot["kv"]["request_resident_kv_bytes"]
                        + tensor_bytes["storage_bytes"]
                    )
                    # Without a request identifier, temporary tensors belong
                    # to the shared process, not to any one request's work.
                    temporary_targets = targets if req is not None or len(targets) == 1 else []
                    for active in temporary_targets:
                        active["temporary_logical_peak_bytes"] = max(
                            active["temporary_logical_peak_bytes"],
                            tensor_bytes["logical_bytes"],
                        )
                        active["temporary_storage_peak_bytes"] = max(
                            active["temporary_storage_peak_bytes"],
                            tensor_bytes["storage_bytes"],
                        )
            for active in peak_targets:
                self._update_peak(snapshot, active)
            return snapshot

    def set_phase(self, name: str, *, req: Any = None, reqs: Optional[Iterable[Any]] = None) -> None:
        if not enabled():
            return
        with self._lock:
            targets = (
                [active for item in reqs for active in self._targets(item)]
                if reqs is not None else self._targets(req)
            )
            # An unbound phase transition cannot identify which overlapping
            # request changed state. Callers with a request pass it explicitly.
            if req is None and reqs is None and len(targets) > 1:
                return
            for active in targets:
                if active["current_phase"]["name"] == name:
                    continue
                end = self._snapshot(f"phase:{active['current_phase']['name']}:end")
                for other in self._actives.values() if concurrent_enabled() else (active,):
                    self._update_peak(end, other)
                current = active.pop("current_phase")
                current.update(
                    end_ns=end["monotonic_ns"],
                    duration_ns=end["monotonic_ns"] - current["start_ns"],
                    end_snapshot=end,
                )
                active["phases"].append(current)
                active["current_phase"] = {
                    "name": name,
                    "start_ns": end["monotonic_ns"],
                    "start_snapshot": end,
                }

    def mark_generation_start(
        self, req: Any, *, normal_kv_tokens: Optional[int] = None
    ) -> None:
        if not enabled():
            return
        with self._lock:
            active = self._actives.get(str(getattr(req, "rid", "")))
            if active is None:
                return
            active["req"] = req
            if active["whole_full_kv_tokens"] is None:
                whole_full = getattr(
                    req, "c2kv_paper_whole_full_kv_tokens", None
                )
                if whole_full is not None:
                    active["whole_full_kv_tokens"] = _as_int(whole_full)
            snapshot = self._snapshot("generation_start")
            # The completed prefill batch excludes a decode slot that overlap
            # scheduling may already have reserved on the mutable request.
            active_tokens = _as_int(
                getattr(req, "kv_committed_len", 0)
                if normal_kv_tokens is None else normal_kv_tokens
            )
            reference = self._reference_payload([req], include_snapshots=False)
            snapshot["request_active_kv_tokens"] = active_tokens + reference["tokens"]
            snapshot["request_active_kv_bytes"] = (
                active_tokens * self._bytes_per_kv_token + reference["bytes"])
            snapshot["reference_history_resident_bytes"] = reference["bytes"]
            active["generation_start"] = snapshot
            for other in self._actives.values() if concurrent_enabled() else (active,):
                self._update_peak(snapshot, other)
            self.set_phase("decode", req=req)

    @staticmethod
    def _req_semantics(req: Any) -> Dict[str, Any]:
        report = getattr(req, "kv_memory_report", None)
        if not isinstance(report, dict):
            report = {}
        physical = report.get("history_kv_physical_eviction")
        if not isinstance(physical, dict):
            physical = {}
        history_kv_backend = report.get("history_kv_backend")
        reference_attention_succeeded = bool(
            physical.get("success") is True
            and history_kv_backend == "reference_attention"
            and report.get("reference_attention_backend")
        )
        runtime_status = report.get("history_kv_runtime_status")
        if reference_attention_succeeded:
            runtime_status = "reference_attention_ok"
        storage_runtime_status = report.get("history_kv_storage_runtime_status")
        if storage_runtime_status is None:
            storage_runtime_status = physical.get("storage_runtime_status")
        if (
            storage_runtime_status is None
            and reference_attention_succeeded
            and physical.get("runtime_status") != "reference_attention_ok"
        ):
            storage_runtime_status = physical.get("runtime_status")
        lifecycle = report.get("history_kv_lifecycle")
        if not isinstance(lifecycle, dict):
            lifecycle = {}
        canonical_full_source = bool(
            getattr(req, "c2kv_paper_canonical_full_source", False)
        )
        request_history_full = getattr(
            req, "c2kv_paper_history_full_kv_tokens", None
        )
        if canonical_full_source and request_history_full is not None:
            # The generic server-tokenized message boundary is the exact
            # denominator for canonical Full/C2KV/H2O/Snap payloads.  Runtime
            # reports may contain zero placeholders or method-local spans.
            history_full = _as_int(request_history_full)
        else:
            report_history_full = report.get("full_equivalent_history_tokens")
            history_full = (
                _as_int(report_history_full)
                if report_history_full is not None
                and _as_int(report_history_full) > 0
                else None
            )

        request_history_active = getattr(
            req, "c2kv_paper_history_active_kv_tokens", None
        )
        report_history_active = _as_int(
            report.get("active_history_kv_tokens")
        )
        if request_history_active is not None:
            # Text arms carry their exact transformed-message span here.
            history_active = _as_int(request_history_active)
        elif report_history_active > 0:
            # KV methods report the post-eviction resident history.
            history_active = report_history_active
        elif canonical_full_source:
            # Plain Full has no runtime eviction report.
            history_active = _as_int(history_full)
        else:
            history_active = 0
        return {
            "history_full_kv_tokens": history_full,
            "history_active_kv_tokens": history_active,
            "full_history_reprefill": bool(
                report.get(
                    "full_history_reprefill_performed",
                    lifecycle.get("full_history_reprefill_performed", False),
                )
            ),
            "history_kv_method": report.get("history_kv_method")
            or report.get("method"),
            "selection_query_tokens_planned": _as_int(
                report.get("selection_query_tokens")
            ),
            "selection_query_tokens_observed": _as_int(
                report.get("selection_query_tokens_observed")
            ),
            "history_kv_backend": history_kv_backend,
            "history_kv_runtime_status": runtime_status,
            "history_kv_storage_runtime_status": storage_runtime_status,
            "history_kv_physical_eviction_success": physical.get("success"),
            "history_kv_lifecycle": lifecycle or None,
            "canonical_full_source": canonical_full_source,
            "denominator_tokenization_duration_ns": getattr(
                req, "c2kv_paper_denominator_tokenization_duration_ns", None
            ),
        }

    def _finish_locked(
        self,
        *,
        req: Any = None,
        server_request_id: Any = None,
        success: bool = True,
        error: Optional[str] = None,
        metric_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if req is not None:
            server_request_id = getattr(req, "rid", server_request_id)
        active = self._actives.get(str(server_request_id)) if server_request_id is not None else self._active
        if active is None and server_request_id is None and len(self._actives) == 1:
            active = next(iter(self._actives.values()))
        if active is None:
            return None
        req = req or active.get("req")
        final = self._snapshot("final")
        for other in self._actives.values() if concurrent_enabled() else (active,):
            self._update_peak(final, other)
        current = active.pop("current_phase")
        current.update(
            end_ns=final["monotonic_ns"],
            duration_ns=final["monotonic_ns"] - current["start_ns"],
            end_snapshot=final,
        )
        active["phases"].append(current)
        active["final"] = final
        generation = active.get("generation_start") or final
        whole_active = _as_int(generation.get("request_active_kv_tokens"))
        semantics = self._req_semantics(req) if req is not None else {}
        history_full = _as_int(semantics.get("history_full_kv_tokens"))
        history_active = _as_int(semantics.get("history_active_kv_tokens"))
        explicit_whole_full = active.get("whole_full_kv_tokens")
        whole_full_source = getattr(req, "c2kv_paper_whole_full_source", None)
        if explicit_whole_full is not None:
            whole_full = _as_int(explicit_whole_full)
        elif whole_full_source == "unknown_missing_client_native_full_renderer":
            # The native tool prompt omits or replaces its raw tool source.
            # History-only reconstruction cannot recover the Full denominator.
            whole_full = None
        elif history_full:
            whole_full = whole_active - history_active + history_full
        else:
            whole_full = None
        bpt = self._bytes_per_kv_token
        baseline_nvml = active["baseline"].get("nvml_process_bytes")
        peak_nvml = active["peak"].get("nvml_process_bytes")
        final_nvml = final.get("nvml_process_bytes")
        peak_kv = active["peak"]["kv"]
        evictable_peak_tokens = _as_int(active.get("cached_evictable_peak_tokens"))
        duration_ns = final["monotonic_ns"] - active["started_ns"]
        metrics = {
            # Gist encoding is a separate synchronous scheduler request.  A
            # normal generation row contributes zero to the additive decision
            # total; c2kv_extract rows overwrite this with their measured
            # cache-miss-only duration in measure_synchronous_request().
            "gist_generation_duration_ns": (
                0 if active["kind"] == "generation" else None
            ),
            "request_peak_resident_kv_tokens": active["peak"]["kv"][
                "request_resident_kv_tokens"
            ],
            "request_peak_resident_kv_bytes": active["peak"]["kv"][
                "request_resident_kv_bytes"
            ],
            "request_peak_pooled_resident_kv_tokens": active["pooled_peak"][
                "kv"
            ]["resident_kv_tokens"],
            "request_peak_pooled_resident_kv_bytes": active["pooled_peak"][
                "kv"
            ]["resident_kv_bytes"],
            # Evictable prefix-cache slots that were part of the resident peak
            # above (same sample), and the request's own baseline/maximum.
            "request_peak_cached_evictable_kv_tokens": _as_int(
                peak_kv.get("cached_evictable_kv_tokens")
            ),
            "request_peak_cached_evictable_kv_bytes": _as_int(
                peak_kv.get("cached_evictable_kv_bytes")
            ),
            "request_peak_c2kv_cached_evictable_kv_tokens": _as_int(
                peak_kv.get("c2kv_cached_evictable_kv_tokens")
            ),
            "request_peak_c2kv_cached_evictable_kv_bytes": _as_int(
                peak_kv.get("c2kv_cached_evictable_kv_bytes")
            ),
            "request_peak_c2kv_cache_accounting_available": bool(
                peak_kv.get("c2kv_cache_accounting_available")
            ),
            "baseline_cached_evictable_kv_tokens": _as_int(
                active["baseline"]["kv"].get("cached_evictable_kv_tokens")
            ),
            "baseline_cached_evictable_kv_bytes": _as_int(
                active["baseline"]["kv"].get("cached_evictable_kv_bytes")
            ),
            "cached_evictable_kv_peak_tokens": evictable_peak_tokens,
            "cached_evictable_kv_peak_bytes": evictable_peak_tokens * bpt,
            "generation_active_kv_tokens": whole_active,
            "generation_active_kv_bytes": generation.get(
                "request_active_kv_bytes", whole_active * bpt),
            "reference_history_resident_bytes": generation.get(
                "reference_history_resident_bytes", 0),
            "whole_full_kv_tokens": whole_full,
            "whole_full_kv_tokens_source": whole_full_source,
            "whole_active_kv_tokens": whole_active,
            "history_full_kv_tokens": semantics.get("history_full_kv_tokens", 0),
            "history_active_kv_tokens": semantics.get("history_active_kv_tokens", 0),
            "temporary_extraction_recovery_peak_kv_tokens": (
                active["temporary_logical_peak_bytes"] // bpt if bpt else 0
            ),
            "temporary_extraction_recovery_peak_kv_bytes": active[
                "temporary_logical_peak_bytes"
            ],
            "temporary_extraction_recovery_peak_storage_bytes": active[
                "temporary_storage_peak_bytes"
            ],
            "torch_peak_allocated_bytes": active["peak"]["torch"].get(
                "peak_allocated_bytes"
            ),
            "torch_peak_reserved_bytes": active["peak"]["torch"].get(
                "peak_reserved_bytes"
            ),
            "nvml_process_start_bytes": baseline_nvml,
            "nvml_process_peak_bytes": peak_nvml,
            "nvml_process_end_bytes": final_nvml,
            "nvml_process_peak_delta_bytes": (
                peak_nvml - baseline_nvml
                if peak_nvml is not None and baseline_nvml is not None
                else None
            ),
            "memory_scope": (
                "process_shared_during_overlap"
                if active["overlapped"] else "request_single_flight"
            ),
            "torch_peak_scope": (
                "process_since_idle_reset"
                if concurrent_enabled() else "process_since_request_start_reset"
            ),
            "duration_scope": "request_wall_clock_elapsed",
            **semantics,
        }
        if active["kind"] == "c2kv_extract":
            metrics["extraction_duration_ns"] = duration_ns
        raw_prefix_cache = getattr(req, "c2kv_raw_prefix_cache", None)
        if isinstance(raw_prefix_cache, dict):
            metrics["c2kv_raw_prefix_cache"] = copy.deepcopy(raw_prefix_cache)
        if metric_overrides:
            metrics.update(metric_overrides)
        result = {
            "schema_version": 1,
            "outer_request_id": active["outer_request_id"],
            "server_request_id": active["server_request_id"],
            "phase": active["phase"],
            "kind": active["kind"],
            "success": bool(success),
            "error": error,
            "metrics": metrics,
            "baseline": active["baseline"],
            "peak": active["peak"],
            "pooled_peak": active["pooled_peak"],
            "generation_start": active.get("generation_start"),
            "final": final,
            "phases": active["phases"],
            "duration_ns": duration_ns,
        }
        self._completed[active["server_request_id"]] = result
        self._completed = dict(list(self._completed.items())[-128:])
        log_path = os.environ.get("C2KV_PAPER_TELEMETRY_LOG")
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
        self._actives.pop(active["server_request_id"], None)
        if self._active is active:
            self._active = None
        return result

    def finish(
        self,
        *,
        req: Any = None,
        server_request_id: Any = None,
        success: bool = True,
        error: Optional[str] = None,
        metric_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not enabled():
            return None
        with self._lock:
            return self._finish_locked(
                req=req,
                server_request_id=server_request_id,
                success=success,
                error=error,
                metric_overrides=metric_overrides,
            )

    def report(self, req: Any, finalize: bool) -> Optional[Dict[str, Any]]:
        if not enabled():
            return None
        rid = str(getattr(req, "rid", ""))
        with self._lock:
            active = self._actives.get(rid)
            if finalize and active is not None:
                reason = getattr(req, "finished_reason", None)
                reason_json = reason.to_json() if hasattr(reason, "to_json") else {}
                is_abort = isinstance(reason_json, dict) and reason_json.get("type") == "abort"
                return self._finish_locked(
                    req=req,
                    success=reason is not None and not is_abort,
                    error=(reason_json.get("message") if is_abort else None),
                )
            if active is None:
                return self._completed.get(rid)
            self.sample("response_snapshot", req=req)
            return {
                "schema_version": 1,
                "outer_request_id": active["outer_request_id"],
                "server_request_id": rid,
                "phase": active["phase"],
                "kind": active["kind"],
                "success": None,
                "metrics": {
                    "request_peak_resident_kv_tokens": active["peak"]["kv"]["request_resident_kv_tokens"],
                    "request_peak_resident_kv_bytes": active["peak"]["kv"]["request_resident_kv_bytes"],
                    "memory_scope": (
                        "process_shared_during_overlap"
                        if active["overlapped"] else "request_single_flight"
                    ),
                },
            }


_STATE = _PaperTelemetry()


configure = _STATE.configure
start_request = _STATE.start
bind_request = _STATE.bind_request
sample = _STATE.sample
set_phase = _STATE.set_phase
mark_generation_start = _STATE.mark_generation_start
finish_request = _STATE.finish
request_report = _STATE.report


def measure_synchronous_request(kind: str, default_phase: str):
    """Measure a scheduler request whose result is returned synchronously."""

    def decorate(func):
        @functools.wraps(func)
        def wrapped(self, recv_req, *args, **kwargs):
            start_request(
                server_request_id=getattr(recv_req, "rid", None),
                outer_request_id=getattr(recv_req, "c2kv_outer_request_id", None),
                phase=getattr(recv_req, "c2kv_measurement_phase", None)
                or default_phase,
                kind=kind,
                whole_full_kv_tokens=len(getattr(recv_req, "input_ids", ()) or ()),
            )
            token = _SYNCHRONOUS_REQUEST_ID.set(str(getattr(recv_req, "rid", "")))
            try:
                try:
                    output = func(self, recv_req, *args, **kwargs)
                except Exception as exc:
                    finish_request(server_request_id=getattr(recv_req, "rid", None), success=False, error=str(exc))
                    raise
                metric_overrides = None
                if kind == "c2kv_extract":
                    metric_overrides = {
                        "gist_generation_duration_ns": getattr(
                            output, "gist_generation_duration_ns", None
                        ),
                        "cache_hit": bool(getattr(output, "cache_hit", False)),
                    }
                result = finish_request(
                    server_request_id=getattr(recv_req, "rid", None),
                    success=bool(getattr(output, "success", True)),
                    error=getattr(output, "error", None) or None,
                    metric_overrides=metric_overrides,
                )
                if result is not None:
                    if kind == "c2kv_extract":
                        extraction_duration_ns = result.get("duration_ns")
                        output.extraction_duration_ns = extraction_duration_ns
                    output.paper_measurement = result
                return output
            finally:
                _SYNCHRONOUS_REQUEST_ID.reset(token)

        return wrapped

    return decorate
