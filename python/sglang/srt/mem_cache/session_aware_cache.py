from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import json
import logging

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class _VirtualNode:
    """Sentinel node for streaming session requests.

    Passed to inc_lock_ref / dec_lock_ref so the wrapper can distinguish
    streaming-session locks (no-op) from real radix-tree locks (forwarded).
    """

    pass


@dataclass
class SessionSlot:
    """Holds KV state between streaming session turns."""

    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)

    # KV pool state (None means no KV is currently held by this slot)
    req_pool_idx: Optional[int] = None
    kv_committed_len: int = 0
    kv_allocated_len: int = 0

    # First req's radix tree node (for dec_lock_ref on session close)
    last_node: Any = None
    cache_protected_len: int = 0
    swa_uuid_for_lock: Optional[str] = None

    # SWA state
    swa_evicted_seqlen: int = 0

    # C2KV physical-history eviction keeps a compact physical sequence while
    # rotary positions remain in the original logical frame. Persist the
    # correction across streaming session requests.
    c2kv_position_correction: int = 0
    history_kv_resident_positions: Any = None
    history_kv_score_state: Any = None
    history_kv_reference_state: Any = None
    history_kv_reference_config: Any = None
    history_kv_runtime_state: Any = None
    racer_held_generation: Any = None
    racer_ephemeral_spans: Any = None
    c2kv_tool_source_spans: Any = None
    c2kv_tool_kv_accounting: Any = None
    c2kv_tool_view: Any = None
    c2kv_tool_source_digest: Optional[str] = None

    # Mamba states
    mamba_pool_idx: Any = None
    mamba_ping_pong_track_buffer: Any = None
    mamba_next_track_idx: Any = None
    mamba_last_track_seqlen: Any = None
    mamba_branching_seqlen: Any = None

    @property
    def is_holding_kv(self) -> bool:
        """Whether this slot currently holds KV pool resources."""
        return self.req_pool_idx is not None

    def save_from_req(self, req: Req, is_first: bool):
        """Save KV state from a finishing request into this slot."""
        self.req_pool_idx = req.req_pool_idx
        self.kv_committed_len = req.kv_committed_len
        self.kv_allocated_len = req.kv_allocated_len
        self.swa_evicted_seqlen = req.swa_evicted_seqlen
        self.c2kv_position_correction = int(
            getattr(req, "c2kv_position_correction", 0) or 0
        )
        self.history_kv_resident_positions = list(getattr(req, "history_kv_resident_positions", []) or [])
        resident = set(self.history_kv_resident_positions)
        self.history_kv_score_state = {layer: {p: s for p, s in scores.items() if p in resident}
                                     for layer, scores in (getattr(req, "history_kv_score_state", {}) or {}).items()}
        self.history_kv_reference_state = getattr(
            req, "history_kv_reference_state", None
        )
        self.history_kv_reference_config = getattr(
            req, "history_kv_reference_config", None
        )
        self.history_kv_runtime_state = getattr(
            req, "history_kv_runtime_state", None
        )
        self.racer_held_generation = getattr(req, "racer_held_generation", None)
        req.racer_held_generation = None
        self.c2kv_tool_source_spans = list(getattr(req, "c2kv_tool_source_spans", []) or [])
        hint = getattr(req, "c2kv_kv_memory_hint", None) or {}
        ephemeral = hint.get("racer_ephemeral_source_span")
        if ephemeral is not None and tuple(ephemeral) not in (self.racer_ephemeral_spans or []):
            self.racer_ephemeral_spans = list(self.racer_ephemeral_spans or []) + [tuple(ephemeral)]
        tool_segments = hint.get("tool_memory_segments") or []
        if tool_segments:
            self.c2kv_tool_view = [dict(item) for item in tool_segments]
            self.c2kv_tool_source_digest = (hint.get("joint_tool_memory") or {}).get(
                "source_protocol_token_sha256"
            )
        report = getattr(req, "kv_memory_report", None) or {}
        self.c2kv_tool_kv_accounting = {key: int(report.get(key) or 0) for key in (
            "active_tool_kv_tokens", "active_tool_gist_tokens", "active_tool_repair_tokens", "tool_encoder_source_tokens")}

        if is_first:
            self.last_node = req.last_node
            self.cache_protected_len = req.cache_protected_len
            self.swa_uuid_for_lock = req.swa_uuid_for_lock

        self.mamba_pool_idx = req.mamba_pool_idx
        self.mamba_ping_pong_track_buffer = req.mamba_ping_pong_track_buffer
        self.mamba_next_track_idx = req.mamba_next_track_idx
        self.mamba_last_track_seqlen = req.mamba_last_track_seqlen
        self.mamba_branching_seqlen = req.mamba_branching_seqlen

        req.req_pool_idx = None
        req.mamba_pool_idx = None
        req.history_kv_reference_state = None
        req.history_kv_runtime_state = None
        req.reference_decode_persistent_state = None
        req.reference_decode_baseline_runtime_state = None

    def restore_to_req(self, req: Req):
        """Restore KV state from this slot into an incoming request."""
        req.req_pool_idx = self.req_pool_idx
        req.kv_committed_len = self.kv_committed_len
        req.kv_allocated_len = self.kv_allocated_len
        req.swa_evicted_seqlen = self.swa_evicted_seqlen
        req.c2kv_position_correction = self.c2kv_position_correction
        req.history_kv_resident_positions = list(
            self.history_kv_resident_positions or []
        )
        req.history_kv_score_state = self.history_kv_score_state
        req.history_kv_reference_state = self.history_kv_reference_state
        incoming_reference_config = getattr(
            req, "history_kv_reference_config", None
        )
        if incoming_reference_config is None:
            req.history_kv_reference_config = self.history_kv_reference_config
        elif self.history_kv_reference_config is not None and str(
            incoming_reference_config.get("method") or ""
        ).lower() != str(
            self.history_kv_reference_config.get("method") or ""
        ).lower():
            raise RuntimeError("PERSISTENT_HISTORY_REFERENCE_METHOD_CHANGED")
        req.history_kv_runtime_state = self.history_kv_runtime_state
        req.c2kv_tool_source_spans = list(self.c2kv_tool_source_spans or [])
        if self.c2kv_tool_source_spans and isinstance(getattr(req, "kv_memory_report", None), dict):
            req.kv_memory_report.update(self.c2kv_tool_kv_accounting or {})
        req.swa_uuid_for_lock = self.swa_uuid_for_lock

        req.mamba_pool_idx = self.mamba_pool_idx
        req.mamba_ping_pong_track_buffer = self.mamba_ping_pong_track_buffer
        req.mamba_next_track_idx = self.mamba_next_track_idx
        req.mamba_last_track_seqlen = self.mamba_last_track_seqlen
        req.mamba_branching_seqlen = self.mamba_branching_seqlen

        # NOTE: req_pool_idx and mamba_pool_idx are intentionally NOT cleared
        # from the slot. During chunked prefill, a request may be rejected by
        # the scheduler (e.g. budget exhausted) and retried in the next cycle.
        # Each retry calls match_prefix -> restore_to_req again, so the slot
        # must remain intact for idempotent restoration.


def _is_streaming(req: Optional[Req]) -> bool:
    return req is not None and req.session is not None and req.session.streaming


class SessionAwareCache(BasePrefixCache):
    """Decorator around any BasePrefixCache that manages streaming session KV.

    Non-streaming requests are pure pass-through. Streaming requests have their
    KV lifecycle managed by SessionSlot objects, avoiding any invasive changes
    to the scheduling pipeline.
    """

    def __init__(self, inner: BasePrefixCache):
        self.inner = inner
        self.slots: Dict[str, SessionSlot] = {}
        self.c2kv_pool = None
        self.c2kv_tool_rope_cache = None

    # -- Forward PrefixCacheTrait properties to inner cache --

    @staticmethod
    def owns_finished_request(req: Req) -> bool:
        """Streaming KV belongs to the session, even without radix insertion."""
        return _is_streaming(req) or bool(
            getattr(req, "session_cache_closed_during_request", False)
        )

    @property
    def req_to_token_pool(self):
        return self.inner.req_to_token_pool

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):
        self.inner.req_to_token_pool = value

    @property
    def token_to_kv_pool_allocator(self):
        return self.inner.token_to_kv_pool_allocator

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):
        self.inner.token_to_kv_pool_allocator = value

    @property
    def page_size(self):
        return self.inner.page_size

    @page_size.setter
    def page_size(self, value):
        self.inner.page_size = value

    @property
    def disable(self):
        return self.inner.disable

    @disable.setter
    def disable(self, value):
        self.inner.disable = value

    @property
    def metrics_collector(self):
        return self.inner.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value):
        self.inner.metrics_collector = value

    # -- BasePrefixCache abstract methods --

    def reset(self):
        self.slots.clear()
        self.inner.reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        req = params.req
        if getattr(req, "session_cache_closed_during_request", False):
            # A retry after explicit close still owns the restored KV row. Do
            # not match its possibly compacted physical indices as a radix key.
            prefix_len = min(
                req.kv_committed_len, max(len(params.key.token_ids) - 1, 0)
            )
            device_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :prefix_len
            ].to(dtype=torch.int64)
            return MatchResult(
                device_indices=device_indices,
                last_device_node=req.last_node,
                last_host_node=req.last_node,
                cache_protected_len=req.cache_protected_len,
            )
        if not _is_streaming(req):
            return self.inner.match_prefix(params)

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        if slot is None or slot.req_pool_idx is None:
            config = getattr(req, "history_kv_eviction", None)
            hint = getattr(req, "c2kv_kv_memory_hint", None) or {}
            if (isinstance(config, dict) and config.get("persistent_continuation")) or (
                self._is_persistent_history_req(req) and int(hint.get("persistent_session_logical_prefix_tokens", 0)) > 0
            ):
                raise RuntimeError("PERSISTENT_HISTORY_SESSION_RESIDENT_CACHE_MISSING")
            return self.inner.match_prefix(params)

        # A persistent arm can follow a legacy streaming turn that did not
        # carry the persistent hint (the first turn has no history to evict).
        # That turn may have left decode KV in the session slot. Adopt only
        # the canonical prompt prefix before restoring the slot, otherwise the
        # old decode suffix becomes allocator-owned but unaccounted after the
        # next persistent request overwrites the slot metadata.
        config = getattr(req, "history_kv_eviction", None)
        hint = getattr(req, "c2kv_kv_memory_hint", None) or {}
        if (
            isinstance(config, dict)
            and config.get("persistent_continuation_pending")
            and not slot.history_kv_resident_positions
        ):
            logical_prefix = int(
                hint.get("persistent_session_logical_prefix_tokens") or 0
            )
            if logical_prefix > 0:
                self._adopt_legacy_persistent_prefix(slot, logical_prefix)

        drop_generation_prefix = int(
            hint.get("persistent_session_drop_generation_prefix_tokens") or 0
        )
        if getattr(slot, "racer_held_generation", None) is not None or (hint.get("persistent_history_session") or {}).get("transaction"):
            self._resolve_racer_transaction(slot, req)
        if drop_generation_prefix:
            self._trim_persistent_generation_prefix(slot, req)
        if hint.get("racer_replacement_source_spans") and not getattr(req, "racer_source_replacement_applied", False):
            self._replace_racer_sources(slot, req)

        if isinstance(config, dict) and config.get("persistent_continuation_pending"):
            # A timed-out generation can finish on the server after its caller
            # retries the same prefix. The slot then belongs to a later
            # canonical horizon than the queued retry. Reject it before
            # transferring ownership or appending duplicate logical positions.
            expected_horizon = int(
                hint.get(
                    "persistent_session_computed_prefix_tokens",
                    hint["persistent_session_logical_prefix_tokens"],
                )
            )
            slot_horizon = int(slot.kv_committed_len) + int(
                slot.c2kv_position_correction
            )
            if expected_horizon != slot_horizon:
                raise RuntimeError("PERSISTENT_HISTORY_SESSION_STALE_CONTINUATION")

        slot.restore_to_req(req)
        self._refresh_persistent_tool_prefix(slot, req)

        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict):
            report["persistent_history_session"] = True
            report["persistent_session_restore"] = True
            report["persistent_session_reused_physical_tokens"] = int(
                req.kv_committed_len
            )
            report["persistent_session_position_correction"] = int(
                req.c2kv_position_correction
            )

        # A persistent physical-history request arrives with only the exact
        # chat-template delta.  Now that the session KV length is restored,
        # split that delta into completed-history and current-query rounds.
        # This must happen after restore: before that point the physical prefix
        # length is unknown.
        config = getattr(req, "history_kv_eviction", None)
        if (
            isinstance(config, dict)
            and config.get("persistent_continuation_pending")
        ):
            from sglang.srt.managers.schedule_batch import C2KVPrefillRound

            from sglang.srt.mem_cache.history_kv_lifecycle import (
                append_resident_positions,
                physical_history_range,
                position_summary,
                selection_query_window,
            )
            hint = req.c2kv_kv_memory_hint or {}
            logical_prefix = int(hint["persistent_session_logical_prefix_tokens"])
            computed_prefix = int(
                hint.get("persistent_session_computed_prefix_tokens", logical_prefix)
            )
            canonical_len = int(hint["persistent_session_canonical_prompt_tokens"])
            prior_positions = list(slot.history_kv_resident_positions or [])
            if len(prior_positions) != int(req.kv_committed_len):
                raise RuntimeError("PERSISTENT_HISTORY_SESSION_LEDGER_LENGTH_MISMATCH")
            descriptors = getattr(req, "c2kv_composition_pending", None)
            if descriptors:
                from sglang.srt.mem_cache.c2kv_composition import resident_positions, split_rounds_at_query, raw_query_window
                positions = resident_positions(len(req.origin_input_ids), descriptors, prior_positions, computed_prefix)
                cursor = len(prior_positions)
                source_cursor = computed_prefix
                for descriptor in descriptors:
                    source_cursor += descriptor["token_start"] - cursor
                    if descriptor.get("region") == "tool":
                        req.c2kv_tool_source_spans.append((source_cursor, source_cursor + descriptor["source_tokens"]))
                    source_cursor += descriptor["source_tokens"]
                    cursor = descriptor["token_end"]
                if positions and positions[-1] >= canonical_len:
                    raise RuntimeError("PERSISTENT_HISTORY_TOOL_CANONICAL_LENGTH_MISMATCH")
            else:
                positions = append_resident_positions(
                    prior_positions, computed_prefix, canonical_len
                )
            req.history_kv_resident_positions = positions
            protected, history_end = physical_history_range(positions,
                int(config.get("persistent_protected_prefix_tokens") or 0),
                int(config["persistent_canonical_history_end"]))
            delta_history = int(config.get("persistent_delta_history_tokens") or 0)
            prefix_len = int(req.kv_committed_len)
            origin_len = len(req.c2kv_virtual_input_ids) if descriptors else len(req.origin_input_ids)
            if not (0 <= protected <= history_end <= origin_len and len(positions) == origin_len):
                req.set_finish_with_abort(
                    "PERSISTENT_HISTORY_SESSION_RANGE_INVALID: "
                    f"{protected=}, {prefix_len=}, {delta_history=}, {origin_len=}"
                )
            else:
                method = str(config.get("method") or "").strip().lower()
                query_window = selection_query_window(
                    method,
                    prefix_len,
                    history_end,
                    origin_len,
                    config.get("history_kv_recent_window"),
                )
                if descriptors:
                    if query_window is None:
                        query_window = (max(prefix_len, history_end - 1), max(prefix_len + 1, history_end))
                    query_start, query_end = raw_query_window(req.c2kv_rounds, descriptors, query_window)
                    rounds = split_rounds_at_query(req.c2kv_rounds, descriptors, query_start, query_end, C2KVPrefillRound)
                    config.update(selection_query_start=query_start, selection_query_end=query_end,
                                  selection_query_tokens=query_end - query_start,
                                  selection_query_phase="new_tail_prefill_before_eviction")
                    req.c2kv_composition_pending = None
                elif query_window is None:
                    rounds = [
                        C2KVPrefillRound(
                            list(
                                req.origin_input_ids[
                                    : max(prefix_len + 1, history_end)
                                ]
                            ),
                            [],
                            post_history_kv_eviction=True,
                        )
                    ]
                    round_end = min(
                        origin_len, max(prefix_len + 1, history_end)
                    )
                    if round_end < origin_len:
                        rounds.append(
                            C2KVPrefillRound(
                                list(req.origin_input_ids[round_end:]), []
                            )
                        )
                else:
                    query_start, query_end = query_window
                    rounds = []
                    if query_start > prefix_len:
                        rounds.append(
                            C2KVPrefillRound(
                                list(req.origin_input_ids[:query_start]), []
                            )
                        )
                        query_tokens = list(
                            req.origin_input_ids[query_start:query_end]
                        )
                    else:
                        query_tokens = list(req.origin_input_ids[:query_end])
                    rounds.append(
                        C2KVPrefillRound(
                            query_tokens,
                            [],
                            post_history_kv_eviction=True,
                        )
                    )
                    config.update(
                        {
                            "selection_query_start": query_start,
                            "selection_query_end": query_end,
                            "selection_query_tokens": query_end - query_start,
                            "selection_query_phase": (
                                "new_tail_prefill_before_eviction"
                            ),
                        }
                    )
                req.c2kv_rounds = rounds
                req.c2kv_round_idx = 0
                req.c2kv_round_start_len = 0
                if not descriptors:
                    req.c2kv_virtual_input_ids = list(req.origin_input_ids)
                config["history_start"] = protected
                config["history_end"] = history_end
                config["persistent_prior_physical_tokens"] = prefix_len
                config["resident_logical_positions"] = positions
                if req.c2kv_tool_source_spans:
                    from sglang.srt.mem_cache.c2kv_composition import protected_history_indices
                    config["protected_history_indices"] = protected_history_indices(positions, protected, history_end, req.c2kv_tool_source_spans)
                config["previous_resident_position_summary"] = position_summary(prior_positions)
                if isinstance(report, dict):
                    report["persistent_session_history_start"] = protected
                    report["persistent_session_history_end"] = history_end
                    report["persistent_session_delta_history_tokens"] = delta_history
                    for key in (
                        "selection_query_start",
                        "selection_query_end",
                        "selection_query_tokens",
                        "selection_query_phase",
                    ):
                        if key in config:
                            report[key] = config[key]
                config.pop("persistent_continuation_pending", None)

        # The caller built params.key before a tool refresh could resize the
        # session's physical prefix. Its token count is stale after refresh;
        # the refreshed active request IDs and committed KV row are the pair
        # that the next prefill round must use together.
        key_tokens = (
            req.origin_input_ids
            if self._is_persistent_history_req(req)
            else params.key.token_ids
        )
        prefix_len = min(req.kv_committed_len, max(len(key_tokens) - 1, 0))
        if self._is_persistent_history_req(req) and prefix_len != req.kv_committed_len:
            raise RuntimeError("PERSISTENT_HISTORY_SESSION_PREFIX_EXCEEDS_ACTIVE_INPUT")
        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)

        return MatchResult(
            device_indices=device_indices,
            last_device_node=slot.virtual_node,
            last_host_node=slot.virtual_node,
            cache_protected_len=slot.cache_protected_len,
        )

    def _resolve_racer_transaction(self, slot: SessionSlot, req: Req) -> None:
        """Promote a held decode, or restore its untouched resident prompt."""
        from sglang.srt.mem_cache.racer_transaction import transaction_config

        transaction = transaction_config(getattr(req, "c2kv_kv_memory_hint", None) or {})
        held = getattr(slot, "racer_held_generation", None)
        report = getattr(req, "kv_memory_report", None)
        if getattr(req, "racer_previous_resolution_applied", False):
            return
        if held is None:
            if transaction and transaction.get("resolution"):
                raise RuntimeError("RACER_TRANSACTION_CHECKPOINT_MISSING")
            return
        if transaction is None or not transaction.get("resolution"):
            raise RuntimeError("RACER_TRANSACTION_PREVIOUS_RESOLUTION_REQUIRED")
        resolution = transaction["resolution"]
        if transaction["phase"] == "regenerate" and (
            resolution != "discard" or held.decision_id != transaction["decision_id"]
        ):
            raise RuntimeError("RACER_TRANSACTION_REGENERATION_MISMATCH")
        if resolution == "discard":
            row = self.req_to_token_pool.req_to_token[slot.req_pool_idx]
            old_len = int(slot.kv_allocated_len)
            if old_len < held.prompt_len:
                raise RuntimeError("RACER_TRANSACTION_PROTECTED_PROMPT_LOST")
            keep = row[:held.prompt_len].long()
            tail = row[held.prompt_len:old_len].long()
            keep_pages = torch.unique(keep[keep > 0] // self.page_size)
            tail_pages = torch.unique(tail[tail > 0] // self.page_size)
            free_pages = tail_pages[~torch.isin(tail_pages, keep_pages)]
            if free_pages.numel():
                self.token_to_kv_pool_allocator.free(free_pages * self.page_size)
            row[held.prompt_len:old_len] = 0
            slot.kv_committed_len = slot.kv_allocated_len = held.prompt_len
            slot.c2kv_position_correction = held.position_correction
            slot.history_kv_resident_positions = list(held.resident_positions)
            slot.history_kv_reference_state = held.reference_state
            slot.history_kv_runtime_state = held.runtime_state
            slot.history_kv_score_state = held.score_state
            if isinstance(report, dict):
                report["racer_previous_resolution"] = {
                    "decision_id": held.decision_id, "resolution": resolution,
                    "restored_prompt_tokens": held.prompt_len,
                    "freed_normal_page_tokens": int(free_pages.numel()) * self.page_size,
                    "restored_algorithm_statistics": True,
                    "discarded_draft_executed": False,
                    "full_history_reprefill_performed": False,
                }
        elif isinstance(report, dict):
            report["racer_previous_resolution"] = {
                "decision_id": held.decision_id, "resolution": resolution,
                "promoted_algorithm_statistics": True,
                "full_history_reprefill_performed": False,
            }
        recovery = ((getattr(req, "c2kv_kv_memory_hint", None) or {}).get("persistent_history_session") or {}).get("recovery_append") or {}
        if transaction["decision_id"] != held.decision_id or recovery.get("replace_previous_evidence"):
            self._expire_racer_evidence(slot, req)
        slot.racer_held_generation = None
        req.racer_previous_resolution_applied = True

    def _expire_racer_evidence(self, slot: SessionSlot, req: Req) -> None:
        """Forget recovery evidence without rebasing any retained RoPE position."""
        spans = list(getattr(slot, "racer_ephemeral_spans", None) or [])
        if not spans:
            return
        from sglang.srt.mem_cache.history_kv_eviction import PhysicalHistoryKVEvictor

        state = getattr(slot, "history_kv_reference_state", None)
        if state is not None:
            for layer in state.layers.values():
                if any(bool(((layer.positions >= start) & (layer.positions < end)).any()) for start, end in spans):
                    raise RuntimeError("RACER_EVIDENCE_MUST_REMAIN_NATIVE_TAIL")
        positions = list(slot.history_kv_resident_positions)
        keep = [i for i, p in enumerate(positions) if not any(start <= p < end for start, end in spans)]
        old_len = len(positions)
        if len(keep) != old_len:
            result = PhysicalHistoryKVEvictor(self.req_to_token_pool, self.token_to_kv_pool_allocator).evict(
                slot, method="racer_evidence_expiry", history_start=0,
                history_end=old_len, target_tokens=len(keep), selected_history_indices=keep,
            )
            if not result.success:
                raise RuntimeError("RACER_EVIDENCE_EXPIRY_FAILED: " + result.error)
            for field in ("origin_input_ids", "origin_input_ids_unpadded", "c2kv_virtual_input_ids"):
                ids = getattr(req, field, None)
                if ids is not None:
                    setattr(req, field, [ids[i] for i in keep] + list(ids[old_len:]))
            slot.history_kv_resident_positions = [positions[i] for i in keep]
            resident = set(slot.history_kv_resident_positions)
            slot.history_kv_score_state = {
                layer: {p: score for p, score in scores.items() if p in resident}
                for layer, scores in (slot.history_kv_score_state or {}).items()
            }
        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict):
            report["racer_evidence_expiry"] = {
                "source_spans": spans, "expired_native_tokens": old_len - len(keep),
                "retained_original_tokens": len(keep), "canonical_ledger_preserved": True,
                "full_history_reprefill_performed": False,
                "trigger": "recovery_replacement" if (((getattr(req, "c2kv_kv_memory_hint", None) or {}).get("persistent_history_session") or {}).get("recovery_append") or {}).get("replace_previous_evidence") else "next_decision",
            }
        slot.racer_ephemeral_spans = []

    def _replace_racer_sources(self, slot: SessionSlot, req: Req) -> None:
        from sglang.srt.mem_cache.history_kv_eviction import PhysicalHistoryKVEvictor
        from sglang.srt.mem_cache.racer_transaction import replace_reference_sources, interrupt_replaced_commit_window

        spans = req.c2kv_kv_memory_hint["racer_replacement_source_spans"]
        initial = ((req.c2kv_kv_memory_hint.get("persistent_history_session") or {})
                   .get("initial_s0_append") or {})
        protected = (req.c2kv_kv_memory_hint.get("racer_protected_pending_source_positions") or []) if initial.get("enabled") else []
        reference, reference_receipt = replace_reference_sources(
            slot.history_kv_reference_state, spans, protected_positions=protected)
        positions = list(slot.history_kv_resident_positions)
        keep = [index for index, p in enumerate(positions) if not any(start <= p < end for start, end in spans)]
        old_len = len(positions)
        if len(keep) < old_len:
            result = PhysicalHistoryKVEvictor(self.req_to_token_pool, self.token_to_kv_pool_allocator).evict(
                slot, method="racer_source_replacement", history_start=0, history_end=old_len,
                target_tokens=len(keep), selected_history_indices=keep,
            )
            if not result.success:
                raise RuntimeError("RACER_SOURCE_REPLACEMENT_FAILED: " + result.error)
            for field in ("origin_input_ids", "origin_input_ids_unpadded", "c2kv_virtual_input_ids"):
                ids = getattr(req, field, None)
                if ids is not None:
                    setattr(req, field, [ids[i] for i in keep] + list(ids[old_len:]))
        slot.history_kv_resident_positions = [positions[i] for i in keep]
        resident = set(slot.history_kv_resident_positions)
        slot.history_kv_score_state = {layer: {p: score for p, score in values.items() if p in resident} for layer, values in (slot.history_kv_score_state or {}).items()}
        slot.history_kv_reference_state = reference
        removed = [p for index, p in enumerate(positions) if index not in set(keep)]
        for layer in reference_receipt["layers"]:
            for name in ("source_positions", "collateral_positions"):
                removed.extend(p for head in layer[name] for p in head)
        lifecycle = interrupt_replaced_commit_window(getattr(slot, "history_kv_runtime_state", None), removed)
        req.racer_source_replacement_applied = True
        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict):
            report["racer_source_replacement"] = {
                "source_spans": spans, "removed_normal_positions": [p for index, p in enumerate(positions) if index not in set(keep)],
                "reference": reference_receipt, "reference_deletion_policy": "union_columns",
                "interrupted_lifecycle_measurement": lifecycle,
                "full_history_reprefill_performed": False,
            }

    def _refresh_persistent_tool_prefix(self, slot: SessionSlot, req: Req) -> None:
        hint = getattr(req, "c2kv_kv_memory_hint", None) or {}
        refresh = hint.get("persistent_tool_refresh")
        if not refresh:
            return
        old = refresh.get("previous_segments") or [refresh["previous_segment"]]
        new = refresh.get("new_segments") or [refresh["new_segment"]]
        if len(old) != len(new):
            raise RuntimeError("PERSISTENT_HISTORY_TOOL_PREFIX_CHANGED")
        for index, (previous, current) in enumerate(zip(old, new)):
            if previous != current:
                self._refresh_persistent_tool_segment(slot, req, {
                    "source_protocol_token_sha256": refresh["source_protocol_token_sha256"],
                    "previous_segment": previous, "new_segment": current,
                }, index)

    def _refresh_persistent_tool_segment(self, slot: SessionSlot, req: Req, refresh: dict, segment_index: int) -> None:
        """Replace the tool view while preserving existing reference history.

        Paged allocators own whole pages: rebuild the normal row on fresh
        pages, then release the old row, including its partially used pages.
        """
        old_view = refresh["previous_segment"]
        new_view = refresh["new_segment"]
        if slot.c2kv_tool_view[segment_index] == new_view:
            return  # match_prefix may retry this same request.
        if (
            slot.c2kv_tool_view[segment_index] != old_view
            or slot.c2kv_tool_source_digest
            != refresh["source_protocol_token_sha256"]
            or segment_index >= len(slot.c2kv_tool_source_spans or [])
        ):
            raise RuntimeError("PERSISTENT_HISTORY_TOOL_PREFIX_CHANGED")
        keys = ([new_view["key_hash"]] if new_view.get("key_hash") else []) + list(new_view.get("repair_key_hashes") or [])
        if not keys or self.c2kv_pool is None:
            raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_ENTRY_UNSUPPORTED")
        key = keys[0]
        if not self.c2kv_pool.pin_many(keys):
            raise RuntimeError("C2KV_CACHE_MISS: persistent tool refresh entry")
        try:
            entries = [self.c2kv_pool.get(key) for key in keys]
            if any(entry is None for entry in entries):
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_ENTRY_UNSUPPORTED")
            source_start, source_end = slot.c2kv_tool_source_spans[segment_index]
            if (
                int(new_view["source_tokens"]) != source_end - source_start
            ):
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_PREFIX_CHANGED")
            entry_positions = [self.c2kv_pool.get_position_ids(entry) + (source_start if entry.entry_type == "gist" else 0) for entry in entries]
            new_positions = torch.cat(entry_positions).tolist()
            order = sorted(range(len(new_positions)), key=new_positions.__getitem__)
            new_positions = [int(new_positions[index]) for index in order]
            if (
                not new_positions
                or
                new_positions != sorted(set(new_positions))
                or any(not source_start <= position < source_end for position in new_positions)
            ):
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_POSITIONS_INVALID")
            old_positions = list(slot.history_kv_resident_positions or [])
            indices = [
                index for index, position in enumerate(old_positions)
                if source_start <= position < source_end
            ]
            if (
                not indices
                or indices != list(range(indices[0], indices[-1] + 1))
            ):
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_LAYOUT_INVALID")
            old_len = int(slot.kv_committed_len)
            old_width = len(indices)
            new_width = len(new_positions)
            old_gist = self.c2kv_pool.get(old_view["key_hash"]) if old_view.get("key_hash") else None
            if old_view.get("key_hash") and old_gist is None and old_view.get("repair_key_hashes"):
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_OLD_ACCOUNTING_UNAVAILABLE")
            old_gist_tokens = int(old_gist.gist_len) if old_gist is not None else (old_width if old_view.get("key_hash") else 0)
            new_gist_tokens = sum(int(entry.gist_len) for entry in entries if entry.entry_type == "gist")
            delta = new_width - old_width
            page_size = int(getattr(self.token_to_kv_pool_allocator, "page_size", 1))
            rebuild_row = page_size > 1
            if (
                int(slot.kv_allocated_len) != old_len
                or len(old_positions) != old_len
            ):
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_SLOT_LAYOUT_UNSUPPORTED")
            shared_prefix = int(slot.cache_protected_len)
            if shared_prefix and (rebuild_row or shared_prefix > indices[0]):
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_SHARED_PREFIX_UNSUPPORTED")
            row = self.req_to_token_pool.req_to_token[slot.req_pool_idx]
            if old_len + delta > row.shape[0]:
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_CONTEXT_OVERFLOW")
            old_loc = row[indices[0] : indices[-1] + 1].long().clone()
            old_row = row[:old_len].long().clone()
            if bool((old_row <= 0).any()):
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_SLOT_LAYOUT_UNSUPPORTED")
            from sglang.srt.mem_cache.c2kv_pool import c2kv_gist_token_ids

            replacement_ids = c2kv_gist_token_ids(key, new_width)
            active_id_fields = ("origin_input_ids", "origin_input_ids_unpadded", "c2kv_virtual_input_ids")
            updated_ids = {}
            for field in active_id_fields:
                ids = getattr(req, field, None)
                if ids is None:
                    continue
                ids = list(ids)
                if len(ids) < old_len:
                    raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_ACTIVE_IDS_INVALID")
                updated_ids[field] = (
                    ids[:indices[0]] + replacement_ids + ids[indices[-1] + 1:]
                )
            new_len = old_len + delta
            allocation_size = ((new_len + page_size - 1) // page_size * page_size
                               if rebuild_row else new_width)
            allocated = self.token_to_kv_pool_allocator.alloc(allocation_size)
            if allocated is None:
                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_KV_OOM")
            new_row = (allocated[:new_len] if rebuild_row else torch.cat(
                (old_row[:indices[0]], allocated, old_row[indices[-1] + 1:])))
            new_loc = new_row[indices[0]:indices[0] + new_width]
            kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
            try:
                for layer_idx in range(self.c2kv_pool.num_layers):
                    layer_id = self.c2kv_pool.start_layer + layer_idx
                    key_buffer, value_buffer = kv_cache.get_kv_buffer(layer_id)
                    dest_key = key_buffer.reshape(-1, *key_buffer.shape[-2:])
                    dest_value = value_buffer.reshape(-1, *value_buffer.shape[-2:])
                    all_keys, all_values = [], []
                    for entry, positions in zip(entries, entry_positions):
                        stored_key, stored_value = self.c2kv_pool.get_layer_kv(entry, layer_idx)
                        if not entry.already_rotated:
                            from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

                            rope = self.c2kv_tool_rope_cache
                            if rope is None or bool(((positions < 0) | (positions >= rope.shape[0])).any()):
                                raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_ROPE_UNAVAILABLE")
                            half = rope.shape[-1] // 2
                            stored_key = apply_rotary_emb(stored_key, rope[positions, :half], rope[positions, half:], True)
                        all_keys.append(stored_key)
                        all_values.append(stored_value)
                    new_key = torch.cat(all_keys)[order]
                    new_value = torch.cat(all_values)[order]
                    if (
                        new_key.shape != dest_key[new_loc].shape
                        or new_value.shape != dest_value[new_loc].shape
                    ):
                        raise RuntimeError("PERSISTENT_HISTORY_TOOL_REFRESH_SHAPE_CHANGED")
                    if rebuild_row:
                        dest_key[new_row] = torch.cat((
                            dest_key[old_row[:indices[0]]], new_key,
                            dest_key[old_row[indices[-1] + 1:]]))
                        dest_value[new_row] = torch.cat((
                            dest_value[old_row[:indices[0]]], new_value,
                            dest_value[old_row[indices[-1] + 1:]]))
                    else:
                        dest_key[new_loc] = new_key
                        dest_value[new_loc] = new_value
            except Exception:
                self.token_to_kv_pool_allocator.free(allocated)
                raise
            row[:new_len] = new_row
            if delta < 0:
                row[old_len + delta:old_len] = 0
            self.token_to_kv_pool_allocator.free(old_row if rebuild_row else old_loc)
            old_positions[indices[0] : indices[-1] + 1] = new_positions
            slot.history_kv_resident_positions = old_positions
            req.history_kv_resident_positions = list(old_positions)
            slot.kv_committed_len += delta
            slot.kv_allocated_len += delta
            slot.c2kv_position_correction -= delta
            req.kv_committed_len = slot.kv_committed_len
            req.kv_allocated_len = slot.kv_allocated_len
            req.c2kv_position_correction = slot.c2kv_position_correction
            if slot.cache_protected_len >= indices[-1] + 1:
                slot.cache_protected_len += delta
            for field, ids in updated_ids.items():
                setattr(req, field, ids)
            resident = set(old_positions)
            slot.history_kv_score_state = {
                layer: {position: score for position, score in scores.items() if position in resident}
                for layer, scores in (slot.history_kv_score_state or {}).items()
            }
            req.history_kv_score_state = slot.history_kv_score_state
            slot.c2kv_tool_view[segment_index] = dict(new_view)
            accounting = slot.c2kv_tool_kv_accounting or {}
            for name, change in (
                ("active_tool_kv_tokens", delta),
                ("active_tool_gist_tokens", new_gist_tokens - old_gist_tokens),
                ("active_tool_repair_tokens", delta - new_gist_tokens + old_gist_tokens),
            ):
                if name in accounting:
                    accounting[name] += change
            pinned = getattr(req, "c2kv_pinned_keys", None)
            if pinned is None:
                req.c2kv_pinned_keys = pinned = []
            pinned.extend(keys)
            report = getattr(req, "kv_memory_report", None)
            if isinstance(report, dict):
                report["persistent_tool_kv_refreshed"] = True
                report["persistent_tool_kv_refresh_tokens"] = len(new_positions)
                report["persistent_tool_kv_refresh_old_tokens"] = old_width
                report["persistent_tool_kv_refresh_delta_tokens"] = delta
                report["persistent_tool_kv_refresh_rebuilt_normal_row"] = rebuild_row
                report["persistent_tool_kv_refresh_allocated_tokens"] = allocation_size
                report["persistent_tool_kv_refresh_page_size"] = page_size
                report.update(slot.c2kv_tool_kv_accounting or {})
        except Exception:
            self.c2kv_pool.unpin_many(keys)
            raise

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        if getattr(req, "session_cache_closed_during_request", False):
            # The orphaned request owns the row transferred by release_session.
            # Its physical KV can no longer be inserted under canonical tokens.
            return self.inner.cache_finished_req(req, is_insert=False, **kwargs)
        if not _is_streaming(req):
            return self.inner.cache_finished_req(req, is_insert=is_insert, **kwargs)

        if self._is_persistent_history_req(req):
            if getattr(req, "persistent_history_eviction_failed", False):
                self._rollback_failed_persistent_request(req)
                return
            from sglang.srt.mem_cache.history_kv_lifecycle import append_resident_positions, position_summary
            positions = getattr(req, "history_kv_resident_positions", None)
            hint = req.c2kv_kv_memory_hint or {}
            if positions is None:
                slot = self.slots.get(req.session.session_id)
                previous = list(slot.history_kv_resident_positions or []) if slot else []
                positions = append_resident_positions(previous,
                    int(hint.get("persistent_session_logical_prefix_tokens", 0)),
                    int(hint.get("persistent_session_canonical_prompt_tokens", len(req.origin_input_ids))))
                req.history_kv_resident_positions = positions
            reference_config = getattr(req, "history_kv_reference_config", None)
            exact_generated_prefix = bool(
                isinstance(reference_config, dict)
                and str(reference_config.get("method") or "").lower()
                in {"agentkv", "commitkv"}
            )
            if (
                not exact_generated_prefix
                and len(positions) != len(req.origin_input_ids)
            ):
                raise RuntimeError("PERSISTENT_HISTORY_SESSION_FINISHED_LEDGER_MISMATCH")
            self._discard_persistent_decode_suffix(req)
            positions = list(req.history_kv_resident_positions or [])
            old_slot = self.slots.get(req.session.session_id)
            evidence_spans = list(getattr(old_slot, "racer_ephemeral_spans", None) or [])
            if hint.get("racer_ephemeral_source_span") is not None:
                evidence_spans.append(tuple(hint["racer_ephemeral_source_span"]))
            if isinstance(getattr(req, "kv_memory_report", None), dict) and evidence_spans:
                native_evidence = sum(any(start <= p < end for start, end in evidence_spans) for p in positions)
                req.kv_memory_report["racer_native_evidence_tokens"] = native_evidence
                req.kv_memory_report["racer_history_and_evidence_tokens"] = int(req.kv_memory_report.get("active_history_kv_tokens", 0)) + native_evidence
            # Ordinary persistent methods discard every decode KV token before
            # saving the session. The validated resident ledger is therefore
            # all prompt, even if overlap advanced the mutable decode length.
            protected_prompt_len = (
                int(getattr(req, "reference_decode_protected_len", len(positions)) or 0)
                if exact_generated_prefix
                else len(positions)
            )
            if not 0 <= protected_prompt_len <= len(positions):
                raise RuntimeError(
                    "PERSISTENT_HISTORY_SESSION_PROTECTED_PROMPT_LENGTH_MISMATCH"
                )
            if isinstance(getattr(req, "kv_memory_report", None), dict):
                req.kv_memory_report["persistent_session_saved_position_summary"] = position_summary(positions)
                old_slot = self.slots.get(req.session.session_id)
                prior = list(old_slot.history_kv_resident_positions or []) if old_slot else []
                event = req.kv_memory_report.setdefault("history_kv_lifecycle", {})
                reference_method = str(
                    (reference_config or {}).get("method") or ""
                ).lower()
                if exact_generated_prefix:
                    # The canonical source interval includes omitted tool
                    # tokens. Count the actual physical prompt before history
                    # eviction, not the source-space boundary difference.
                    appended_before_eviction = event.get(
                        "resident_tokens_after_append"
                    )
                    if appended_before_eviction is None:
                        appended_before_eviction = protected_prompt_len
                    appended_physical_tokens = int(appended_before_eviction) - len(prior)
                else:
                    appended_physical_tokens = int(
                        hint.get("persistent_session_delta_tokens", 0)
                    )
                if appended_physical_tokens < 0:
                    raise RuntimeError(
                        "PERSISTENT_HISTORY_SESSION_APPEND_COUNT_INVALID"
                    )
                resident_after_append = len(prior) + appended_physical_tokens
                prompt_positions = positions[:protected_prompt_len]
                event.update({
                    **{k: hint.get(k) for k in ("episode_id", "turn_id", "step_id")},
                    "event": "session_prompt_saved", "session_id": req.session.session_id,
                    "history_kv_method": str(hint.get("history_kv_method") or reference_method or (getattr(req, "history_kv_eviction", None) or {}).get("method") or ""),
                    "history_kv_backend": (
                        "reference_attention"
                        if reference_method in {"pyramid", "pyramidkv", "agentkv", "commitkv"}
                        else "physical_eviction"
                    ), "persistent_session_enabled": True,
                    "resident_tokens_before_append": len(prior),
                    "new_turn_tokens": int(hint.get("persistent_session_delta_tokens", 0)),
                    "physical_tokens_appended": appended_physical_tokens,
                    "resident_tokens_after_append": resident_after_append,
                    "resident_tokens_after_eviction": protected_prompt_len,
                    "evicted_tokens_this_turn": resident_after_append - protected_prompt_len,
                    "resident_stream_tokens_after_decode": len(positions),
                    "resident_decode_normal_tokens": len(positions) - protected_prompt_len,
                    "previous_resident_position_summary": position_summary(prior),
                    "resident_position_summary": position_summary(prompt_positions),
                    "resident_stream_position_summary": position_summary(positions),
                    "full_history_reprefill_performed": False,
                    "history_prefill_tokens": int((getattr(req, "history_kv_eviction", None) or {}).get("persistent_delta_history_tokens", 0)),
                    "canonical_delta_prefill_tokens": int(hint.get("persistent_session_delta_tokens", 0)),
                    "count_scope": "canonical_prompt_including_protected_system_and_current",
                })
                event.setdefault("full_history_tokens", int(hint.get("full_equivalent_history_tokens", 0)))
                event.setdefault("retained_tokens_this_turn", int(req.kv_memory_report.get("active_history_kv_tokens", 0)))
                logging.getLogger(__name__).info("HISTORY_KV_LIFECYCLE %s", json.dumps(event, sort_keys=True))

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        is_first = slot is None
        if is_first:
            slot = SessionSlot()
            self.slots[session_id] = slot

        if (hint.get("persistent_history_session") or {}).get("transaction") and isinstance(getattr(req, "kv_memory_report", None), dict):
            from sglang.srt.mem_cache.racer_transaction import (
                protected_pending_positions, regeneration_mandatory_history,
            )

            runtime = getattr(req, "history_kv_runtime_state", None)
            pending_positions = protected_pending_positions(runtime)
            req.kv_memory_report["racer_current_protected_pending_positions"] = pending_positions
            req.kv_memory_report["racer_current_protected_pending_tokens"] = len(pending_positions)
            req.kv_memory_report["racer_current_mandatory_history"] = regeneration_mandatory_history(runtime)
        slot.save_from_req(req, is_first=is_first)

    @staticmethod
    def _is_persistent_history_req(req: Req) -> bool:
        hint = getattr(req, "c2kv_kv_memory_hint", None)
        return bool(
            isinstance(hint, dict)
            and isinstance(hint.get("persistent_history_session"), dict)
            and hint["persistent_history_session"].get("enabled")
        )

    def _discard_persistent_decode_suffix(self, req: Req) -> None:
        """Keep canonical prompt KV; free decode pages by PHYSICAL ownership.

        Allocator page IDs are not logical prompt offsets. In particular a
        session may start at any page in the pool; comparing a physical page
        ID with ceil(prompt_len/page_size) can free the prompt itself.
        """
        reference_config = getattr(req, "history_kv_reference_config", None)
        reference_method = (
            str(reference_config.get("method") or "").lower()
            if isinstance(reference_config, dict)
            else ""
        )
        if reference_method in {"agentkv", "commitkv"}:
            # These methods checkpoint generated KV during decode. Persist the
            # exact raw token stream instead of rolling back and re-prefilling
            # a structured serialization, which would revive evicted tokens.
            committed_len = int(req.kv_committed_len)
            allocated_len = int(req.kv_allocated_len)
            if not 0 <= committed_len <= allocated_len:
                raise RuntimeError(
                    "PERSISTENT_HISTORY_SESSION_INVALID_FINISHED_LENGTHS"
                )
            correction = int(getattr(req, "c2kv_position_correction", 0) or 0)
            ledger = list(getattr(req, "history_kv_resident_positions", None) or [])
            if len(ledger) > committed_len:
                raise RuntimeError("PERSISTENT_HISTORY_SESSION_FINISHED_LEDGER_MISMATCH")
            if len(ledger) < committed_len:
                start = len(ledger) + correction
                ledger.extend(range(start, start + committed_len - len(ledger)))
            logical_start = getattr(req, "reference_decode_logical_start", None)
            if logical_start is None:
                raise RuntimeError(
                    "PERSISTENT_HISTORY_SESSION_GENERATION_START_UNAVAILABLE"
                )
            logical_start = int(logical_start)
            computed_horizon = committed_len + correction
            output_ids = list(getattr(req, "output_ids", None) or [])
            computed_output_tokens = computed_horizon - logical_start
            if not 0 <= computed_output_tokens <= len(output_ids):
                raise RuntimeError(
                    "PERSISTENT_HISTORY_SESSION_OUTPUT_HORIZON_MISMATCH"
                )
            normal_output_ids = []
            for position in ledger:
                if logical_start <= position < computed_horizon:
                    output_index = position - logical_start
                    if not 0 <= output_index < computed_output_tokens:
                        raise RuntimeError(
                            "PERSISTENT_HISTORY_SESSION_OUTPUT_POSITION_MISMATCH"
                        )
                    normal_output_ids.append(output_ids[output_index])
            req.persistent_session_active_output_ids = (
                normal_output_ids + output_ids[computed_output_tokens:]
            )
            if (
                len(req.origin_input_ids) + len(normal_output_ids)
                != committed_len
            ):
                raise RuntimeError(
                    "PERSISTENT_HISTORY_SESSION_ACTIVE_STREAM_LENGTH_MISMATCH"
                )
            req.history_kv_resident_positions = ledger

            row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            keep_slots = row[:committed_len].long()
            keep_pages = torch.unique(keep_slots[keep_slots > 0] // self.page_size)
            tail_slots = row[committed_len:allocated_len].long()
            explicit = getattr(req, "persistent_decode_cache_locs", None) or []
            if explicit:
                tail_slots = torch.cat(
                    [
                        tail_slots,
                        torch.stack([slot.reshape(()) for slot in explicit])
                        .long()
                        .to(row.device),
                    ]
                )
            tail_pages = torch.unique(tail_slots[tail_slots > 0] // self.page_size)
            free_pages = tail_pages[~torch.isin(tail_pages, keep_pages)]
            if free_pages.numel():
                self.token_to_kv_pool_allocator.free(free_pages * self.page_size)
            row[committed_len:allocated_len] = 0
            req.persistent_decode_cache_locs = []
            req.kv_allocated_len = committed_len
            req.already_computed = committed_len
            report = getattr(req, "kv_memory_report", None)
            if isinstance(report, dict):
                state = getattr(req, "history_kv_reference_state", None)
                if state is not None and hasattr(state, "layers"):
                    slots = sum(
                        int(layer.key.shape[0] * layer.key.shape[1])
                        for layer in state.layers.values()
                    )
                    denominator = sum(
                        int(layer.key.shape[0])
                        for layer in state.layers.values()
                    )
                    active = (slots + max(1, denominator) - 1) // max(
                        1, denominator
                    )
                    report.update(
                        active_history_kv_tokens=active,
                        active_full_raw_tokens=active,
                        active_history_kv_tokens_source=(
                            "reference_history_physical_token_equivalent"
                        ),
                        history_kv_backend="reference_attention",
                        reference_attention_backend="torch_sdpa",
                        history_kv_runtime_status="reference_attention_ok",
                        reference_history_token_slots=slots,
                        reference_history_resident_bytes=int(state.resident_bytes),
                        reference_history_layer_count=len(state.layers),
                        reference_history_selection_metadata=dict(
                            state.selection_metadata
                        ),
                    )
                report.update(
                    persistent_session_continuation_mode="exact_generated_prefix",
                    persistent_session_computed_logical_horizon=computed_horizon,
                    persistent_session_uncomputed_output_tokens=(
                        len(output_ids) - computed_output_tokens
                    ),
                    persistent_session_active_output_tokens=len(
                        req.persistent_session_active_output_ids
                    ),
                    persistent_session_discarded_decode_kv_tokens=0,
                    persistent_session_reclaimed_decode_kv_tokens=(
                        int(free_pages.numel()) * self.page_size
                    ),
                    persistent_session_prompt_physical_tokens=committed_len,
                    persistent_session_decode_page_free_scope=(
                        "overallocated_pages_excluding_committed_stream"
                    ),
                )
            return

        # Periodic decode checkpoints may hold raw generated K/V in a
        # transient reference state for same-generation attention. The next
        # turn reserializes that output canonically, so transfer only the
        # state that existed at generation start into the persistent slot.
        if hasattr(req, "reference_decode_persistent_state"):
            req.history_kv_reference_state = (
                req.reference_decode_persistent_state
            )
            req.reference_decode_persistent_state = None
        prompt_len = len(req.origin_input_ids)
        committed_len = int(req.kv_committed_len)
        allocated_len = int(req.kv_allocated_len)
        if not prompt_len <= committed_len <= allocated_len:
            raise RuntimeError("PERSISTENT_HISTORY_SESSION_INVALID_FINISHED_LENGTHS")
        row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        prompt_slots = row[:prompt_len].long()
        if prompt_len and (prompt_slots <= 0).any():
            raise RuntimeError("PERSISTENT_HISTORY_SESSION_MISSING_PROMPT_SLOT")
        prompt_pages = torch.unique(prompt_slots // self.page_size)
        tail_slots = row[prompt_len:allocated_len].long()
        explicit = getattr(req, "persistent_decode_cache_locs", None) or []
        if explicit:
            tail_slots = torch.cat([tail_slots, torch.stack(
                [slot.reshape(()) for slot in explicit]).long().to(row.device)])
        tail_pages = torch.unique(tail_slots[tail_slots > 0] // self.page_size)
        free_pages = tail_pages[~torch.isin(tail_pages, prompt_pages)]
        if free_pages.numel():
            self.token_to_kv_pool_allocator.free(free_pages * self.page_size)
        row[prompt_len:allocated_len] = 0
        req.persistent_decode_cache_locs = []
        req.kv_committed_len = prompt_len
        req.kv_allocated_len = prompt_len
        req.already_computed = prompt_len
        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict):
            state = getattr(req, "history_kv_reference_state", None)
            if state is not None:
                slots = sum(
                    int(layer.key.shape[0] * layer.key.shape[1])
                    for layer in state.layers.values()
                )
                denominator = sum(
                    int(layer.key.shape[0])
                    for layer in state.layers.values()
                )
                active = (slots + max(1, denominator) - 1) // max(
                    1, denominator
                )
                report.update(
                    active_history_kv_tokens=active,
                    active_full_raw_tokens=active,
                    reference_history_token_slots=slots,
                    reference_history_resident_bytes=int(state.resident_bytes),
                    reference_history_layer_count=len(state.layers),
                    reference_history_selection_metadata=dict(
                        state.selection_metadata
                    ),
                )
            else:
                physical_eviction = report.get("history_kv_physical_eviction")
                physical_eviction_succeeded = (
                    isinstance(physical_eviction, dict)
                    and physical_eviction.get("success") is True
                )
                measured_history = (
                    int(physical_eviction["kept_history_tokens"])
                    if physical_eviction_succeeded
                    else 0
                )
                report.update(
                    active_history_kv_tokens=measured_history,
                    active_full_raw_tokens=measured_history,
                    reference_history_token_slots=0,
                    reference_history_resident_bytes=0,
                    reference_history_layer_count=0,
                )
                if physical_eviction_succeeded:
                    report["active_history_kv_tokens_source"] = "physical_eviction_measured"
            report["persistent_session_discarded_decode_kv_tokens"] = allocated_len - prompt_len
            report["persistent_session_reclaimed_decode_kv_tokens"] = int(free_pages.numel()) * self.page_size
            report["persistent_session_prompt_physical_tokens"] = prompt_len
            report["persistent_session_decode_page_free_scope"] = "request_owned_pages_excluding_prompt_pages"

    def _trim_persistent_generation_prefix(self, slot: SessionSlot, req: Req) -> None:
        """Replace only the verified generation scaffold for an evidence append.

        Canonical body positions missing after compression remain missing.
        Page reclamation uses physical ownership, including partial tail pages.
        The logical horizon and physical length shrink together, so the RoPE
        correction for older evicted positions does not change.
        """
        hint = req.c2kv_kv_memory_hint or {}
        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict) and isinstance(
            report.get("persistent_session_generation_prefix_splice"), dict
        ):
            return
        rollback_free_pages = torch.empty(0, dtype=torch.long)
        rollback_generated = 0
        logical_prefix = int(hint["persistent_session_logical_prefix_tokens"])
        drop = int(hint["persistent_session_drop_generation_prefix_tokens"])
        positions = list(slot.history_kv_resident_positions or [])
        tail = [p for p in positions if p >= logical_prefix]
        if tail and tail != list(range(logical_prefix, logical_prefix + drop)):
            raise RuntimeError("PERSISTENT_HISTORY_RECOVERY_GENERATION_PREFIX_LEDGER_MISMATCH")
        # match_prefix can run repeatedly while a request waits for admission.
        trim = len(tail)
        old_len = int(slot.kv_allocated_len)
        if len(positions) != old_len or int(slot.kv_committed_len) != old_len:
            raise RuntimeError("PERSISTENT_HISTORY_RECOVERY_SLOT_LENGTH_MISMATCH")
        keep_len = old_len - trim
        row = self.req_to_token_pool.req_to_token[slot.req_pool_idx]
        keep_slots = row[:keep_len].long()
        tail_slots = row[keep_len:old_len].long()
        keep_pages = torch.unique(keep_slots[keep_slots > 0] // self.page_size)
        tail_pages = torch.unique(tail_slots[tail_slots > 0] // self.page_size)
        free_pages = tail_pages[~torch.isin(tail_pages, keep_pages)]
        if free_pages.numel():
            self.token_to_kv_pool_allocator.free(free_pages * self.page_size)
        row[keep_len:old_len] = 0
        slot.kv_committed_len = keep_len
        slot.kv_allocated_len = keep_len
        slot.history_kv_resident_positions = positions[:keep_len]
        resident = set(slot.history_kv_resident_positions)
        slot.history_kv_score_state = {
            layer: {p: score for p, score in scores.items() if p in resident}
            for layer, scores in (slot.history_kv_score_state or {}).items()
        }
        if isinstance(report, dict):
            report["persistent_session_generation_prefix_splice"] = {
                "enabled": True,
                "session_id": req.session.session_id,
                "generation_prefix_tokens": drop,
                "retained_body_physical_tokens": keep_len,
                "retained_body_resident_page_tokens": (
                    int(keep_pages.numel()) * self.page_size
                ),
                "freed_physical_page_tokens": int(free_pages.numel()) * self.page_size,
                "rolled_back_generated_tokens": rollback_generated,
                "rollback_freed_physical_page_tokens": (
                    int(rollback_free_pages.numel()) * self.page_size
                ),
                "full_history_reprefill_performed": False,
                "scope": "verified_generation_prefix_only",
            }

    def _rollback_failed_persistent_request(self, req: Req) -> None:
        """Drop a failed turn's partial suffix while preserving prior session KV."""
        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        if slot is None or slot.req_pool_idx is None:
            self.slots.pop(session_id, None)
            self.inner.cache_finished_req(req, is_insert=False)
            return

        keep_len = int(slot.kv_committed_len)
        allocated_len = int(getattr(req, "kv_allocated_len", keep_len) or 0)
        row = self.req_to_token_pool.req_to_token[slot.req_pool_idx]
        keep_slots = row[:keep_len].long()
        keep_pages = torch.unique(keep_slots[keep_slots > 0] // self.page_size)
        tail_slots = row[keep_len:max(keep_len, allocated_len)].long()
        tail_pages = torch.unique(tail_slots[tail_slots > 0] // self.page_size)
        free_pages = tail_pages[~torch.isin(tail_pages, keep_pages)]
        if free_pages.numel():
            self.token_to_kv_pool_allocator.free(free_pages * self.page_size)
        row[keep_len:max(keep_len, allocated_len)] = 0

        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict):
            report["persistent_session_failed_turn_rolled_back"] = True
            report["persistent_session_rollback_kept_tokens"] = keep_len
            report["persistent_session_rollback_freed_pages"] = int(
                free_pages.numel()
            )
        req.req_pool_idx = None
        req.mamba_pool_idx = None
        req.history_kv_reference_state = None
        req.history_kv_runtime_state = None
        req.reference_decode_persistent_state = None
        req.reference_decode_baseline_runtime_state = None
        # SessionSlot is now the sole owner while the request node remains
        # available for streaming-session protocol bookkeeping.

    def _adopt_legacy_persistent_prefix(
        self, slot: SessionSlot, logical_prefix: int
    ) -> None:
        """Trim a pre-marker streaming slot before persistent continuation.

        Older callers omitted ``persistent_history_session`` on the first
        turn. The slot then contains the canonical prompt plus raw decode KV,
        but has no resident-position ledger. The continuation request carries
        the canonical prefix length, so reclaim the suffix by physical page
        ownership before the slot is reused.
        """

        old_len = int(slot.kv_allocated_len)
        keep_len = max(int(logical_prefix), int(slot.cache_protected_len))
        if keep_len > old_len:
            raise RuntimeError(
                "PERSISTENT_HISTORY_LEGACY_PREFIX_EXCEEDS_SESSION: "
                f"{keep_len=}, {old_len=}"
            )
        row = self.req_to_token_pool.req_to_token[slot.req_pool_idx]
        keep_slots = row[:keep_len].long()
        keep_pages = torch.unique(keep_slots[keep_slots > 0] // self.page_size)
        tail_slots = row[keep_len:old_len].long()
        tail_pages = torch.unique(tail_slots[tail_slots > 0] // self.page_size)
        free_pages = tail_pages[~torch.isin(tail_pages, keep_pages)]
        if free_pages.numel():
            self.token_to_kv_pool_allocator.free(free_pages * self.page_size)
        row[keep_len:old_len] = 0

        slot.kv_committed_len = keep_len
        slot.kv_allocated_len = keep_len
        slot.history_kv_resident_positions = list(range(keep_len))
        slot.history_kv_score_state = {}
        logging.getLogger(__name__).info(
            "PERSISTENT_HISTORY_LEGACY_SLOT_ADOPTED %s",
            json.dumps(
                {
                    "kept_prompt_tokens": keep_len,
                    "discarded_decode_tokens": old_len - keep_len,
                    "freed_pages": int(free_pages.numel()),
                },
                sort_keys=True,
            ),
        )

    def cache_unfinished_req(self, req: Req, **kwargs):
        if getattr(req, "session_cache_closed_during_request", False):
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return
        if _is_streaming(req):
            if self._is_persistent_history_req(req):
                # Physical history eviction rewrites and frees request-owned
                # KV pages.  Those pages must never also be inserted into the
                # radix tree: after compaction its cached prefix length can be
                # larger than the entire resident session (for example a
                # 256-token cached chunk compacted to 134 tokens), producing
                # negative session-held accounting and stale tree references.
                # Keep the already computed prefix directly on the request;
                # SessionSlot takes sole ownership when the request finishes.
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : len(req.fill_ids)
                ]
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
                req.cache_protected_len = 0
                return
            # in chunked_prefill for streaming, we skip the stash path which triggers radix.
            # only the last chunk in first turn trigger a full prompt radix insert.
            if kwargs.get("chunked", False):
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : len(req.fill_ids)
                ]
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
                return
            if req.session.session_id in self.slots:
                # Subsequent turns: slot exists, skip inner entirely.
                return
            # First turn (no slot): fall through to inner for lock management,
            # tree insertion, and cache_protected_len updates between chunks.
        self.inner.cache_unfinished_req(req, **kwargs)

    def evict(self, params: EvictParams) -> EvictResult:
        return self.inner.evict(params)

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if isinstance(node, _VirtualNode):
            return IncLockRefResult()
        return self.inner.inc_lock_ref(node)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if isinstance(node, _VirtualNode):
            return DecLockRefResult()
        return self.inner.dec_lock_ref(node, params)

    # -- Session lifecycle --

    def release_session(self, session_id: str, active_req: Optional[Req] = None):
        """Release all KV resources held by a streaming session."""
        slot = self.slots.pop(session_id, None)
        if slot is None:
            return

        # restore_to_req leaves the slot's pool index intact for retry. While
        # that request is running, however, the slot and request point to the
        # same KV. An explicit close transfers ownership to the request, whose
        # normal completion path releases it after req.session is cleared.
        borrowed_by_active_req = (
            active_req is not None
            and active_req.req_pool_idx is not None
            and active_req.req_pool_idx == slot.req_pool_idx
        )

        if borrowed_by_active_req:
            # The running request now owns both its KV row and the slot's
            # radix lock. Its regular completion path releases the lock once.
            active_req.session_cache_closed_during_request = True
            active_req.last_node = slot.last_node
            active_req.swa_uuid_for_lock = slot.swa_uuid_for_lock
            active_req.cache_protected_len = slot.cache_protected_len
        elif slot.last_node is not None:
            if slot.swa_uuid_for_lock is not None:
                self.inner.dec_lock_ref(
                    slot.last_node,
                    DecLockRefParams(swa_uuid_for_lock=slot.swa_uuid_for_lock),
                )
            else:
                self.inner.dec_lock_ref(slot.last_node)

        if slot.is_holding_kv and not borrowed_by_active_req:
            start = slot.cache_protected_len
            end = slot.kv_allocated_len
            if start < end:
                kv_indices = self.req_to_token_pool.req_to_token[
                    slot.req_pool_idx, start:end
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_slots.append(slot.req_pool_idx)
        logging.getLogger(__name__).info("HISTORY_KV_SESSION_CLOSED %s", json.dumps({
            "session_id": session_id,
            "resident_tokens_released": (
                0 if borrowed_by_active_req else slot.kv_allocated_len
            ),
            "resident_tokens_transferred_to_active_req": (
                slot.kv_allocated_len if borrowed_by_active_req else 0
            ),
            "remaining_session_slots": len(self.slots)}))

    def session_held_tokens(self) -> int:
        """Total KV tokens held by session slots, not tracked by the tree."""
        total = 0
        for slot in self.slots.values():
            if slot.is_holding_kv:
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)
                total += allocated - slot.cache_protected_len
        return total

    def session_held_full_tokens(self) -> int:
        """An alias to align the naming style of SWA"""
        return self.session_held_tokens()

    def session_held_swa_tokens(self) -> int:
        """Total SWA tokens held by session slots, not tracked by the tree."""
        total = 0
        for slot in self.slots.values():
            if slot.is_holding_kv:
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)
                total += allocated - max(
                    slot.cache_protected_len, slot.swa_evicted_seqlen
                )
        return total

    def session_held_req_count(self) -> int:
        """Number of req pool slots held by session slots."""
        return sum(s.is_holding_kv for s in self.slots.values())

    # -- Pass-through methods --

    def evictable_size(self):
        return self.inner.evictable_size()

    def full_evictable_size(self):
        return self.inner.full_evictable_size()

    def swa_evictable_size(self):
        return self.inner.swa_evictable_size()

    def protected_size(self):
        return self.inner.protected_size()

    def full_protected_size(self):
        return self.inner.full_protected_size()

    def swa_protected_size(self):
        return self.inner.swa_protected_size()

    def total_size(self):
        return self.inner.total_size()

    def pretty_print(self):
        return self.inner.pretty_print()

    def init_load_back(self, params: InitLoadBackParams):
        return self.inner.init_load_back(params)

    def ready_to_load_host_cache(self):
        return self.inner.ready_to_load_host_cache()

    def flush_write_through_acks(self) -> None:
        return self.inner.flush_write_through_acks()

    def check_hicache_events(self):
        return self.inner.check_hicache_events()

    def take_events(self):
        return self.inner.take_events()

    def supports_swa(self):
        return self.inner.supports_swa()

    def supports_mamba(self):
        return self.inner.supports_mamba()

    def is_chunk_cache(self):
        return self.inner.is_chunk_cache()

    def is_tree_cache(self):
        return self.inner.is_tree_cache()

    def available_and_evictable_str(self):
        return self.inner.available_and_evictable_str()

    def init_metrics_collector(self):
        return self.inner.init_metrics_collector()

    def sanity_check(self):
        # Skip inner sanity check when sessions hold tree locks, because
        # the check asserts all nodes are unlocked during idle.
        if any(s.is_holding_kv for s in self.slots.values()):
            return
        self.inner.sanity_check()

    # Forward attribute access for cache-specific methods (e.g.
    # sliding_window_size, all_values_flatten, etc.)
    def __getattr__(self, name):
        return getattr(self.inner, name)
