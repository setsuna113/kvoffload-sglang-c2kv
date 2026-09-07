from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.routed_experts_capturer import get_global_experts_capturer
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.schedule_batch import (
    BaseFinishReason,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = logging.getLogger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def _get_storage_backend_type(self) -> str:
        """Get storage backend type from tree_cache."""
        storage_backend_type = "none"
        cache_controller = getattr(self.tree_cache, "cache_controller", None)
        if cache_controller and hasattr(cache_controller, "storage_backend"):
            storage_backend = cache_controller.storage_backend
            if storage_backend is not None:
                storage_backend_type = type(storage_backend).__name__
        return storage_backend_type

    def _get_cached_tokens_details(self: Scheduler, req: Req) -> Optional[dict]:
        """Get detailed cache breakdown for a request, if available.

        Returns:
            - None if no cached tokens at all
            - {"device": X, "host": Y} without storage breakdown
            - {"device": X, "host": Y, "storage": Z} with storage breakdown
        """
        if (
            req.cached_tokens_device > 0
            or req.cached_tokens_host > 0
            or req.cached_tokens_storage > 0
        ):
            details = {
                "device": req.cached_tokens_device,
                "host": req.cached_tokens_host,
            }
            # Only include storage fields if L3 storage is enabled
            if getattr(self, "enable_hicache_storage", False):
                details["storage"] = req.cached_tokens_storage
                details["storage_backend"] = self._get_storage_backend_type()
            return details

        if req.cached_tokens > 0:
            return {
                "device": req.cached_tokens,
                "host": 0,
            }

        return None

    def _get_kv_runtime_stats(self: Scheduler, req=None) -> Optional[dict]:
        """Return a real KV allocator residency snapshot for accounting.

        When `req` is given, the request's C2KV layout ledger (gist / repair
        injections with their RoPE positions) and the server's query-projection
        mode are attached so the client can verify position-frame consistency
        and record provenance. See c2kv/c2kv_serving_semantics.md.
        """

        snapshot = self._get_physical_kv_snapshot()
        if snapshot is None:
            return None
        stats = {
            "kv_pool_size": snapshot["main_kv_pool_size"],
            "kv_available_tokens": snapshot["main_kv_available_slots"],
            "kv_resident_tokens": snapshot["physical_main_kv_slots"],
            "kv_peak_resident_tokens": snapshot["peak_main_paged_kv_slots"],
            "kv_page_size": snapshot["page_size_tokens"],
            **snapshot,
        }
        server_args = getattr(self, "server_args", None)
        if server_args is not None:
            # D5: the tool-serialization mode decides the rendered prompt of
            # /v1/chat/completions, /v1/c2kv/extract and the `messages` form of
            # /v1/c2kv/repair_extract alike (http_server._c2kv_flat_tools,
            # serving_chat._chat_template_tools), i.e. the token frame every
            # client-computed insertion point and repair span is measured in.
            # It is echoed whether or not C2KV is enabled so a client can check
            # that the frame it assumed is the frame the server served.
            stats["c2kv_tools_dump"] = getattr(server_args, "c2kv_tools_dump", "full")
        c2kv_enabled = server_args is not None and bool(
            getattr(server_args, "enable_c2kv", False)
        )
        if c2kv_enabled:
            # D6 projection provenance: THREE keys, deliberately not merged.
            #
            #   c2kv_query_proj            the --c2kv-query-proj SERVER FLAG.
            #                              Constant for the life of the run,
            #                              exactly what serve-align echoed; the
            #                              bench's mixed-regime check keys on
            #                              it, so it must never vary with the
            #                              request (turn 1 of a conversation
            #                              carries no gisted history yet, and a
            #                              per-request value here made a
            #                              single-flag run self-report as
            #                              mixing projection regimes).
            #   c2kv_query_proj_effective  what THIS request actually ran:
            #                              "gist" iff some token of it was
            #                              projected with gist_{q,k,v}_proj.
            #                              The per-request resolver in
            #                              Scheduler.handle_generate_request
            #                              runs inside `if
            #                              recv_req.c2kv_segments:`, and a
            #                              segment-less request builds no mask
            #                              at all (schedule_batch.py
            #                              get_model_worker_batch ->
            #                              forward_batch_info.py), so it ran
            #                              base whatever the flag says.
            #   c2kv_query_proj_source     which rule chose the mode:
            #                              "request" (request-wide override),
            #                              "message" (agreeing message-level
            #                              overrides),
            #                              "flag" (--c2kv-query-proj), or
            #                              "none" (no C2KV segments, so no
            #                              projection decision was ever made
            #                              for this request).
            stats["c2kv_query_proj"] = getattr(server_args, "c2kv_query_proj", "base")
            if req is not None and getattr(req, "c2kv_segments", None):
                stats["c2kv_query_proj_effective"] = (
                    "gist"
                    if getattr(req, "c2kv_use_gist_projection", False)
                    else "base"
                )
                stats["c2kv_query_proj_source"] = getattr(
                    req, "c2kv_query_proj_source", "flag"
                )
            else:
                stats["c2kv_query_proj_effective"] = "base"
                stats["c2kv_query_proj_source"] = "none"
            # CUDA and NPU full-graph runners own a dynamic projection-mask
            # buffer. CPU and piecewise runners still fall back to eager for a
            # request that uses the gist projection.
            device = str(getattr(server_args, "device", "")).lower()
            graph_eligible = (
                stats["c2kv_query_proj_effective"] != "gist"
                or device in {"cuda", "npu"}
            )
            stats["c2kv_query_proj_graph_eligible"] = graph_eligible
            stats["c2kv_query_proj_decode_verified"] = True
        if req is not None:
            layout = getattr(req, "c2kv_layout", None)
            if layout:
                stats["c2kv_layout"] = list(layout)
                stats["c2kv_position_correction"] = int(
                    getattr(req, "c2kv_position_correction", 0) or 0
                )
                stats["c2kv_gist_seen"] = bool(getattr(req, "c2kv_gist_seen", False))
            elif c2kv_enabled:
                # Positive "nothing was injected" signal. An absent key is
                # ambiguous -- it also happens when the whole stats dict is
                # dropped for a missing allocator, or on a build that predates
                # the ledger -- so on a C2KV-enabled server the three ledger
                # keys are always present and an empty list means "this request
                # injected nothing" (a `full`-arm request, or turn 1 of a
                # compression arm). Off a C2KV server the keys stay absent, so
                # an upstream client sees exactly what it saw before.
                stats["c2kv_layout"] = []
                stats["c2kv_position_correction"] = 0
                stats["c2kv_gist_seen"] = False
            # Machine-readable reason the injection failed, when one did. The
            # abort that carries it has no status_code (the FINISH_ABORT built
            # in process_batch_result_prefill below), so the response is an
            # HTTP 200 whose finish_reason.type is "abort"; this key, and
            # ChatCompletionResponse.metadata.finish_message next to it, are
            # the only places an OpenAI client can read WHY. Absent when the
            # request injected without failing. See
            # c2kv/c2kv_serving_semantics.md section 3.
            injection_error = getattr(req, "c2kv_injection_error", None)
            if injection_error:
                stats["c2kv_injection_error"] = str(injection_error)
        return stats

    def _bytes_per_kv_token(self: Scheduler) -> Optional[int]:
        try:
            model_runner = self.tp_worker.model_runner
            dtype = getattr(model_runner, "dtype", None)
            if dtype is None:
                dtype = getattr(self.token_to_kv_pool_allocator, "dtype", torch.float16)
            dtype_bytes = torch.tensor([], dtype=dtype).element_size()
            num_layers = int(getattr(model_runner, "num_effective_layers"))
            num_kv_heads = int(
                self.model_config.get_num_kv_heads(self.attn_tp_group.world_size)
            )
            head_dim = int(self.model_config.head_dim)
            value_head_dim = int(getattr(self.model_config, "v_head_dim", head_dim))
            return num_layers * num_kv_heads * (head_dim + value_head_dim) * dtype_bytes
        except Exception:
            return None

    def _get_physical_kv_snapshot(self: Scheduler) -> Optional[dict]:
        """Return physical GPU KV residency for main paged KV and C2KV pool."""

        allocator = getattr(self, "token_to_kv_pool_allocator", None)
        if allocator is None:
            return None
        try:
            size = int(getattr(allocator, "size", 0) or 0)
            available = int(allocator.available_size())
            page_size = int(getattr(allocator, "page_size", 1) or 1)
            main_slots = max(0, size - available)
            main_pages = (
                (main_slots + page_size - 1) // page_size if page_size > 0 else 0
            )

            c2kv_pool = getattr(self, "c2kv_pool", None)
            c2kv_slots = int(c2kv_pool.current_tokens()) if c2kv_pool is not None else 0
            c2kv_pages = c2kv_slots
            if c2kv_pool is not None:
                c2kv_page_size = int(
                    getattr(getattr(c2kv_pool, "allocator", None), "page_size", 1) or 1
                )
                c2kv_pages = (
                    (c2kv_slots + c2kv_page_size - 1) // c2kv_page_size
                    if c2kv_page_size > 0
                    else 0
                )

            bytes_per_token = self._bytes_per_kv_token() or 0
            main_bytes = main_slots * bytes_per_token
            c2kv_bytes = c2kv_slots * bytes_per_token
            total_bytes = main_bytes + c2kv_bytes

            peak_main = max(
                int(getattr(self, "_c2kv_runtime_peak_main_kv_slots", 0) or 0),
                main_slots,
            )
            peak_c2kv = max(
                int(getattr(self, "_c2kv_runtime_peak_c2kv_pool_slots", 0) or 0),
                c2kv_slots,
            )
            peak_total = max(
                int(getattr(self, "_c2kv_runtime_peak_total_gpu_kv_bytes", 0) or 0),
                total_bytes,
            )
            self._c2kv_runtime_peak_main_kv_slots = peak_main
            self._c2kv_runtime_peak_c2kv_pool_slots = peak_c2kv
            self._c2kv_runtime_peak_total_gpu_kv_bytes = peak_total
            return {
                "main_kv_pool_size": size,
                "main_kv_available_slots": available,
                "page_size_tokens": page_size,
                "bytes_per_kv_token": bytes_per_token,
                "physical_main_kv_pages": main_pages,
                "physical_main_kv_slots": main_slots,
                "physical_main_kv_bytes": main_bytes,
                "physical_c2kv_pool_pages": c2kv_pages,
                "physical_c2kv_pool_slots": c2kv_slots,
                "physical_c2kv_pool_bytes": c2kv_bytes,
                "total_gpu_kv_bytes": total_bytes,
                "peak_main_paged_kv_slots": peak_main,
                "peak_c2kv_pool_slots": peak_c2kv,
                "peak_total_gpu_kv_bytes": peak_total,
            }
        except Exception:
            return None

    def _get_kv_memory_report(self: Scheduler, req: Req) -> Optional[dict]:
        report = getattr(req, "kv_memory_report", None)
        if not isinstance(report, dict):
            snapshot = getattr(req, "history_kv_eviction_report_snapshot", None)
            if isinstance(snapshot, dict):
                logger.warning(
                    "KV-memory report recovered from eviction snapshot rid=%s",
                    req.rid,
                )
                report = dict(snapshot)
        if not isinstance(report, dict):
            return None
        snapshot = self._get_physical_kv_snapshot()
        if snapshot is not None:
            report = {**report, **snapshot}
        return report

    def process_batch_result_prebuilt(self: Scheduler, batch: ScheduleBatch):
        assert self.disaggregation_mode == DisaggregationMode.DECODE
        for req in batch.reqs:
            req.time_stats.set_decode_prebuilt_finish_time()
            req.check_finished()
            if req.finished():
                req.time_stats.set_quick_finish_time()
                self._release_c2kv_pins(req)
                release_kv_cache(
                    req,
                    self.tree_cache,
                    is_insert=req.c2kv_rounds is None,
                )

        # Note: Logprobs should be handled on the prefill engine.
        self.stream_output(batch.reqs, batch.return_logprob)

    def maybe_collect_routed_experts(self: Scheduler, req: Req):
        """Collect routed experts for a finished request."""
        req.routed_experts = get_global_experts_capturer().get_routed_experts(
            req_pool_idx=req.req_pool_idx,
            seqlen=req.seqlen,
            req_to_token_pool=self.req_to_token_pool,
        )

    def maybe_collect_customized_info(
        self: Scheduler, i: int, req: Req, logits_output: LogitsProcessorOutput
    ):
        if logits_output is not None and logits_output.customized_info is not None:
            if req.customized_info is None:
                req.customized_info = {}
            for k, v in logits_output.customized_info.items():
                if k not in req.customized_info:
                    req.customized_info[k] = []
                # Copy the element so it doesn't retain the entire batch
                # tensor/array via a view reference.
                elem = v[i]
                if isinstance(elem, torch.Tensor):
                    elem = elem.clone()
                elif hasattr(elem, "copy") and callable(elem.copy):
                    elem = elem.copy()
                req.customized_info[k].append(elem)

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        skip_stream_req = None

        if self.is_generation:
            if result.copy_done is not None:
                result.copy_done.synchronize()

            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            # Move next_token_ids and logprobs to cpu
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                if logits_output.next_token_logprobs is not None:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                if logits_output.input_token_logprobs is not None:
                    logits_output.input_token_logprobs = tuple(
                        logits_output.input_token_logprobs.tolist()
                    )
                if logits_output.next_token_top_logprobs_val:
                    logits_output.next_token_top_logprobs_val = [
                        v.tolist() for v in logits_output.next_token_top_logprobs_val
                    ]
                    logits_output.next_token_top_logprobs_idx = [
                        x.tolist() for x in logits_output.next_token_top_logprobs_idx
                    ]
                if logits_output.next_token_token_ids_logprobs_val:
                    logits_output.next_token_token_ids_logprobs_val = [
                        v.tolist()
                        for v in logits_output.next_token_token_ids_logprobs_val
                    ]

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0

            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.finished() or req.is_retracted:
                    # decode req in mixed batch or retracted req
                    continue

                if req.is_chunked <= 0:
                    req.time_stats.set_prefill_finished_time()

                    # C2KV multi-round prefill: inject gist segments and maybe re-queue
                    if req.c2kv_rounds is not None:
                        cur_round = req.c2kv_rounds[req.c2kv_round_idx]
                        self._log_c2kv_token_usage(
                            "prefill_round_finished",
                            req=req,
                            batch_seq_len=int(batch.seq_lens_cpu[i].item()),
                            post_inject=list(cur_round.post_inject_seg_indices),
                        )

                        # Inject all gist segments scheduled after this round
                        abort = False
                        logical_kv_start = int(batch.seq_lens_cpu[i].item())
                        for seg_idx in cur_round.post_inject_seg_indices:
                            if not self._inject_c2kv_gist_segment(
                                req, seg_idx, logical_kv_start
                            ):
                                logger.warning(
                                    f"C2KV injection failed for {req.rid}; aborting"
                                )
                                from sglang.srt.managers.schedule_batch import FINISH_ABORT as _FA

                                # Machine-readable reason: every injection
                                # failure path in Scheduler sets one through
                                # _set_c2kv_injection_error, so this is a
                                # C2KV_* code (C2KV_CACHE_MISS,
                                # C2KV_APPEND_TAIL_REQUIRES_PRE_ROPE,
                                # C2KV_ALLOC_FAILED, ...) rather than the bare
                                # fallback. This is mid-prefill, so it stays an
                                # abort finish reason with NO status_code
                                # (HTTP 200, meta_info.finish_reason.message)
                                # exactly as before -- unlike the
                                # pre-scheduling admission errors, which go
                                # through set_finish_with_abort and are a 400.
                                # A /generate client reads the message off
                                # meta_info.finish_reason; an OpenAI client
                                # reads it off
                                # metadata.sglang_runtime.c2kv_injection_error
                                # (_get_kv_runtime_stats above) or
                                # metadata.finish_message
                                # (serving_chat._build_chat_response), because
                                # the choice carries only
                                # finish_reason["type"].
                                req.to_finish = _FA(
                                    getattr(req, "c2kv_injection_error", None)
                                    or "C2KV injection failed"
                                )
                                req.check_finished()
                                self._log_c2kv_token_usage(
                                    "inject_abort_release_before",
                                    req=req,
                                    failed_seg_idx=seg_idx,
                                )
                                self._release_c2kv_pins(req)
                                release_kv_cache(req, self.tree_cache, is_insert=False)
                                self._log_c2kv_token_usage(
                                    "inject_abort_release_after",
                                    req=req,
                                    failed_seg_idx=seg_idx,
                                )
                                self.stream_output([req], req.return_logprob)
                                abort = True
                                break
                            logical_kv_start = req.kv_committed_len
                        if abort:
                            continue

                        if getattr(cur_round, "post_history_kv_eviction", False):
                            score_map = getattr(result, "history_kv_selection_scores", None)
                            if isinstance(score_map, dict):
                                req.history_kv_selection_scores = score_map.get(
                                    int(req.req_pool_idx)
                                )
                            if not self._apply_history_kv_eviction(req):
                                self._release_c2kv_pins(req)
                                release_kv_cache(req, self.tree_cache, is_insert=False)
                                self.stream_output([req], req.return_logprob)
                                continue

                        req.c2kv_round_idx += 1

                        if req.c2kv_round_idx < len(req.c2kv_rounds):
                            next_round = req.c2kv_rounds[req.c2kv_round_idx]
                            model_runner = self.tp_worker.model_runner

                            req.prefix_indices = (
                                model_runner.req_to_token_pool.req_to_token[
                                    req.req_pool_idx, : req.kv_committed_len
                                ].to(torch.int64)
                            )
                            req.already_computed = req.kv_committed_len
                            req.c2kv_round_start_len = req.kv_committed_len

                            # Release the tree lock from this round's PrefillAdder;
                            # the next round's PrefillAdder will re-acquire it.
                            if req.last_node is not None:
                                self.tree_cache.dec_lock_ref(req.last_node)

                            self._log_c2kv_token_usage(
                                "requeue_next_round",
                                req=req,
                                next_round_idx=req.c2kv_round_idx,
                                next_round_len=len(next_round.tokens),
                                prefix_indices_len=len(req.prefix_indices),
                                round_start_len=req.c2kv_round_start_len,
                                c2kv_virtual_len=(
                                    len(req.c2kv_virtual_input_ids)
                                    if req.c2kv_virtual_input_ids is not None
                                    else None
                                ),
                            )

                            req.c2kv_requeued = True
                            self.waiting_queue.insert(0, req)
                            continue

                        # If gist was injected this round, batch.seq_lens is stale.
                        # Patch it so decode allocates at the correct position.
                        if (
                            cur_round.post_inject_seg_indices
                            or getattr(cur_round, "post_history_kv_eviction", False)
                        ):
                            old_seq_len = int(batch.seq_lens_cpu[i].item())
                            seq_delta = req.kv_committed_len - old_seq_len
                            if seq_delta != 0:
                                batch.seq_lens_cpu[i] = req.kv_committed_len
                                if batch.seq_lens is None:
                                    batch.seq_lens = batch.seq_lens_cpu.to(
                                        batch.device
                                    )
                                else:
                                    batch.seq_lens[i] += seq_delta
                                if batch.seq_lens_sum is not None:
                                    batch.seq_lens_sum += seq_delta

                        self._log_c2kv_token_usage(
                            "all_rounds_finished",
                            req=req,
                            batch_seq_len=int(batch.seq_lens_cpu[i].item()),
                            c2kv_virtual_len=(
                                len(req.c2kv_virtual_input_ids)
                                if req.c2kv_virtual_input_ids is not None
                                else None
                            ),
                        )
                        persistent_active_ids = getattr(
                            req, "c2kv_persistent_active_input_ids", None
                        )
                        if persistent_active_ids is not None:
                            # Future streaming-session appends must see the
                            # compact physical order rather than the discarded
                            # logical history. c2kv_position_correction keeps
                            # RoPE in the original logical coordinate frame.
                            req.origin_input_ids = list(persistent_active_ids)
                            req.origin_input_ids_unpadded = list(persistent_active_ids)
                            req.c2kv_virtual_input_ids = list(persistent_active_ids)
                        self._release_c2kv_pins(req)

                    # req output_ids are set here
                    req.output_ids.append(next_token_id)

                    self._maybe_update_reasoning_tokens(req, next_token_id)

                    req.check_finished()
                    if req.finished():
                        self.maybe_collect_routed_experts(req)
                        self._release_c2kv_pins(req)
                        release_kv_cache(
                            req,
                            self.tree_cache,
                            is_insert=req.c2kv_rounds is None,
                        )
                        req.time_stats.set_completion_time()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        if req.c2kv_rounds is None:
                            self.tree_cache.cache_unfinished_req(req)
                        if self.enable_hisparse:
                            self.hisparse_coordinator.admit_request_into_staging(req)

                    self.maybe_collect_customized_info(i, req, logits_output)

                    if batch.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]

                        num_input_logprobs = self._calculate_num_input_logprobs(
                            req, extend_input_len, extend_logprob_start_len
                        )

                        if req.return_logprob:
                            self.add_logprob_return_values(
                                i,
                                req,
                                logprob_pt,
                                next_token_ids,
                                num_input_logprobs,
                                logits_output,
                            )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if req.grammar is not None:
                        # FIXME: this try-except block is for handling unexpected xgrammar issue.
                        try:
                            req.grammar.accept_token(next_token_id)
                        except ValueError as e:
                            # Grammar accept_token can raise ValueError if the token is not in the grammar.
                            # This can happen if the grammar is not set correctly or the token is invalid.
                            logger.error(
                                f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                            )
                            self.abort_request(AbortReq(rid=req.rid))
                        req.grammar.finished = req.finished()

                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if batch.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = self._calculate_num_input_logprobs(
                                req, extend_input_len, extend_logprob_start_len
                            )
                            if req.return_logprob:
                                self.add_input_logprob_return_values(
                                    i,
                                    req,
                                    logits_output,
                                    logprob_pt,
                                    num_input_logprobs,
                                    last_prefill_chunk=False,
                                )
                            logprob_pt += num_input_logprobs

                    req.time_stats.set_last_chunked_prefill_finish_time()

        else:  # embedding or reward model
            if result.copy_done is not None:
                result.copy_done.synchronize()

            is_sparse = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()

            embeddings = result.embeddings

            if is_sparse:
                batch_ids, token_ids = embeddings.indices()
                values = embeddings.values()

                embeddings = [{} for _ in range(embeddings.size(0))]
                for i in range(batch_ids.shape[0]):
                    embeddings[batch_ids[i].item()][token_ids[i].item()] = values[
                        i
                    ].item()
            else:
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.tolist()
                else:
                    embeddings = [tensor.tolist() for tensor in embeddings]

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    req.time_stats.set_prefill_finished_time()
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self._release_c2kv_pins(req)
                        release_kv_cache(req, self.tree_cache)
                        req.time_stats.set_completion_time()
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    req.time_stats.set_last_chunked_prefill_finish_time()

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        self.report_prefill_stats(
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _resolve_spec_overlap_token_ids(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> List[List[int]]:
        """Resolve the padding next token ids for speculative decoding with overlap."""
        assert result.next_token_ids.is_cpu
        assert result.accept_lens.is_cpu

        next_token_ids = result.next_token_ids.tolist()
        accept_lens = result.accept_lens.tolist()
        result.num_accepted_tokens = sum(accept_lens) - len(batch.reqs)
        result.accept_length_per_req_cpu = [x - 1 for x in accept_lens]

        predict_tokens = []
        stride = self.draft_worker.speculative_num_draft_tokens

        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_lens[i]
            predict_tokens.append(
                next_token_ids[i * stride : i * stride + accept_lens[i]]
            )
            req.spec_verify_ct += 1

            accepted_draft_tokens = result.accept_length_per_req_cpu[i]
            req.spec_accepted_tokens += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)

        return predict_tokens

    def process_batch_result_idle(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        self.stream_output_generation(
            batch.reqs, batch.return_logprob, is_idle_batch=True
        )

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        logits_output, next_token_ids, can_run_cuda_graph = (
            result.logits_output,
            result.next_token_ids,
            result.can_run_cuda_graph,
        )

        if batch.spec_algorithm.is_none() or batch.is_spec_v2:
            if batch.is_spec_v2:
                next_token_ids = self._resolve_spec_overlap_token_ids(result, batch)
            else:
                next_token_ids = next_token_ids.tolist()

            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()
                if logits_output.next_token_top_logprobs_val:
                    logits_output.next_token_top_logprobs_val = [
                        v.tolist() for v in logits_output.next_token_top_logprobs_val
                    ]
                    logits_output.next_token_top_logprobs_idx = [
                        x.tolist() for x in logits_output.next_token_top_logprobs_idx
                    ]

                if logits_output.next_token_token_ids_logprobs_val:
                    logits_output.next_token_token_ids_logprobs_val = [
                        v.tolist()
                        for v in logits_output.next_token_token_ids_logprobs_val
                    ]
        # else: Spec V1 — output_ids, check_finished, grammar, and reasoning tokens
        # are already handled in the verify phase (eagle_info.py / ngram_info.py).

        self.num_generated_tokens += len(batch.reqs)
        if not batch.spec_algorithm.is_none():
            self.update_spec_metrics(batch.batch_size(), result.num_accepted_tokens)
        if self.enable_metrics:
            self.metrics_collector.increment_decode_cuda_graph_pass(
                value=can_run_cuda_graph
            )

        self.token_to_kv_pool_allocator.free_group_begin()

        # Spec V1 handles output_ids, check_finished, grammar, and reasoning tokens
        # in the verify phase. Non-spec and V2 handle them here in post-processing.
        is_spec_v1 = not batch.spec_algorithm.is_none() and not batch.is_spec_v2

        for i, req in enumerate(batch.reqs):
            req: Req

            if self.enable_overlap and (req.finished() or req.is_retracted):
                # NOTE: This (req.finished() or req.is_retracted) should only happen when overlap scheduling is enabled.
                # And all the over-allocated tokens will be freed in `release_kv_cache`.
                continue

            if is_spec_v1:
                self._mamba_prefix_cache_update(req, batch, result, i)
                req.time_stats.set_last_decode_finish_time()
                self._handle_finished_req(req, i, logits_output)
                if req.return_hidden_states and logits_output.hidden_states is not None:
                    req.hidden_states.append(
                        logits_output.hidden_states[i].cpu().clone().tolist()
                    )
                if req.grammar is not None:
                    req.grammar.finished = req.finished()
                continue

            # Non-spec and V2: full post-processing
            next_token_id = next_token_ids[i]
            new_accepted_len = 1
            if batch.spec_algorithm.is_none():
                req.output_ids.append(next_token_id)
            else:
                req.output_ids.extend(next_token_id)
                new_accepted_len = len(next_token_id)

            self._maybe_update_reasoning_tokens(req, next_token_id)

            # Update Mamba last track seqlen
            self._mamba_prefix_cache_update(req, batch, result, i)
            req.time_stats.set_last_decode_finish_time()
            req.check_finished(new_accepted_len)

            self._handle_finished_req(req, i, logits_output)

            if req.return_logprob:
                # Spec v1 handles logprobs inside its own worker.
                # Normalize: non-spec has 1 token, spec v2 has multiple.
                if batch.is_spec_v2:
                    accepted_logprobs = next_token_logprobs[i]
                    accepted_ids = next_token_id
                    max_accept = len(accepted_logprobs)
                else:
                    accepted_logprobs = [next_token_logprobs[i]]
                    accepted_ids = [next_token_id]
                    max_accept = 1

                for j, tok_id in enumerate(accepted_ids):
                    req.output_token_logprobs_val.append(accepted_logprobs[j])
                    req.output_token_logprobs_idx.append(tok_id)
                    if req.top_logprobs_num > 0:
                        flat_idx = i * max_accept + j
                        req.output_top_logprobs_val.append(
                            logits_output.next_token_top_logprobs_val[flat_idx]
                        )
                        req.output_top_logprobs_idx.append(
                            logits_output.next_token_top_logprobs_idx[flat_idx]
                        )
                    if req.token_ids_logprob is not None:
                        flat_idx = i * max_accept + j
                        req.output_token_ids_logprobs_val.append(
                            logits_output.next_token_token_ids_logprobs_val[flat_idx]
                        )
                        req.output_token_ids_logprobs_idx.append(
                            logits_output.next_token_token_ids_logprobs_idx[flat_idx]
                        )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    if batch.spec_algorithm.is_none():
                        # Normal decode: single token
                        req.grammar.accept_token(next_token_id)
                    elif batch.is_spec_v2:
                        # Speculative decode: next_token_id is a list of accepted tokens
                        for token_id in next_token_id:
                            req.grammar.accept_token(token_id)
                except ValueError as e:
                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    logger.error(
                        f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                    )
                    self.abort_request(AbortReq(rid=req.rid))
                req.grammar.finished = req.finished()

        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        self.report_decode_stats(
            can_run_cuda_graph,
            running_batch=batch,
            num_accepted_tokens=result.num_accepted_tokens,
        )

    def _handle_finished_req(
        self: Scheduler, req: Req, i: int, logits_output: LogitsProcessorOutput
    ):
        if (
            self.server_args.disaggregation_decode_enable_offload_kvcache
            and not req.finished()
        ):
            self.decode_offload_manager.offload_kv_cache(req)

        if req.finished():
            # delete feature to save memory
            if req.multimodal_inputs is not None and req.session is None:
                req.multimodal_inputs.release_features()
            self.maybe_collect_routed_experts(req)

            if self.server_args.disaggregation_decode_enable_offload_kvcache:
                # Asynchronously offload KV cache; release_kv_cache will be called after Device->Host transfer completes
                if not self.decode_offload_manager.offload_kv_cache(req):
                    self.decode_offload_manager.finalize_release_on_finish(req)
            else:
                if self.enable_hisparse:
                    self.hisparse_coordinator.request_finished(req)
                self._release_c2kv_pins(req)
                release_kv_cache(
                    req,
                    self.tree_cache,
                    is_insert=req.c2kv_rounds is None,
                )

            req.time_stats.set_completion_time()

        self.maybe_collect_customized_info(i, req, logits_output)

    def _maybe_update_reasoning_tokens(
        self: Scheduler, req: Req, next_token_id: Union[int, List[int]]
    ):
        think_end_id = self.model_config.think_end_id
        if req.require_reasoning and think_end_id is not None:
            req.update_reasoning_tokens(next_token_id, think_end_id)

    def _mamba_prefix_cache_update(
        self: Scheduler,
        req: Req,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        i: int,
    ) -> None:
        seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1
        if req.mamba_ping_pong_track_buffer is not None:
            mamba_track_interval = get_global_server_args().mamba_track_interval
            if batch.spec_algorithm.is_none() and seq_len % mamba_track_interval == 0:
                # for non-spec decode, we update mamba_last_track_seqlen at the end of each track interval
                req.mamba_next_track_idx = (
                    batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                        req.mamba_next_track_idx
                    )
                )
                req.mamba_last_track_seqlen = seq_len
            elif (
                not batch.spec_algorithm.is_none()
                and result.accept_length_per_req_cpu is not None
            ):
                # for spec decode, update mamba_last_track_seqlen if this iteration crosses a track interval
                actual_seq_len = req.seqlen - 1
                if (
                    actual_seq_len // mamba_track_interval
                    != (actual_seq_len - result.accept_length_per_req_cpu[i])
                    // mamba_track_interval
                ):
                    req.mamba_next_track_idx = (
                        batch.req_to_token_pool.get_mamba_ping_pong_other_idx(
                            req.mamba_next_track_idx
                        )
                    )
                    req.mamba_last_track_seqlen = (
                        actual_seq_len // mamba_track_interval * mamba_track_interval
                    )

    def _process_input_token_logprobs(
        self: Scheduler, req: Req, input_token_logprobs: List
    ) -> None:
        """Process input token logprobs values and indices."""
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Process logprob values - handle multi-item scoring vs regular requests
        if is_multi_item_scoring:
            # Multi-item scoring: use all logprobs as-is
            req.input_token_logprobs_val = input_token_logprobs
        else:
            # Regular request: add None at start, remove last (sampling token)
            req.input_token_logprobs_val = [None] + input_token_logprobs[:-1]

        # Process logprob indices based on scoring type
        if is_multi_item_scoring:
            # Multi-item scoring: only include delimiter token positions
            relevant_tokens = req.origin_input_ids[req.logprob_start_len :]
            input_token_logprobs_idx = [
                token_id
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            ]
        else:
            # Regular request: include all tokens from logprob_start_len onwards
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]

        # Clip padded hash values from image tokens to prevent detokenization errors
        req.input_token_logprobs_idx = [
            x if x < self.model_config.vocab_size - 1 else 0
            for x in input_token_logprobs_idx
        ]

    def _process_input_top_logprobs(self: Scheduler, req: Req) -> None:
        """Process input top logprobs."""
        if req.top_logprobs_num <= 0:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_top_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_top_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Extend arrays with temp values
        for val, idx in zip(
            req.temp_input_top_logprobs_val,
            req.temp_input_top_logprobs_idx,
            strict=True,
        ):
            req.input_top_logprobs_val.extend(val)
            req.input_top_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_top_logprobs_val.pop()
            req.input_top_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_top_logprobs_idx = None
        req.temp_input_top_logprobs_val = None

    def _process_input_token_ids_logprobs(self: Scheduler, req: Req) -> None:
        """Process input token IDs logprobs."""
        if req.token_ids_logprob is None:
            return

        is_multi_item_scoring = self._is_multi_item_scoring(req)

        # Initialize arrays - multi-item scoring starts empty, others start with None
        req.input_token_ids_logprobs_val = [] if is_multi_item_scoring else [None]
        req.input_token_ids_logprobs_idx = [] if is_multi_item_scoring else [None]

        # Process temp values - convert tensors to lists and extend arrays
        for val, idx in zip(
            req.temp_input_token_ids_logprobs_val,
            req.temp_input_token_ids_logprobs_idx,
            strict=True,
        ):
            val_list = val.tolist() if isinstance(val, torch.Tensor) else val
            req.input_token_ids_logprobs_val.extend(
                val_list if isinstance(val_list, list) else [val_list]
            )
            req.input_token_ids_logprobs_idx.extend(idx)

        # Remove last token (sampling token) for non multi-item scoring requests
        if not is_multi_item_scoring:
            req.input_token_ids_logprobs_val.pop()
            req.input_token_ids_logprobs_idx.pop()

        # Clean up temp storage
        req.temp_input_token_ids_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None

    def _calculate_relevant_tokens_len(self: Scheduler, req: Req) -> int:
        """Calculate the expected length of logprob arrays based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions from logprob_start_len onwards have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)
        relevant_tokens = req.origin_input_ids[req.logprob_start_len :]

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens from logprob_start_len onwards
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens from logprob_start_len onwards
            return len(relevant_tokens)

    def _calculate_num_input_logprobs(
        self: Scheduler, req: Req, extend_input_len: int, extend_logprob_start_len: int
    ) -> int:
        """Calculate the number of input logprobs based on whether multi-item scoring is enabled.

        For multi-item scoring, only delimiter positions have logprobs.
        For regular requests, all positions in the range have logprobs.
        """
        is_multi_item_scoring = self._is_multi_item_scoring(req)

        if is_multi_item_scoring:
            # Multi-item scoring: count delimiter tokens in the relevant portion
            relevant_tokens = req.origin_input_ids[
                extend_logprob_start_len:extend_input_len
            ]
            return sum(
                1
                for token_id in relevant_tokens
                if token_id == self.server_args.multi_item_scoring_delimiter
            )
        else:
            # Regular request: all tokens in the range
            return extend_input_len - extend_logprob_start_len

    def _is_multi_item_scoring(self: Scheduler, req: Req) -> bool:
        """Check if request uses multi-item scoring.

        Multi-item scoring applies to prefill-only requests when a delimiter
        token is configured. In this mode, only positions containing the
        delimiter token receive logprobs.
        """
        return req.is_prefill_only and self.server_args.multi_item_scoring_delimiter

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Process all input logprob types using helper functions
            self._process_input_token_logprobs(req, input_token_logprobs)
            self._process_input_top_logprobs(req)

            self._process_input_token_ids_logprobs(req)

            if req.return_logprob:
                relevant_tokens_len = self._calculate_relevant_tokens_len(req)
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        if output.next_token_logprobs is not None:
            req.output_token_logprobs_val.append(output.next_token_logprobs[i])
            req.output_token_logprobs_idx.append(next_token_ids[i])

        # Only add input logprobs if there are input tokens to process
        # Note: For prefill-only requests with default logprob_start_len, this will be 0,
        # meaning we only compute output logprobs (which is the intended behavior)
        if num_input_logprobs > 0:
            self.add_input_logprob_return_values(
                i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
            )
        else:
            self._initialize_empty_logprob_containers(req)

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if (
            req.token_ids_logprob is not None
            and output.next_token_token_ids_logprobs_val is not None
        ):
            # Convert GPU tensor to list if needed
            logprobs_val = output.next_token_token_ids_logprobs_val[i]
            if isinstance(logprobs_val, torch.Tensor):
                logprobs_val = logprobs_val.tolist()
            req.output_token_ids_logprobs_val.append(logprobs_val)
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _initialize_empty_logprob_containers(self: Scheduler, req: Req) -> None:
        """
        Initialize logprob fields to empty lists if unset.

        This is needed for prefill-only requests where the normal initialization
        flow might be bypassed, but downstream code expects these fields to be lists.
        """
        if req.input_token_logprobs_val is None:
            req.input_token_logprobs_val = []
        if req.input_token_logprobs_idx is None:
            req.input_token_logprobs_idx = []
        if req.input_top_logprobs_val is None:
            req.input_top_logprobs_val = []
        if req.input_top_logprobs_idx is None:
            req.input_top_logprobs_idx = []
        if req.input_token_ids_logprobs_val is None:
            req.input_token_ids_logprobs_val = []
        if req.input_token_ids_logprobs_idx is None:
            req.input_token_ids_logprobs_idx = []

    def stream_output(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

        if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
            self._trigger_crash_for_tests(
                envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get()
            )

    def _trigger_crash_for_tests(self: Scheduler, crash_threshold: int):
        # Crash trigger: crash after stream_output is called N times
        # This is used for testing purposes.
        if not hasattr(self, "_test_stream_output_count"):
            self._test_stream_output_count = 0
        self._test_stream_output_count += 1
        if self._test_stream_output_count >= crash_threshold:
            raise RuntimeError(
                f"Test crash after stream_output called {self._test_stream_output_count} times"
            )

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
        is_idle_batch: bool = False,
    ):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        reasoning_tokens = []
        completion_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        kv_runtime_stats = []
        kv_memory_reports = []
        spec_verify_ct = []
        spec_accepted_tokens = []
        spec_acceptance_histogram = []
        retraction_counts = []
        output_hidden_states = None
        load = self.get_load()
        routed_experts = None
        customized_info = {}

        time_stats = []

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                if req.finished_len is None:
                    req.finished_len = len(req.output_ids)
                should_output = True
            else:
                if req.stream:
                    stream_interval = (
                        req.sampling_params.stream_interval or self.stream_interval
                    )

                    # origin stream_interval logic
                    should_output = (
                        len(req.output_ids) % stream_interval == 1
                        if stream_interval > 1
                        else len(req.output_ids) % stream_interval == 0
                    )

                    if should_output:
                        # check_match_stop_str_prefix if  tail_str's suffix match stop_str prefix
                        should_output &= not req.check_match_stop_str_prefix()
                else:
                    should_output = (
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                    )

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                # Exclude the tokens after stop condition
                output_ids_ = req.output_ids_through_stop

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                output_ids.append(output_ids_[send_token_offset:])
                req.send_token_offset = len(output_ids_)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                reasoning_tokens.append(req.reasoning_tokens)
                completion_tokens.append(len(output_ids_))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))
                kv_runtime_stats.append(self._get_kv_runtime_stats(req))
                kv_memory_report = self._get_kv_memory_report(req)
                if getattr(req, "history_kv_eviction_result", None) is not None:
                    logger.info(
                        "KV-memory output snapshot rid=%s report_present=%s",
                        req.rid,
                        kv_memory_report is not None,
                    )
                kv_memory_reports.append(kv_memory_report)

                retraction_counts.append(req.retraction_count)

                time_stats.append(req.time_stats)

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)
                    spec_accepted_tokens.append(req.spec_accepted_tokens)
                    spec_acceptance_histogram.append(req.spec_acceptance_histogram)

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                        # Only send when input logprobs have been computed (after prefill)
                        and req.input_token_logprobs_val is not None
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        logprob_end = max(len(output_ids_), 1)
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:logprob_end
                            ]
                        )
                        req.send_output_token_logprobs_offset = logprob_end
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)
                if req.return_routed_experts:
                    if routed_experts is None:
                        routed_experts = []
                    routed_experts.append(req.routed_experts)

                if req.customized_info is not None:
                    for k, v in req.customized_info.items():
                        if k not in customized_info:
                            customized_info[k] = []
                        customized_info[k].append(
                            v[send_token_offset : len(output_ids_)]
                        )

            if (
                req.finished()
                and self.attn_tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        dp_ranks = [self.dp_rank] * len(rids) if rids else None

        # Send to detokenizer
        if reqs or is_idle_batch:
            self.send_to_detokenizer.send_output(
                BatchTokenIDOutput(
                    rids=rids,
                    http_worker_ipcs=http_worker_ipcs,
                    spec_verify_ct=spec_verify_ct,
                    spec_accepted_tokens=spec_accepted_tokens,
                    spec_acceptance_histogram=spec_acceptance_histogram,
                    time_stats=time_stats,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids_list,
                    read_offsets=read_offsets,
                    output_ids=output_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    no_stop_trim=no_stop_trim,
                    prompt_tokens=prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cached_tokens_details=cached_tokens_details,
                    kv_runtime_stats=kv_runtime_stats,
                    kv_memory_reports=kv_memory_reports,
                    input_token_logprobs_val=input_token_logprobs_val,
                    input_token_logprobs_idx=input_token_logprobs_idx,
                    output_token_logprobs_val=output_token_logprobs_val,
                    output_token_logprobs_idx=output_token_logprobs_idx,
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                    input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val=output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
                    output_token_entropy_val=None,
                    output_hidden_states=output_hidden_states,
                    routed_experts=routed_experts,
                    customized_info=customized_info,
                    placeholder_tokens_idx=None,
                    placeholder_tokens_val=None,
                    retraction_counts=retraction_counts,
                    load=load,
                    dp_ranks=dp_ranks,
                )
            )

    def stream_output_embedding(self: Scheduler, reqs: List[Req]):
        rids = []
        http_worker_ipcs = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        cached_tokens_details = []  # Detailed breakdown by cache source
        time_stats = []
        retraction_counts = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                http_worker_ipcs.append(req.http_worker_ipc)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)

                # Collect detailed cache breakdown if available
                cached_tokens_details.append(self._get_cached_tokens_details(req))
                time_stats.append(req.time_stats)
                retraction_counts.append(req.retraction_count)
        self.send_to_detokenizer.send_output(
            BatchEmbeddingOutput(
                rids=rids,
                http_worker_ipcs=http_worker_ipcs,
                time_stats=time_stats,
                finished_reasons=finished_reasons,
                embeddings=embeddings,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached_tokens,
                cached_tokens_details=cached_tokens_details,
                placeholder_tokens_idx=None,
                placeholder_tokens_val=None,
                retraction_counts=retraction_counts,
            )
        )
