# Adapted from qwen2.py
import logging
import os
from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.mem_cache.gist_utils import (
    C2KV_KERNEL_OPTIONS,
    GistConfig,
    get_apply_gist_residual_func,
    get_prepare_gist_input_func,
    prepare_pic_input,
)
from sglang.srt.mem_cache.history_kv_selection import (
    HEADWISE_HISTORY_KV_METHODS,
    attention_scores_by_kv_head,
    deduplicated_recovery_indices,
    dense_headwise_recovery_indices,
    gather_paired_kv,
    repair_score_query_start,
    require_rotated_headwise_storage,
    select_h2o_prefill_indices,
    select_snapkv_indices,
    select_streamingllm_indices,
    summarize_headwise_indices,
)
from sglang.srt.mem_cache.history_kv_reference import (
    ReferenceLayerKV,
    reference_sdpa,
)
from sglang.srt.mem_cache.repair_tool_selection import (
    SPARSE_REPAIR_METHODS,
    select_sparse_repair_indices,
    validate_sparse_repair_partition,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.observability import paper_telemetry
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.mem_cache.cacheblend import CacheBlendConfig, ChunkKVCache
from sglang.srt.mem_cache.cacheblend import blend as cacheblend_blend
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_hip, is_npu

Qwen3Config = None

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

_has_fused_qk_norm_mrope = False
if _use_aiter:
    try:
        from aiter import fused_qk_norm_mrope_3d_cache_pts_quant_shuffle

        _has_fused_qk_norm_mrope = True
        logger.info("aiter fused_qk_norm_mrope_3d kernel available")
    except ImportError:
        pass

if _is_npu:
    import torch_npu

    try:
        from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import (
            split_qkv_rmsnorm_rope,
        )
    except ImportError:
        # older triton-ascend without language.extra.cann: fall back to the
        # native path (split + qk_norm + rope) for decode as well
        # (compat 27f21a588, ported onto 22fbf3146)
        split_qkv_rmsnorm_rope = None

    from sglang.srt.hardware_backend.npu.cmo import get_cmo_stream, wait_cmo_stream


def _npu_fusion_attention_output(
    output: Any, expected_shape: torch.Size
) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if not isinstance(output, tuple) or not output:
        raise RuntimeError(f"Unexpected npu_fusion_attention output type: {type(output)!r}")

    candidates = [
        item for item in output if isinstance(item, torch.Tensor) and item.dim() == 4
    ]
    if not candidates:
        raise RuntimeError("npu_fusion_attention returned no 4-D attention output tensor.")
    for tensor in candidates:
        if tensor.shape == expected_shape:
            return tensor
    return candidates[0]


def _requires_reference_runtime_qkv(forward_batch: ForwardBatch) -> bool:
    """Keep explicit Q/K/V available for reference selection and attention."""

    return any(
        item is not None
        for name in ("history_kv_reference_configs", "history_kv_reference_states")
        for item in (getattr(forward_batch, name, None) or [])
    )


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        pic_enabled: bool = False,
        pic_param: str = "qkv",
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        tool_gist_uses_served_t0: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.pic_enabled = pic_enabled
        self.pic_param = pic_param.lower()
        if not self.pic_param or set(self.pic_param) - set("qkv"):
            raise ValueError(
                "pic_param must be a non-empty combination of q, k, and v; "
                f"got {pic_param!r}."
            )

        # Which of q/k/v ordinary tokens switch to the gist projections when the
        # per-token C2KV mask selects them (empty = never = base). Derived from
        # --c2kv-gist-param ALONE: --c2kv-query-proj is only the per-request
        # DEFAULT of the mask (D1), so a request that explicitly asks for the
        # gist projection must still find the parts wired up here.
        self.c2kv_query_proj_parts = frozenset()
        if get_global_server_args().enable_c2kv:
            c2kv_proj_name = "residual_qkv_proj" if pic_enabled else "gist_qkv_proj"
            c2kv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=attention_bias,
                # C1000 was trained with FP32 gist parameters.  Keep the
                # checkpoint values in FP32 and cast only for the base model's
                # mixed-precision compute, matching the native HF runtime.
                params_dtype=(torch.float32 if not pic_enabled else None),
                quant_config=(None if not pic_enabled else quant_config),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix(c2kv_proj_name, prefix),
            )
            setattr(self, c2kv_proj_name, c2kv_proj)
            # Tool extraction uses the served T0 projection directly when both
            # names identify that same checkpoint.  A distinct tool checkpoint
            # keeps its own FP32 projection set.
            self.c2kv_tool_gist_enabled = False
            if not pic_enabled and getattr(
                get_global_server_args(), "c2kv_tool_gist_weights", None
            ):
                if tool_gist_uses_served_t0:
                    self.tool_gist_qkv_proj = c2kv_proj
                else:
                    self.tool_gist_qkv_proj = QKVParallelLinear(
                        hidden_size,
                        self.head_dim,
                        self.total_num_heads,
                        self.total_num_kv_heads,
                        bias=attention_bias,
                        params_dtype=torch.float32,
                        quant_config=None,
                        tp_rank=attn_tp_rank,
                        tp_size=attn_tp_size,
                        prefix=add_prefix("tool_gist_qkv_proj", prefix),
                    )
                self.c2kv_tool_gist_enabled = True
            if not pic_enabled:
                # PIC/residual_qkv_proj is excluded by construction: there is no
                # gist_qkv_proj to switch to.
                _gist_param = str(
                    getattr(get_global_server_args(), "c2kv_gist_param", "qkv") or ""
                ).lower()
                self.c2kv_query_proj_parts = frozenset(
                    part for part in "qkv" if part in _gist_param
                )
            if pic_enabled:
                # Loading a base Qwen3 checkpoint with PIC enabled must initially
                # preserve its QKV projections exactly.
                with torch.no_grad():
                    if hasattr(c2kv_proj, "weight"):
                        c2kv_proj.weight.zero_()
                    if c2kv_proj.bias is not None:
                        c2kv_proj.bias.zero_()
            if pic_enabled:
                try:
                    from flash_attn import flash_attn_func
                except ImportError as e:
                    raise ImportError(
                        "Full-length PIC extraction requires FlashAttention 2."
                    ) from e
                self.flash_attention_2 = flash_attn_func
            else:
                self.flex_attention = torch.compile(
                    partial(
                        flex_attention, kernel_options=C2KV_KERNEL_OPTIONS
                    ),
                    dynamic=True,
                )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream

        self.use_fused_qk_norm_mrope = (
            _has_fused_qk_norm_mrope
            and isinstance(self.rotary_emb, MRotaryEmbedding)
            and getattr(self.rotary_emb, "mrope_section", None) is not None
        )
        if self.use_fused_qk_norm_mrope:
            # Scale tensors MUST stay on CPU: the C++ kernel uses .item<float>()
            # which triggers hipMemcpy D2H + sync on CUDA tensors, breaking graph capture.
            # Explicit device='cpu' is required because SGLang constructs models inside
            # a `with torch.device('cuda'):` context that changes the default device.
            self._fused_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
            self._fused_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")

    def _c2kv_project_qkv(self, hidden_states, forward_batch):
        """QKV projection honouring --c2kv-query-proj.

        The paper/reference lowercase-qkv regime leaves ordinary query tokens
        on the base projections. The post-2026-08-09 local fork can instead use
        gist_{q,k,v}_proj for the main forward; that extension is selected
        explicitly with ``--c2kv-query-proj gist``. The system prefix is
        prefilled separately with base projections.
        `forward_batch.c2kv_use_gist_projection` is the
        per-token mask built in ForwardBatch from the request's EFFECTIVE mode
        (explicit message-level ``c2kv_use_gist_projection`` if the client sent
        one, otherwise ``ServerArgs.c2kv_query_proj``) gated by the absolute
        position of the request's first gist segment; those rows take the gist
        projection for the parts listed in `c2kv_query_proj_parts` (derived from
        --c2kv-gist-param). Everything else, including repair KV extraction
        (`generate_raw_repair_kv`, forward_batch=None), stays base.
        """
        qkv, _ = self.qkv_proj(hidden_states)
        parts = self.c2kv_query_proj_parts
        if not parts or not hasattr(self, "gist_qkv_proj"):
            return qkv
        mask = (
            getattr(forward_batch, "c2kv_use_gist_projection", None)
            if forward_batch is not None
            else None
        )
        if mask is None:
            return qkv
        if mask.ndim != 1 or mask.shape[0] != qkv.shape[0]:
            raise RuntimeError(
                "c2kv_use_gist_projection mask shape mismatch: "
                f"{tuple(mask.shape)} != {(qkv.shape[0],)}"
            )
        qkv_gist, _ = self._c2kv_project_gist_qkv(hidden_states)
        sizes = [self.q_size, self.kv_size, self.kv_size]
        base_parts = qkv.split(sizes, dim=-1)
        gist_parts = qkv_gist.split(sizes, dim=-1)
        sel = mask.to(qkv.device).view(-1, 1)
        merged = [
            torch.where(sel, gist_t, base_t) if name in parts else base_t
            for name, base_t, gist_t in zip("qkv", base_parts, gist_parts)
        ]
        return torch.cat(merged, dim=-1)

    def _c2kv_gist_projection(self, projection_set: str = "history"):
        """The fused gist QKV linear of one projection set.

        ``history`` is the served checkpoint's own set (``gist_qkv_proj``);
        ``tool`` is the optional --c2kv-tool-gist-weights set.  Requesting a
        set that was not loaded is a hard error, never a silent fallback to the
        other set (the two encoders are trained on different corpora).
        """
        if projection_set == "history":
            return self.gist_qkv_proj
        if projection_set == "tool":
            projection = getattr(self, "tool_gist_qkv_proj", None)
            if projection is None:
                raise RuntimeError(
                    "C2KV_TOOL_GIST_UNAVAILABLE: projection_set='tool' needs "
                    "a server started with --c2kv-tool-gist-weights"
                )
            return projection
        raise ValueError(f"Unknown C2KV projection set {projection_set!r}")

    def _c2kv_project_gist_qkv(self, hidden_states, projection_set: str = "history"):
        """Apply FP32-stored gist weights in the base compute dtype."""

        projection = self._c2kv_gist_projection(projection_set)
        if projection.weight.dtype == hidden_states.dtype:
            return projection(hidden_states)
        with torch.autocast(
            device_type=hidden_states.device.type,
            dtype=hidden_states.dtype,
        ):
            return projection(hidden_states)

    def forward_prepare_native(self, positions, hidden_states, forward_batch=None):
        qkv = self._c2kv_project_qkv(hidden_states, forward_batch)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def _collect_history_kv_eviction_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        configs = getattr(forward_batch, "c2kv_history_kv_eviction_configs", None)
        if not configs or not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            return
        if (
            forward_batch.extend_seq_lens_cpu is None
            or forward_batch.extend_prefix_lens_cpu is None
            or forward_batch.req_pool_indices is None
        ):
            return
        if q is None or k is None or positions is None:
            return

        score_store = getattr(forward_batch, "c2kv_history_kv_selection_scores", None)
        if score_store is None:
            score_store = {}
            forward_batch.c2kv_history_kv_selection_scores = score_store

        offset = 0
        flat_positions = positions.reshape(-1)
        for batch_idx, config in enumerate(configs):
            extend_len = int(forward_batch.extend_seq_lens_cpu[batch_idx])
            prefix_len = int(forward_batch.extend_prefix_lens_cpu[batch_idx])
            token_start = offset
            token_end = offset + extend_len
            offset = token_end
            if not isinstance(config, dict):
                continue
            history_start = int(config.get("history_start") or 0)
            history_end = int(config.get("history_end") or 0)
            available_end = prefix_len + extend_len
            tool_kv_eviction = bool(config.get("tool_kv_eviction"))
            reference_state = None
            states = getattr(
                forward_batch, "history_kv_reference_states", None
            )
            if (
                states
                and batch_idx < len(states)
                and states[batch_idx] is not None
            ):
                reference_state = states[batch_idx].layer(
                    self.attn.layer_id
                )
            reference_len = 0
            if reference_state is not None:
                reference_state.validate()
                reference_len = int(reference_state.key.shape[1])
            if not (0 <= history_start <= history_end) or (
                not tool_kv_eviction and history_end > available_end
            ) or (
                history_start == history_end and reference_len == 0
            ):
                continue

            method = str(config.get("method") or "").strip().lower()
            if method in {"", "streamingllm"}:
                continue
            recent_window = max(1, int(config.get("history_kv_recent_window") or 64))
            if tool_kv_eviction:
                q_start = max(0, int(config["selection_query_start"]) - prefix_len)
                q_end = min(extend_len, int(config["selection_query_end"]) - prefix_len)
            elif prefix_len > 0 and history_end <= prefix_len:
                q_end = extend_len
                q_start = max(0, q_end - recent_window)
            else:
                q_end = min(extend_len, max(1, history_end - prefix_len))
                q_start = max(0, q_end - recent_window)
            if q_start >= q_end:
                continue

            q_req = q[token_start:token_end].view(
                extend_len, self.num_heads, self.head_dim
            ).transpose(0, 1).contiguous()
            k_req = k[token_start:token_end].view(
                extend_len, self.num_kv_heads, self.head_dim
            )
            # Keys of an evicted token do not exist in this candidate set.
            # Read only the resident request-table prefix, never repair_extract
            # or full-history text. Cached K is already at its original RoPE.
            if prefix_len:
                req_pool_idx = int(forward_batch.req_pool_indices[batch_idx].item())
                slots = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :prefix_len].long()
                key_buffer = forward_batch.token_to_kv_pool._get_key_buffer(self.attn.layer_id)
                # Request-table values are physical TOKEN slots, not page IDs.
                # Ascend stores [pages, page_size, Hkv, D] (FIA uses
                # [tokens, 1, Hkv, D]); normalize before gathering. Indexing
                # the page axis with token slots reads whole pages and can OOB.
                if key_buffer.ndim not in (3, 4) or tuple(key_buffer.shape[-2:]) != (
                    self.num_kv_heads, self.head_dim
                ):
                    raise RuntimeError("HISTORY_KV_UNSUPPORTED_KEY_BUFFER_LAYOUT")
                cached = key_buffer.reshape(-1, self.num_kv_heads, self.head_dim)[slots]
                k_req = torch.cat([cached.to(k_req.dtype), k_req], dim=0)
            # Keep this construction local: lifecycle unit tests extract this
            # method in isolation, and production needs the same canonical
            # ledger fallback as the reference attention helper.
            normal_seq_len = prefix_len + extend_len
            ledgers = getattr(
                forward_batch, "history_kv_resident_positions", None
            )
            ledger = (
                list(ledgers[batch_idx])
                if ledgers and batch_idx < len(ledgers)
                else list(config.get("resident_logical_positions") or [])
            )
            if len(ledger) >= normal_seq_len:
                normal_positions = torch.tensor(
                    ledger[:normal_seq_len],
                    dtype=torch.long,
                    device=flat_positions.device,
                )
            else:
                query_position_list = [
                    int(item)
                    for item in flat_positions[token_start:token_end].tolist()
                ]
                known = ledger[:prefix_len]
                missing = prefix_len - len(known)
                if missing:
                    start = (
                        query_position_list[0] - missing
                        if query_position_list
                        else (known[-1] + 1 if known else 0)
                    )
                    known.extend(range(start, start + missing))
                normal_positions = torch.tensor(
                    known + query_position_list,
                    dtype=torch.long,
                    device=flat_positions.device,
                )
            k_req = k_req.transpose(0, 1).contiguous()
            groups = self.num_heads // self.num_kv_heads

            # The query may follow an already-cached history boundary. Include
            # the resident current prefix and query's own key in the softmax;
            # only slice to history candidates after normalization. Otherwise
            # heads attending to current content get overstated history scores.
            key_end = prefix_len + q_end
            total_key_len = reference_len + key_end
            key_positions = normal_positions[:key_end].to(k_req.device)
            if reference_state is not None:
                k_pos = torch.cat(
                    [
                        reference_state.positions.to(k_req.device),
                        key_positions.view(1, -1).expand(self.num_kv_heads, -1),
                    ],
                    dim=1,
                ).view(self.num_kv_heads, 1, 1, -1)
            else:
                k_pos = key_positions.view(1, 1, 1, -1).expand(
                    self.num_kv_heads, -1, -1, -1
                )
            headwise_probs = torch.zeros(
                self.num_kv_heads,
                total_key_len,
                dtype=torch.float32,
                device=k_req.device,
            )
            # A fixed query count still allocates hundreds of MiB of logits
            # once AppWorld's persistent history grows past 100k keys. Bound
            # float logits by bytes, including grouped query heads, while
            # preserving the same softmax denominator and score reduction.
            score_query_chunk = max(
                1,
                min(
                    64,
                    (16 * 1024 * 1024)
                    // max(1, self.num_heads * total_key_len * 4),
                ),
            )
            # Score one KV head at a time. Keep reference/current keys separate
            # until their small logits are joined for the full-key softmax.
            # This avoids a large grouped key copy and full-history float cast.
            grouped_query = q_req.reshape(
                self.num_kv_heads, groups, extend_len, self.head_dim
            )
            for kv_head in range(self.num_kv_heads):
                current_key = k_req[kv_head, :key_end, :].transpose(0, 1).float()
                reference_key = (
                    reference_state.key[kv_head]
                    .to(k_req.dtype)
                    .transpose(0, 1)
                    .float()
                    if reference_state is not None
                    else None
                )
                for query_left in range(q_start, q_end, score_query_chunk):
                    query_right = min(q_end, query_left + score_query_chunk)
                    query_count = query_right - query_left
                    score_query = grouped_query[
                        kv_head, :, query_left:query_right, :
                    ].reshape(1, groups * query_count, self.head_dim).float()
                    logits = torch.bmm(score_query, current_key.unsqueeze(0))
                    if reference_key is not None:
                        reference_logits = torch.bmm(
                            score_query, reference_key.unsqueeze(0)
                        )
                        logits = torch.cat([reference_logits, logits], dim=-1)
                    logits = logits.view(groups, query_count, -1) * self.scaling
                    q_pos = flat_positions[
                        token_start + query_left : token_start + query_right
                    ].to(logits.device).view(1, -1, 1)
                    logits = logits.masked_fill(
                        k_pos[kv_head] > q_pos, float("-inf")
                    )
                    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
                    headwise_probs[kv_head] += probs.sum(dim=(0, 1))
            candidate_end = min(history_end, key_end)
            selected_probs = headwise_probs[
                :, reference_len + history_start : reference_len + candidate_end
            ]
            if tool_kv_eviction and candidate_end < history_end:
                padded = torch.zeros(
                    self.num_kv_heads,
                    history_end - history_start,
                    device=selected_probs.device,
                    dtype=selected_probs.dtype,
                )
                padded[:, : candidate_end - history_start] = selected_probs
                selected_probs = padded
            layer_score = selected_probs.sum(dim=0)
            headwise_layer_score = torch.cat(
                [
                    headwise_probs[:, :reference_len],
                    selected_probs,
                ],
                dim=1,
            )

            req_pool_idx = int(forward_batch.req_pool_indices[batch_idx].item())
            entry = score_store.setdefault(
                req_pool_idx,
                {
                    "method": method,
                    "history_start": history_start,
                    "history_end": history_end,
                    "history_len": history_end - history_start,
                    "query_tokens": q_end - q_start,
                    "selection_query_start": config.get("selection_query_start"),
                    "selection_query_end": config.get("selection_query_end"),
                    "selection_query_phase": config.get("selection_query_phase"),
                    "layers": [],
                    "headwise_layers": [],
                    "layer_ids": [],
                },
            )
            entry["layers"].append(layer_score.detach().cpu())
            entry["headwise_layers"].append(
                headwise_layer_score.detach().cpu()
            )
            entry["layer_ids"].append(int(self.attn.layer_id))

    def _capture_history_kv_runtime_queries(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Persist method query observations, including first-turn decode."""

        configs = getattr(forward_batch, "history_kv_reference_configs", None)
        states = getattr(forward_batch, "history_kv_runtime_states", None)
        if not configs or not states or q is None or positions is None:
            return
        if forward_batch.forward_mode.is_decode():
            query_lens = [1] * int(forward_batch.batch_size)
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            if forward_batch.extend_seq_lens_cpu is None:
                return
            query_lens = [int(item) for item in forward_batch.extend_seq_lens_cpu]
        else:
            return
        query = q.view(-1, self.num_heads, self.head_dim)
        flat_positions = positions.reshape(-1)
        flat_token_ids = forward_batch.input_ids.reshape(-1)
        offset = 0
        for batch_idx, query_len in enumerate(query_lens):
            token_query = query[offset : offset + query_len]
            token_positions = flat_positions[offset : offset + query_len].to(
                dtype=torch.long
            )
            token_ids = flat_token_ids[offset : offset + query_len]
            offset += query_len
            config = configs[batch_idx] if batch_idx < len(configs) else None
            state = states[batch_idx] if batch_idx < len(states) else None
            if not isinstance(config, dict) or state is None:
                continue
            method = str(config.get("method") or "").lower()
            if method == "commitkv":
                from sglang.srt.mem_cache.history_kv_reference import (
                    CommitKVServingState,
                )

                if not isinstance(state, CommitKVServingState):
                    raise RuntimeError("COMMITKV_RUNTIME_STATE_TYPE_MISMATCH")
                if self.attn.layer_id != state.policy.config.measurement_layer_id:
                    continue
                state.configure_events(config.get("event_token_spans") or [])
                if not forward_batch.forward_mode.is_decode():
                    continue
                if k is None or v is None or query_len != 1:
                    raise RuntimeError("COMMITKV_DECODE_CAPTURE_REQUIRES_EXPLICIT_KV")
                seq_len = int(forward_batch.seq_lens[batch_idx].item())
                req_pool_idx = int(
                    forward_batch.req_pool_indices[batch_idx].item()
                )
                prefix_slots = forward_batch.req_to_token_pool.req_to_token[
                    req_pool_idx, : max(0, seq_len - 1)
                ].long()
                key_buffer, value_buffer = (
                    forward_batch.token_to_kv_pool.get_kv_buffer(
                        self.attn.layer_id
                    )
                )
                key_buffer = key_buffer.reshape(
                    -1, self.num_kv_heads, self.head_dim
                )
                value_buffer = value_buffer.reshape(
                    -1, self.num_kv_heads, self.head_dim
                )
                normal_key = torch.cat(
                    [
                        key_buffer[prefix_slots].to(token_query.dtype),
                        k[offset - query_len : offset].view(
                            query_len, self.num_kv_heads, self.head_dim
                        ),
                    ],
                    dim=0,
                )
                normal_value = torch.cat(
                    [
                        value_buffer[prefix_slots].to(token_query.dtype),
                        v[offset - query_len : offset].view(
                            query_len, self.num_kv_heads, self.head_dim
                        ),
                    ],
                    dim=0,
                )
                normal_positions = self._reference_normal_positions(
                    forward_batch,
                    batch_idx,
                    seq_len,
                    token_positions,
                )
                reference_layer = None
                reference_states = getattr(
                    forward_batch, "history_kv_reference_states", None
                ) or []
                if (
                    batch_idx < len(reference_states)
                    and reference_states[batch_idx] is not None
                ):
                    reference_layer = reference_states[batch_idx].layer(
                        self.attn.layer_id
                    )
                key = normal_key.transpose(0, 1)
                value = normal_value.transpose(0, 1)
                key_positions = normal_positions
                if reference_layer is not None:
                    reference_layer.validate()
                    if not torch.equal(
                        reference_layer.positions,
                        reference_layer.positions[:1].expand_as(
                            reference_layer.positions
                        ),
                    ):
                        raise RuntimeError(
                            "COMMITKV_REQUIRES_COMMON_HEADWISE_POSITIONS"
                        )
                    key = torch.cat([reference_layer.key, key], dim=1)
                    value = torch.cat([reference_layer.value, value], dim=1)
                    key_positions = torch.cat(
                        [reference_layer.positions[0], normal_positions], dim=0
                    )
                state.record_decode_window(
                    token_query,
                    token_positions,
                    key,
                    value,
                    key_positions,
                    scale=self.scaling,
                )
                continue
            if method != "agentkv":
                # CommitKV captures paired pre/post windows at explicit action
                # boundaries; ordinary prompt queries must not enter them.
                continue
            from sglang.srt.mem_cache.agentkv import (
                AGENTKV_STAGE_THINK,
                AgentKVQueryRing,
                agentkv_stage_for_event,
            )

            if not isinstance(state, AgentKVQueryRing):
                raise RuntimeError("AGENTKV_RUNTIME_STATE_TYPE_MISMATCH")
            stage_ids = torch.full(
                (query_len,),
                AGENTKV_STAGE_THINK,
                dtype=torch.int32,
                device=token_query.device,
            )
            for span in config.get("event_token_spans") or []:
                if not isinstance(span, dict):
                    continue
                start = int(span.get("start", -1))
                end = int(span.get("end", -1))
                if end <= start:
                    continue
                stage = agentkv_stage_for_event(
                    str(span.get("role") or ""),
                    str(span.get("phase") or "others"),
                )
                mask = (token_positions >= start) & (token_positions < end)
                stage_ids[mask] = stage
            marker_reassignments = []
            if forward_batch.forward_mode.is_decode():
                tails = getattr(state, "_decode_marker_tails", None)
                stages = getattr(state, "_decode_marker_stages", None)
                if tails is None:
                    tails = state._decode_marker_tails = {}
                if stages is None:
                    stages = state._decode_marker_stages = {}
                tail = list(tails.get(self.attn.layer_id, []))
                current_stage = int(
                    stages.get(self.attn.layer_id, AGENTKV_STAGE_THINK)
                )
                markers = sorted(
                    [
                        (
                            tuple(int(x) for x in item.get("token_ids") or []),
                            int(item.get("stage")),
                        )
                        for item in config.get(
                            "agentkv_marker_stage_sequences", []
                        )
                        if item.get("token_ids")
                    ],
                    key=lambda item: len(item[0]),
                    reverse=True,
                )
                max_marker = max((len(item[0]) for item in markers), default=1)
                for local_idx, (token_id, token_position) in enumerate(
                    zip(token_ids.tolist(), token_positions.tolist())
                ):
                    tail.append((int(token_id), int(token_position)))
                    tail = tail[-max_marker:]
                    for sequence, marker_stage in markers:
                        if tuple(item[0] for item in tail[-len(sequence) :]) == sequence:
                            current_stage = marker_stage
                            marker_reassignments.append(
                                (
                                    [item[1] for item in tail[-len(sequence) :]],
                                    marker_stage,
                                )
                            )
                            break
                    stage_ids[local_idx] = current_stage
                tails[self.attn.layer_id] = tail
                stages[self.attn.layer_id] = current_stage
            state.write_layer(
                layer_id=self.attn.layer_id,
                query=token_query,
                positions=token_positions,
                stage_ids=stage_ids,
            )
            for marker_positions, marker_stage in marker_reassignments:
                state.reassign_positions(
                    layer_id=self.attn.layer_id,
                    positions=marker_positions,
                    stage=marker_stage,
                )

    @staticmethod
    def _reference_normal_positions(
        forward_batch: ForwardBatch,
        batch_idx: int,
        seq_len: int,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return canonical positions for the ordinary paged sequence."""

        device = query_positions.device
        ledgers = getattr(forward_batch, "history_kv_resident_positions", None)
        ledger = list(ledgers[batch_idx]) if ledgers and batch_idx < len(ledgers) else []
        if len(ledger) >= seq_len:
            return torch.tensor(ledger[:seq_len], dtype=torch.long, device=device)
        q_positions = query_positions.reshape(-1).to(dtype=torch.long)
        prefix_len = seq_len - q_positions.numel()
        known = ledger[:prefix_len]
        missing = prefix_len - len(known)
        if missing:
            if q_positions.numel():
                start = q_positions[:1] - missing
            else:
                start = torch.tensor(
                    [known[-1] + 1 if known else 0],
                    dtype=torch.long,
                    device=device,
                )
            missing_positions = start + torch.arange(
                missing, dtype=torch.long, device=device
            )
        else:
            missing_positions = q_positions.new_empty(0)
        positions = torch.cat(
            [torch.tensor(known, dtype=torch.long, device=device),
             missing_positions, q_positions]
        )
        if positions.numel() != seq_len:
            raise RuntimeError("REFERENCE_HISTORY_POSITION_LENGTH_MISMATCH")
        return positions

    def _reference_history_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Correctness route for per-layer/per-head persistent history."""

        if k is None or v is None:
            raise RuntimeError("REFERENCE_HISTORY_REQUIRES_EXPLICIT_KV")
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn,
            forward_batch.out_cache_loc,
            k,
            v,
            self.attn.k_scale,
            self.attn.v_scale,
        )
        key_buffer, value_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
            self.attn.layer_id
        )
        if key_buffer.ndim not in (3, 4) or tuple(key_buffer.shape[-2:]) != (
            self.num_kv_heads,
            self.head_dim,
        ):
            raise RuntimeError("REFERENCE_HISTORY_UNSUPPORTED_KV_BUFFER_LAYOUT")
        key_buffer = key_buffer.reshape(-1, self.num_kv_heads, self.head_dim)
        value_buffer = value_buffer.reshape(-1, self.num_kv_heads, self.head_dim)

        if forward_batch.forward_mode.is_decode():
            query_lens = [1] * int(forward_batch.batch_size)
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            if forward_batch.extend_seq_lens_cpu is None:
                raise RuntimeError("REFERENCE_HISTORY_EXTEND_LENGTHS_REQUIRED")
            query_lens = [int(item) for item in forward_batch.extend_seq_lens_cpu]
        else:
            raise RuntimeError("REFERENCE_HISTORY_FORWARD_MODE_UNSUPPORTED")

        states = getattr(forward_batch, "history_kv_reference_states", None) or []
        outputs = []
        offset = 0
        flat_positions = positions.reshape(-1)
        for batch_idx, query_len in enumerate(query_lens):
            query = q[offset : offset + query_len]
            query_pos = flat_positions[offset : offset + query_len]
            offset += query_len
            seq_len = int(forward_batch.seq_lens[batch_idx].item())
            req_pool_idx = int(forward_batch.req_pool_indices[batch_idx].item())
            slots = forward_batch.req_to_token_pool.req_to_token[
                req_pool_idx, :seq_len
            ].long()
            normal_key = key_buffer[slots].to(query.dtype)
            normal_value = value_buffer[slots].to(query.dtype)
            normal_positions = self._reference_normal_positions(
                forward_batch, batch_idx, seq_len, query_pos
            )
            layer = None
            if batch_idx < len(states) and states[batch_idx] is not None:
                layer = states[batch_idx].layer(self.attn.layer_id)
            if layer is None:
                layer = ReferenceLayerKV(
                    key=normal_key.new_empty(
                        self.num_kv_heads, 0, self.head_dim
                    ),
                    value=normal_value.new_empty(
                        self.num_kv_heads, 0, self.head_dim
                    ),
                    positions=torch.empty(
                        self.num_kv_heads,
                        0,
                        dtype=torch.long,
                        device=normal_key.device,
                    ),
                )
            outputs.append(
                reference_sdpa(
                    query,
                    layer,
                    normal_key,
                    normal_value,
                    normal_positions,
                    query_pos,
                    scale=self.scaling,
                    validate_history=False,
                    decode_causal=forward_batch.forward_mode.is_decode(),
                ).reshape(query_len, -1)
            )
        if offset != q.shape[0]:
            raise RuntimeError("REFERENCE_HISTORY_BATCH_LENGTH_MISMATCH")
        return torch.cat(outputs, dim=0)

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        if split_qkv_rmsnorm_rope is None:
            # compat fallback (27f21a588 port): repair_extract calls this
            # path directly, bypassing the forward dispatch guard
            return self.forward_prepare_native(
                positions, hidden_states, forward_batch=forward_batch
            )
        qkv = self._c2kv_project_qkv(hidden_states, forward_batch)

        if self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward_prepare_aiter_fused_mrope(
        self, positions, hidden_states, forward_batch
    ):
        """Fused QK-norm + 3D mRoPE + KV cache write for decode (ROCm/aiter).

        The fused HIP kernel replaces split → QK norm → mRoPE → cache write,
        so KV is already in the paged cache when this returns.
        Returns (q, None, None); caller must pass save_kv_cache=False to attn.
        """
        qkv, _ = self.qkv_proj(hidden_states)
        num_tokens = qkv.shape[0]

        qkv_3d = qkv.view(num_tokens, -1, self.head_dim)

        token_to_kv_pool = forward_batch.token_to_kv_pool
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(self.attn.layer_id)
        slot_mapping = forward_batch.out_cache_loc

        cos_sin = self.rotary_emb.cos_sin_cache
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(dtype=qkv.dtype)

        q_out = torch.empty(
            num_tokens,
            self.num_heads,
            self.head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        )

        fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
            qkv_3d,
            self.q_norm.weight,
            self.k_norm.weight,
            cos_sin,
            positions,
            num_tokens,
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_emb.is_neox_style,
            self.rotary_emb.mrope_section,
            self.rotary_emb.mrope_interleaved,
            self.q_norm.variance_epsilon,
            q_out,
            k_cache,
            v_cache,
            slot_mapping,
            self._fused_k_scale,
            self._fused_v_scale,
            None,
            None,
            False,
            False,
            0,
            0,
        )

        q = q_out.reshape(num_tokens, -1)
        return q, None, None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if get_global_server_args().rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        save_kv_cache = True
        reference_states = getattr(
            forward_batch, "history_kv_reference_states", None
        ) or []
        use_reference_attention = any(
            state is not None and state.layer(self.attn.layer_id) is not None
            for state in reference_states
        )
        use_reference_runtime = _requires_reference_runtime_qkv(forward_batch)
        use_aiter_fused = (
            self.use_fused_qk_norm_mrope
            and forward_batch.forward_mode.is_decode()
            and getattr(forward_batch, "c2kv_use_gist_projection", None) is None
            and get_global_server_args().rl_on_policy_target is None
            and not use_reference_runtime
        )

        if use_aiter_fused:
            q, k, v = self.forward_prepare_aiter_fused_mrope(
                positions, hidden_states, forward_batch
            )
            save_kv_cache = False
        elif (
            getattr(forward_batch, "c2kv_use_gist_projection", None) is not None
            or
            use_reference_runtime
            or
            not _is_npu
            or split_qkv_rmsnorm_rope is None
            or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        ):
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if get_global_server_args().rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        self._collect_history_kv_eviction_scores(q, k, positions, forward_batch)
        self._capture_history_kv_runtime_queries(
            q, k, v, positions, forward_batch
        )

        # ---------------------------------------------------------
        # C2KV_LAYER0_DIFF_DUMP
        #
        # Dump exactly one real C2KV EXTEND at layer 0:
        #   hidden -> RoPE Q/K/V -> paged attention output
        #   + the exact logical KV sequence read from paged cache.
        # ---------------------------------------------------------
        _c2kv_diff_path = os.environ.get("C2KV_DEBUG_LAYER0_DUMP")
        _c2kv_force_dump = (
            os.environ.get("C2KV_DEBUG_LAYER0_DUMP_FORCE") == "1"
        )
        _c2kv_min_qlen = int(
            os.environ.get("C2KV_DEBUG_LAYER0_MIN_QLEN", "100")
        )

        # ForwardBatch already contains corrected positions, but it does
        # not necessarily retain c2kv_position_corrections itself.
        #
        # For an EXTEND request:
        #   normal first position = extend_prefix_len
        #   C2KV first position   = extend_prefix_len + correction
        #
        # Therefore infer correction directly from the actual positions.
        _c2kv_prefix_len = None
        _c2kv_corr = None

        if (
            forward_batch.extend_prefix_lens_cpu is not None
            and positions is not None
            and positions.numel() > 0
        ):
            _c2kv_prefix_len = int(
                forward_batch.extend_prefix_lens_cpu[0]
            )
            _c2kv_corr_value = (
                int(positions.reshape(-1)[0].item())
                - _c2kv_prefix_len
            )

            if _c2kv_corr_value != 0:
                _c2kv_corr = [_c2kv_corr_value]

        if _c2kv_force_dump and _c2kv_corr is None:
            _c2kv_corr = [0]

        _c2kv_do_dump = bool(
            _c2kv_diff_path
            and self.attn.layer_id == 0
            and _c2kv_corr is not None
            and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            and positions is not None
            and positions.numel() >= _c2kv_min_qlen
            and not os.path.exists(_c2kv_diff_path)
        )

        if _c2kv_do_dump:
            _c2kv_debug = {
                "positions": positions.detach().cpu().clone(),
                "hidden": hidden_states.detach().cpu().clone(),
                "q": q.detach().cpu().clone(),
                "k_new": (
                    k.detach().cpu().clone()
                    if k is not None
                    else None
                ),
                "v_new": (
                    v.detach().cpu().clone()
                    if v is not None
                    else None
                ),
                "correction": list(_c2kv_corr),
                "num_heads": int(self.num_heads),
                "num_kv_heads": int(self.num_kv_heads),
                "head_dim": int(self.head_dim),
                "scaling": float(self.scaling),
            }

        if use_reference_attention:
            attn_output = self._reference_history_attention(
                q, k, v, positions, forward_batch
            )
        else:
            attn_output = self.attn(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )

        if _c2kv_do_dump:
            # self.attn() has now written current query K/V into cache.
            _req_idx = int(
                forward_batch.req_pool_indices[0].item()
            )
            _seq_len = int(
                forward_batch.seq_lens[0].item()
            )

            _slots = (
                forward_batch.req_to_token_pool.req_to_token[
                    _req_idx,
                    :_seq_len,
                ]
                .long()
            )

            _k_cache, _v_cache = (
                forward_batch.token_to_kv_pool.get_kv_buffer(
                    self.attn.layer_id
                )
            )

            _c2kv_debug.update(
                {
                    "req_idx": _req_idx,
                    "seq_len": _seq_len,
                    "slots": _slots.detach().cpu().clone(),
                    "cache_raw_shape": tuple(
                        _k_cache.shape
                    ),
                    "cache_page_size": (
                        int(_k_cache.shape[1])
                        if _k_cache.dim() == 4
                        else None
                    ),
                    "attn_mask": (
                        forward_batch.attn_backend.mask
                        .detach()
                        .cpu()
                        .clone()
                        if getattr(
                            forward_batch.attn_backend,
                            "mask",
                            None,
                        ) is not None
                        else None
                    ),
                    "k_cache_seq": (
                        (
                            _k_cache[
                                torch.div(
                                    _slots,
                                    int(_k_cache.shape[1]),
                                    rounding_mode="floor",
                                ),
                                torch.remainder(
                                    _slots,
                                    int(_k_cache.shape[1]),
                                ),
                            ]
                            if _k_cache.dim() == 4
                            else _k_cache[_slots]
                        )
                        .detach()
                        .cpu()
                        .clone()
                    ),
                    "v_cache_seq": (
                        (
                            _v_cache[
                                torch.div(
                                    _slots,
                                    int(_v_cache.shape[1]),
                                    rounding_mode="floor",
                                ),
                                torch.remainder(
                                    _slots,
                                    int(_v_cache.shape[1]),
                                ),
                            ]
                            if _v_cache.dim() == 4
                            else _v_cache[_slots]
                        )
                        .detach()
                        .cpu()
                        .clone()
                    ),
                    "attn_output": (
                        attn_output.detach().cpu().clone()
                    ),
                    "extend_prefix_lens": (
                        list(forward_batch.extend_prefix_lens_cpu)
                        if forward_batch.extend_prefix_lens_cpu is not None
                        else None
                    ),
                    "extend_seq_lens": (
                        list(forward_batch.extend_seq_lens_cpu)
                        if forward_batch.extend_seq_lens_cpu is not None
                        else None
                    ),
                }
            )

        output, _ = self.o_proj(attn_output)

        if _c2kv_do_dump:
            _c2kv_debug["o_proj_output"] = (
                output.detach().cpu().clone()
            )

            _dir = os.path.dirname(_c2kv_diff_path)
            if _dir:
                os.makedirs(_dir, exist_ok=True)

            torch.save(
                _c2kv_debug,
                _c2kv_diff_path,
            )

            print(
                "[C2KV LAYER0 DIFF DUMP]",
                {
                    "path": _c2kv_diff_path,
                    "seq_len": _seq_len,
                    "q_len": int(positions.numel()),
                    "positions": [
                        int(positions[0].item()),
                        int(positions[-1].item()),
                    ],
                    "correction": list(_c2kv_corr),
                    "k_cache_shape": tuple(
                        _c2kv_debug["k_cache_seq"].shape
                    ),
                },
                flush=True,
            )

        return output

    def forward_with_gist(
        self,
        hidden_states: torch.Tensor,   # (1, total_len, hidden_size)
        gist_mask: torch.Tensor,        # (1, gist_len) bool
        positions: torch.Tensor,        # (1, total_len) int64
        attention_mask,                 # BlockMask or None
        apply_gist_residual,
        projection_set: str = "history",
        **kwargs,
    ):

        gist_len = gist_mask.shape[1]
        total_len = hidden_states.shape[1]
        seq_len = total_len - gist_len

        input_hidden = hidden_states[:, :seq_len]    # (1, seq_len, hidden_size)
        gist_hidden = hidden_states[:, seq_len:]      # (1, gist_len, hidden_size)

        gist_hidden = apply_gist_residual(input_hidden, gist_hidden, **kwargs)

        qkv_input, _ = self.qkv_proj(input_hidden)
        q_input, k_input, v_input = qkv_input.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        qkv_gist, _ = self._c2kv_project_gist_qkv(gist_hidden, projection_set)
        q_gist, k_gist, v_gist = qkv_gist.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        q = torch.cat([q_input, q_gist], dim=1)   # (1, total_len, q_size)
        k = torch.cat([k_input, k_gist], dim=1)   # (1, total_len, kv_size)
        v = torch.cat([v_input, v_gist], dim=1)   # (1, total_len, kv_size)

        q, k = apply_qk_norm(
            q=q, k=k, q_norm=self.q_norm, k_norm=self.k_norm, head_dim=self.head_dim
        )

        # Save pre-RoPE gist K and V
        gist_key_values = (
            k[0, -gist_len:].contiguous().clone(),   # (gist_len, kv_size)
            v[0, -gist_len:].contiguous().clone(),   # (gist_len, kv_size)
        )

        # Apply RoPE; squeeze batch dim so rotary_emb gets (total_len, size)
        q = q.squeeze(0)   # (total_len, q_size)
        k = k.squeeze(0)   # (total_len, kv_size)
        v = v.squeeze(0)   # (total_len, kv_size)
        q, k = self.rotary_emb(positions, q, k)

        # Reshape for flex_attention: (batch, num_heads, seq_len, head_dim)
        q = q.view(1, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, total_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if _is_npu:
            if self.num_heads % self.num_kv_heads != 0:
                raise RuntimeError(
                    f"Invalid GQA heads: num_heads={self.num_heads}, "
                    f"num_kv_heads={self.num_kv_heads}"
                )

            q_attn = q.contiguous()
            # Keep KV heads unexpanded. Ascend FusionAttention supports GQA
            # directly when Q heads are an integer multiple of KV heads.
            k_attn = k.contiguous()
            v_attn = v.contiguous()

            # Ascend attention mask is a block mask: True/1 means masked.
            # C2KV attention_mask uses True as "can attend", so invert it.
            npu_mask = None if attention_mask is None else (~attention_mask).contiguous()

            attn_output = torch_npu.npu_fusion_attention(
                q_attn,
                k_attn,
                v_attn,
                q_attn.shape[1],
                input_layout="BNSD",
                atten_mask=npu_mask,
                scale=self.scaling,
                keep_prob=1.0,
                sparse_mode=0,
            )
            attn_output = _npu_fusion_attention_output(attn_output, q_attn.shape)

        else:
            attn_output = self.flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                scale=self.scaling,
                enable_gqa=True,
            )

        # Reshape back: (1, num_heads, total_len, head_dim) -> (total_len, hidden)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(
                total_len, self.num_heads * self.head_dim
            )
        )
        output, _ = self.o_proj(attn_output)
        # Manual all-reduce since o_proj has reduce_results=False
        output = tensor_model_parallel_all_reduce(output)
        output = output.unsqueeze(0)   # (1, total_len, hidden_size)

        return output, gist_key_values

    def forward_with_pic(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Encode every document token with residual QKV and retain pre-RoPE K/V."""
        qkv, _ = self.qkv_proj(hidden_states)
        residual_qkv, _ = self.residual_qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        residual_q, residual_k, residual_v = residual_qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        if "q" in self.pic_param:
            q = q + residual_q
        if "k" in self.pic_param:
            k = k + residual_k
        if "v" in self.pic_param:
            v = v + residual_v

        q, k = apply_qk_norm(
            q=q, k=k, q_norm=self.q_norm, k_norm=self.k_norm, head_dim=self.head_dim
        )
        pic_key_values = (
            k[0].contiguous().clone(),
            v[0].contiguous().clone(),
        )

        seq_len = hidden_states.shape[1]
        q, k = self.rotary_emb(positions, q.squeeze(0), k.squeeze(0))
        v = v.squeeze(0)
        q = q.view(1, seq_len, self.num_heads, self.head_dim).contiguous()
        k = k.view(1, seq_len, self.num_kv_heads, self.head_dim).contiguous()
        v = v.view(1, seq_len, self.num_kv_heads, self.head_dim).contiguous()

        attn_output = self.flash_attention_2(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=self.scaling,
            causal=True,
        )
        attn_output = attn_output.reshape(
            seq_len, self.num_heads * self.head_dim
        )
        output, _ = self.o_proj(attn_output)
        output = tensor_model_parallel_all_reduce(output).unsqueeze(0)
        return output, pic_key_values


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_parameters["rope_theta"]
        rope_scaling = config.rope_parameters
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            pic_enabled=getattr(config, "pic_enabled", False),
            pic_param=getattr(config, "pic_param", "qkv"),
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            tool_gist_uses_served_t0=_c2kv_tool_gist_uses_served_t0(
                config, get_global_server_args()
            ),
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, **norm_kwargs
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
            is_next_layer_sparse=False,
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            post_residual_addition=post_residual_addition,
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            cache=(
                [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
                if _is_npu
                and not get_global_server_args().disable_piecewise_cuda_graph
                and (
                    hasattr(self.mlp.gate_up_proj, "weight")
                    and hasattr(self.mlp.down_proj, "weight")
                )
                else None
            ),
        )
        hidden_states = self.mlp(hidden_states)
        if _is_npu and get_cmo_stream():
            wait_cmo_stream()
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual

    def forward_with_gist(
        self,
        hidden_states: torch.Tensor,
        gist_mask: torch.Tensor,
        positions: torch.Tensor,
        attention_mask,
        apply_gist_residual,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, gist_key_values = self.self_attn.forward_with_gist(
            hidden_states,
            gist_mask,
            positions,
            attention_mask,
            apply_gist_residual=apply_gist_residual,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, gist_key_values

    def forward_with_pic(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, pic_key_values = self.self_attn.forward_with_pic(
            hidden_states,
            positions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, pic_key_values


class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )

    def _init_c2kv(self, config, server_args) -> GistConfig:
        gist_cfg = GistConfig(
            gist_type=server_args.c2kv_gist_type,
            gist_param=server_args.c2kv_gist_param,
            gist_extra_embed_num=getattr(config, "gist_extra_embed_num", 1),
            gist_token_id=getattr(config, "gist_token_id", None),
            gist_residual_type=getattr(config, "gist_residual_type", "none"),
            gist_overlap=getattr(config, "gist_overlap", 0),
            hidden_size=config.hidden_size,
            attention_bias=getattr(config, "attention_bias", False),
        )
        self.gist_embed_tokens = nn.Embedding(
            gist_cfg.gist_extra_embed_num,
            config.hidden_size,
            dtype=torch.float32,
        )
        self.prepare_gist_input = get_prepare_gist_input_func(gist_cfg)
        return gist_cfg

    def _init_c2kv_tool_set(self, config, tool_config, server_args) -> GistConfig:
        """Gist embedding and mask/position builder of the second ("tool") set.

        The layout constants come from the tool checkpoint's own config.json;
        the projection weights are loaded afterwards by
        ``Qwen3ForCausalLM.load_c2kv_tool_gist_weights``.  Nothing here is
        consulted by the ordinary decode path.
        """
        gist_cfg = GistConfig(
            gist_type=server_args.c2kv_gist_type,
            gist_param=server_args.c2kv_gist_param,
            gist_extra_embed_num=int(tool_config.get("gist_extra_embed_num", 1)),
            gist_token_id=tool_config.get("gist_token_id"),
            gist_residual_type=tool_config.get("gist_residual_type", "none"),
            gist_overlap=int(tool_config.get("gist_overlap", 0)),
            hidden_size=config.hidden_size,
            attention_bias=bool(tool_config.get("attention_bias", False)),
        )
        self.tool_gist_embed_tokens = nn.Embedding(
            gist_cfg.gist_extra_embed_num,
            config.hidden_size,
            dtype=torch.float32,
        )
        self.prepare_tool_gist_input = get_prepare_gist_input_func(gist_cfg)
        return gist_cfg


# Architecture fields the tool gist checkpoint must share with the served
# model: its gist projections are applied to the served model's hidden states.
_C2KV_TOOL_GIST_SHAPE_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)


def _load_c2kv_tool_gist_config(source: str, config, server_args) -> Dict[str, Any]:
    """Read and validate ``<source>/config.json`` of the tool gist set."""
    import json

    with open(os.path.join(source, "config.json"), "r", encoding="utf-8") as handle:
        tool_config = json.load(handle)
    if tool_config.get("pic_enabled"):
        raise ValueError("--c2kv-tool-gist-weights must be a gist checkpoint, not PIC")
    gist_type = tool_config.get("gist_type")
    if gist_type is not None and gist_type != server_args.c2kv_gist_type:
        raise ValueError(
            "C2KV tool gist set declares gist_type "
            f"{gist_type!r} but the server runs {server_args.c2kv_gist_type!r}"
        )
    gist_param = tool_config.get("gist_param")
    if gist_param is not None and str(gist_param) != str(server_args.c2kv_gist_param):
        raise ValueError(
            "C2KV tool gist set declares gist_param "
            f"{gist_param!r} but the server runs {server_args.c2kv_gist_param!r}"
        )
    for field in _C2KV_TOOL_GIST_SHAPE_FIELDS:
        expected = getattr(config, field, None)
        actual = tool_config.get(field)
        if expected is not None and actual is not None and int(actual) != int(expected):
            raise ValueError(
                f"C2KV tool gist set {field}={actual} does not match the served "
                f"model ({expected})"
            )
    return tool_config


def _c2kv_tool_gist_uses_served_t0(config, server_args) -> bool:
    """Reuse the served FP32 gist Parameters only for one physical T0 source."""
    source = getattr(server_args, "c2kv_tool_gist_weights", None)
    served = getattr(server_args, "model_path", None)
    if (
        not source
        or not served
        or getattr(config, "history_memory_compression_domain", None) != "tool"
        or getattr(config, "history_memory_variant", None) != "T0"
    ):
        return False
    try:
        return os.path.samefile(
            os.path.expanduser(source), os.path.expanduser(served)
        )
    except OSError:
        return False


def _c2kv_gist_weight_files(source: str) -> List[str]:
    """Safetensors files of ``source`` that can hold gist tensors."""
    import json

    package = os.path.join(source, "c2kv-gist.safetensors")
    if os.path.isfile(package):
        return [package]
    index = os.path.join(source, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map") or {}
        files = sorted(
            {name for key, name in weight_map.items() if "gist_" in key}
        )
        if not files:
            raise ValueError(f"No gist tensors listed in {index}")
        return [os.path.join(source, name) for name in files]
    single = os.path.join(source, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    raise ValueError(
        "--c2kv-tool-gist-weights needs c2kv-gist.safetensors, "
        f"model.safetensors or model.safetensors.index.json under {source}"
    )


class Qwen3ForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # handle the lm head on different pp ranks
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

        _server_args = get_global_server_args()
        self.enable_c2kv = _server_args and _server_args.enable_c2kv
        self.full_length_pic = self.enable_c2kv and getattr(
            config, "pic_enabled", False
        )
        if self.enable_c2kv:
            if self.full_length_pic:
                logger.info(
                    "C2KV is using full-length residual-QKV PIC with storage "
                    "compression ratio 1."
                )
            else:
                self.gist_cfg = self.model._init_c2kv(config, _server_args)
            # Optional second gist projection set for tool-definition
            # compression (T0).  Weights are loaded by the model runner right
            # after the served checkpoint, see load_c2kv_tool_gist_weights.
            self.tool_gist_cfg = None
            self.c2kv_tool_gist_source = getattr(
                _server_args, "c2kv_tool_gist_weights", None
            )
            self.c2kv_tool_gist_identity = None
            self.c2kv_tool_gist_metadata = None
            self.c2kv_tool_gist_uses_served_t0 = False
            if self.c2kv_tool_gist_source:
                if self.full_length_pic:
                    raise ValueError(
                        "--c2kv-tool-gist-weights is not supported with PIC"
                    )
                tool_config = _load_c2kv_tool_gist_config(
                    self.c2kv_tool_gist_source, config, _server_args
                )
                self.c2kv_tool_gist_uses_served_t0 = (
                    _c2kv_tool_gist_uses_served_t0(config, _server_args)
                )
                if self.c2kv_tool_gist_uses_served_t0:
                    self.tool_gist_cfg = self.gist_cfg
                    self.model.tool_gist_embed_tokens = self.model.gist_embed_tokens
                    self.model.prepare_tool_gist_input = (
                        self.model.prepare_gist_input
                    )
                else:
                    self.tool_gist_cfg = self.model._init_c2kv_tool_set(
                        config, tool_config, _server_args
                    )
                self.c2kv_tool_gist_metadata = {
                    key: value
                    for key, value in tool_config.items()
                    if key.startswith("history_memory_") or key.startswith("gist_")
                }
            shadow_layer = getattr(_server_args, "c2kv_shadow_feature_layer", None)
            if shadow_layer is not None:
                num_layers = int(config.num_hidden_layers)
                normalized_layer = (
                    shadow_layer if shadow_layer >= 0 else num_layers + shadow_layer
                )
                # The fused residual path exposes the complete output of layer L
                # immediately before layer L+1. D3 requests -2, which therefore
                # maps to the final layer's input capture point.
                if not 0 <= normalized_layer < num_layers - 1:
                    raise ValueError(
                        "--c2kv-shadow-feature-layer must resolve before the final "
                        f"decoder layer; got {shadow_layer} for {num_layers} layers"
                    )
                if not self.pp_group.is_last_rank:
                    raise ValueError(
                        "C2KV native shadow feature capture does not support "
                        "pipeline parallel serving"
                    )
                self.capture_aux_hidden_states = True
                self.model.layers_to_capture = [normalized_layer + 1]
                self.c2kv_shadow_feature_layer = normalized_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    @torch.no_grad()
    def _c2kv_gist_set(self, projection_set: str):
        """(gist_cfg, gist embedding, prepare_gist_input) of one projection set."""
        if projection_set == "history":
            return (
                self.gist_cfg,
                self.model.gist_embed_tokens,
                self.model.prepare_gist_input,
            )
        if projection_set == "tool":
            if getattr(self, "tool_gist_cfg", None) is None:
                raise RuntimeError(
                    "C2KV_TOOL_GIST_UNAVAILABLE: projection_set='tool' needs "
                    "a server started with --c2kv-tool-gist-weights"
                )
            if self.c2kv_tool_gist_identity is None:
                raise RuntimeError(
                    "C2KV_TOOL_GIST_UNLOADED: the tool gist weights were not "
                    "loaded before extraction"
                )
            return (
                self.tool_gist_cfg,
                self.model.tool_gist_embed_tokens,
                self.model.prepare_tool_gist_input,
            )
        raise ValueError(f"Unknown C2KV projection set {projection_set!r}")

    def generate_gist(
        self, input_ids, attention_mask, ratio=4, projection_set="history", **kwargs
    ):
        """
        Run the gist extraction pass for one document.

        Args:
            input_ids:       (1, seq_len) int64 on GPU
            attention_mask:  (1, seq_len) bool on GPU
            ratio:           compression ratio; gist_len = ceil(seq_len / ratio)
            projection_set:  "history" (the served checkpoint's gist set) or
                             "tool" (--c2kv-tool-gist-weights)

        Returns:
            gist_key_values: List[(K, V)] per layer, each (gist_len, kv_size) float,
                             pre-RoPE. TP-local.
            gist_mask:       (1, gist_len) bool
            gist_position_ids: (1, gist_len) int64
        """
        autocast_active = bool(kwargs.pop("_c2kv_fp32_autocast_active", False))
        gist_cfg, gist_embed_tokens, prepare_gist_input = self._c2kv_gist_set(
            projection_set
        )
        base_dtype = self.model.embed_tokens.weight.dtype
        gist_dtype = gist_embed_tokens.weight.dtype
        if gist_dtype != base_dtype and not autocast_active:
            with torch.autocast(
                device_type=input_ids.device.type,
                dtype=base_dtype,
            ):
                return self.generate_gist(
                    input_ids,
                    attention_mask,
                    ratio=ratio,
                    projection_set=projection_set,
                    _c2kv_fp32_autocast_active=True,
                    **kwargs,
                )

        block_mask, gist_mask, position_ids = prepare_gist_input(
            input_ids, attention_mask, ratio=ratio
        )
        gist_len = gist_mask.shape[1]
        device = input_ids.device

        gist_embed = gist_embed_tokens(
            torch.zeros((1, gist_len), dtype=torch.long, device=device)
        ).to(dtype=self.model.embed_tokens.weight.dtype)
        inputs_embeds = torch.cat(
            [self.model.embed_tokens(input_ids), gist_embed], dim=1
        )

        hidden_states = inputs_embeds
        gist_key_values = []
        for layer_idx, layer in enumerate(self.model.layers):
            layer_residual = get_apply_gist_residual_func(gist_cfg, layer_idx)
            hidden_states, layer_kv = layer.forward_with_gist(
                hidden_states,
                gist_mask,
                positions=position_ids.squeeze(0),
                attention_mask=block_mask,
                apply_gist_residual=layer_residual,
                projection_set=projection_set,
                ratio=ratio,
            )
            gist_key_values.append(layer_kv)
            # These are cloned K/V payload tensors. Logical bytes therefore
            # exclude the fused QKV backing storage; CUDA allocator peaks still
            # account for the full Q/K/V workspace separately.
            paper_telemetry.sample(
                "forward_with_gist",
                tensors=gist_key_values,
                temporary_kv=True,
            )

        gist_position_ids = position_ids[:, -gist_len:].contiguous()

        # Debug: dump the pre-RoPE C2KV states produced by SGLang.
        dump_path = os.environ.get("C2KV_DEBUG_GIST_DUMP")
        if dump_path:
            dump_obj = {
                "input_ids": input_ids.detach().cpu(),
                "gist_mask": gist_mask.detach().cpu(),
                "gist_position_ids": gist_position_ids.detach().cpu(),
                "kv": [
                    (
                        k.detach().cpu(),
                        v.detach().cpu(),
                    )
                    for k, v in gist_key_values
                ],
            }
            torch.save(dump_obj, dump_path)
            logger.warning(
                "[C2KV DEBUG] saved SGLang pre-RoPE gist KV to %s",
                dump_path,
            )

        return gist_key_values, gist_mask, gist_position_ids

    @torch.no_grad()
    def generate_raw_repair_kv(
        self,
        input_ids: torch.Tensor,
        span_start: int,
        span_end: int,
        *,
        position_offset: int = 0,
        repair_position_ids: Optional[List[int]] = None,
        raw_kv_position_mode: str = "rotated",
        history_kv_method: Optional[str] = None,
        history_kv_target_tokens: Optional[int] = None,
        history_kv_retention_ratio: Optional[float] = None,
        history_kv_recent_window: int = 64,
        history_kv_kernel_size: int = 5,
        history_kv_pooling: str = "avgpool",
        history_kv_h2o_recent_fraction: float = 0.5,
        history_kv_selectable_relative_indices: Optional[List[int]] = None,
        history_kv_mandatory_relative_indices: Optional[List[int]] = None,
        history_kv_recovery_mode: Optional[str] = None,
        history_kv_recovery_relative_indices: Optional[List[int]] = None,
        cacheblend: Optional[Dict[str, Any]] = None,
    ):
        """Run a correctness-first full prefill and capture raw repair KV.

        Used by the C2KV repair endpoints. It intentionally captures ordinary
        self-attention K/V with the frozen base projections, not gist/PIC K/V
        (paper 2607.17715 section 3.3.2, original-token invariance: raw KV must
        be what the base model computes in this exact context). The forward runs
        at ``position_offset + i`` so the attention output, and therefore every
        later layer's K/V, is the full-context one.

        ``raw_kv_position_mode`` selects the STORED form of K, independently of
        where the entry is later placed:

        * ``rotated``  - K carries native Full-prompt RoPE (already_rotated=True)
          and can only be re-injected at its original absolute positions.
        * ``pre_rope`` - K is captured after base QKV + QK norm but before RoPE
          (already_rotated=False); injection applies RoPE exactly once, either at
          the recorded positions (in_place / append_keep_ledger) or at a fresh
          tail position (append_tail, which requires this mode).

        ``repair_position_ids`` records the positions to store with the entry.
        See c2kv/c2kv_serving_semantics.md.
        """

        if cacheblend:
            # CacheBlend (chunk-KV reuse + selective recompute) shares this
            # entry point so every caller/route is the same; the algorithm
            # lives in mem_cache/cacheblend.py (see its module docstring).
            return self.generate_cacheblend_kv(
                input_ids,
                span_start=span_start,
                span_end=span_end,
                position_offset=position_offset,
                raw_kv_position_mode=raw_kv_position_mode,
                history_kv_method=history_kv_method,
                cacheblend=cacheblend,
            )

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"generate_raw_repair_kv expects input_ids shape (1, L), got {input_ids.shape}."
            )
        seq_len = int(input_ids.shape[1])
        if not (0 <= span_start <= span_end <= seq_len):
            raise ValueError(
                f"Invalid repair span: {span_start=}, {span_end=}, {seq_len=}."
            )
        if span_start == span_end:
            raise ValueError("repair span must be non-empty.")
        if raw_kv_position_mode not in {"rotated", "pre_rope"}:
            raise ValueError(
                f"Unsupported raw_kv_position_mode: {raw_kv_position_mode!r}."
            )
        if repair_position_ids is not None and len(repair_position_ids) != (
            span_end - span_start
        ):
            raise ValueError(
                "repair_position_ids length mismatch: "
                f"{len(repair_position_ids)} != {span_end - span_start}"
            )

        device = input_ids.device
        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            dtype=torch.long,
            device=device,
        )
        hidden_states = self.model.embed_tokens(input_ids).squeeze(0)
        raw_key_values = []
        history_scores: List[torch.Tensor] = []
        requested_span_tokens = span_end - span_start
        sparse_selectable, sparse_mandatory = validate_sparse_repair_partition(
            requested_span_tokens,
            history_kv_selectable_relative_indices,
            history_kv_mandatory_relative_indices,
            history_kv_target_tokens,
        )
        history_method = (history_kv_method or "").strip().lower()
        if history_method == "snapkv":
            history_method = "snapkv_persistent"
        if history_method == "pyramid":
            history_method = "pyramidkv"
        require_rotated_headwise_storage(history_method, raw_kv_position_mode)
        if history_kv_selectable_relative_indices is not None:
            if history_method not in SPARSE_REPAIR_METHODS:
                raise ValueError("unsupported sparse repair history method")
            if history_method != "streamingllm" and raw_kv_position_mode != "rotated":
                raise ValueError("headwise sparse repair requires rotated raw KV")
        if history_method.startswith("snapkv"):
            snap_recent_window = int(history_kv_recent_window)
            snap_kernel_size = int(history_kv_kernel_size)
            snap_pooling = history_kv_pooling.strip().lower()
            if snap_recent_window <= 0:
                raise ValueError(
                    "history_kv_recent_window must be positive for SnapKV, got "
                    f"{snap_recent_window}"
                )
            if snap_kernel_size <= 0:
                raise ValueError(
                    "history_kv_kernel_size must be positive for SnapKV, got "
                    f"{snap_kernel_size}"
                )
            if snap_pooling not in {"avgpool", "maxpool"}:
                raise ValueError(
                    "history_kv_pooling must be 'avgpool' or 'maxpool' for "
                    f"SnapKV, got {snap_pooling!r}"
                )
        npu_forward_batch_stub = None
        if _is_npu:
            npu_forward_batch_stub = SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(
                    start_layer=self.model.layers[0].self_attn.attn.layer_id
                )
            )

        for layer in self.model.layers:
            residual = hidden_states
            attn_input = layer.input_layernorm(hidden_states)
            # Normal SGLang prefill/extend on Ascend takes the native
            # QK-norm/RoPE preparation path before entering the Ascend
            # attention backend.  Repair KV must be captured from that same
            # Full-prefill path; using the decode-oriented NPU fused prepare
            # changes the raw K/V slightly and breaks raw-all replacement
            # equivalence on sensitive BFCL trajectories.
            # Same ops as forward_prepare_native (qkv_proj -> qk_norm -> rope),
            # split so the span's K can be captured pre-RoPE. Base projections
            # only (self.qkv_proj, never gist_qkv_proj): paper 3.3.2
            # original-token invariance.
            attn = layer.self_attn
            qkv, _ = attn.qkv_proj(attn_input)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=attn.q_norm,
                k_norm=attn.k_norm,
                head_dim=attn.head_dim,
                alt_stream=attn.alt_stream,
            )
            # Clone BEFORE rope: rotary_emb may rotate k in place, so a pre_rope
            # capture taken after the call would silently be rotated.
            k_pre_rope = k[span_start:span_end].contiguous().clone()
            v_span = v[span_start:span_end].contiguous().clone()
            q, k = attn.rotary_emb(positions, q, k)
            repair_k_span = (
                k_pre_rope
                if raw_kv_position_mode == "pre_rope"
                else k[span_start:span_end].contiguous().clone()
            )
            raw_key_values.append((repair_k_span, v_span))
            paper_telemetry.sample(
                "forward_repair_kv",
                tensors=raw_key_values,
                temporary_kv=True,
            )

            q = q.view(1, seq_len, layer.self_attn.num_heads, layer.self_attn.head_dim)
            k_attn = k.view(
                1,
                seq_len,
                layer.self_attn.num_kv_heads,
                layer.self_attn.head_dim,
            )
            v_attn = v.view(
                1,
                seq_len,
                layer.self_attn.num_kv_heads,
                layer.self_attn.head_dim,
            )
            q = q.transpose(1, 2).contiguous()
            k_attn = k_attn.transpose(1, 2).contiguous()
            v_attn = v_attn.transpose(1, 2).contiguous()

            if history_method in HEADWISE_HISTORY_KV_METHODS or (
                history_kv_selectable_relative_indices is not None
                and history_method == "pyramidkv"
            ):
                score_query_start = repair_score_query_start(
                    history_method, seq_len, history_kv_recent_window
                )
                layer_score = attention_scores_by_kv_head(
                    q,
                    k_attn,
                    scale=layer.self_attn.scaling,
                    query_start=score_query_start,
                    query_end=seq_len,
                    key_start=span_start,
                    key_end=span_end,
                )
                history_scores.append(layer_score.detach())

            if _is_npu:
                # Match the serving attention path: do not materialize repeated
                # KV heads for GQA. Repair KV must be captured from the same
                # Full-context computation that the normal Ascend backend uses.
                k_run = k_attn.contiguous()
                v_run = v_attn.contiguous()
                blocked = torch.triu(
                    torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
                    diagonal=1,
                ).view(1, 1, seq_len, seq_len)
                if os.environ.get(
                    "C2KV_REPAIR_EXTRACT_ATTN_IMPL",
                    "prompt_flash",
                ) == "prompt_flash" and hasattr(
                    torch_npu, "npu_prompt_flash_attention"
                ):
                    attn_output = torch_npu.npu_prompt_flash_attention(
                        q,
                        k_run,
                        v_run,
                        num_heads=q.shape[1],
                        num_key_value_heads=k_run.shape[1],
                        input_layout="BNSD",
                        atten_mask=blocked,
                        scale_value=layer.self_attn.scaling,
                        sparse_mode=0,
                    )
                else:
                    attn_output = torch_npu.npu_fusion_attention(
                        q,
                        k_run,
                        v_run,
                        q.shape[1],
                        input_layout="BNSD",
                        atten_mask=blocked,
                        scale=layer.self_attn.scaling,
                        keep_prob=1.0,
                        sparse_mode=0,
                    )
                attn_output = _npu_fusion_attention_output(attn_output, q.shape)
            else:
                if layer.self_attn.num_heads != layer.self_attn.num_kv_heads:
                    groups = layer.self_attn.num_heads // layer.self_attn.num_kv_heads
                    k_run = k_attn.repeat_interleave(groups, dim=1)
                    v_run = v_attn.repeat_interleave(groups, dim=1)
                else:
                    k_run = k_attn
                    v_run = v_attn
                scores = torch.matmul(
                    q.float(),
                    k_run.transpose(-2, -1).float(),
                ) * layer.self_attn.scaling
                keep = torch.tril(
                    torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
                ).view(1, 1, seq_len, seq_len)
                scores = scores.masked_fill(~keep, float("-inf"))
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                    v_run.dtype
                )
                attn_output = torch.matmul(probs, v_run)

            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(seq_len, layer.self_attn.num_heads * layer.self_attn.head_dim)
            )
            attn_output, _ = layer.self_attn.o_proj(attn_output)
            attn_output = tensor_model_parallel_all_reduce(attn_output)
            hidden_states = residual + attn_output

            residual = hidden_states
            mlp_input = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(mlp_input)

        if repair_position_ids is None:
            repair_positions = positions[span_start:span_end]
        else:
            repair_positions = torch.tensor(
                repair_position_ids, dtype=torch.long, device=device
            )
        full_raw_key_values = raw_key_values
        full_repair_positions = repair_positions

        history_meta = None
        if history_method:
            if history_method not in {
                "streamingllm",
                "h2o",
                "snapkv_persistent",
                "snapkv_refresh",
                "pyramidkv",
                "kivi",
            }:
                raise ValueError(
                    f"Unsupported history_kv_method for repair extraction: {history_method!r}."
                )
            if history_kv_target_tokens is not None:
                target_tokens = int(history_kv_target_tokens)
            elif history_kv_retention_ratio is not None:
                target_tokens = int(
                    torch.ceil(
                        torch.tensor(
                            requested_span_tokens * float(history_kv_retention_ratio)
                        )
                    ).item()
                )
            else:
                target_tokens = requested_span_tokens
            target_tokens = max(1, min(requested_span_tokens, target_tokens))
            if history_kv_selectable_relative_indices is not None and (
                target_tokens < len(sparse_mandatory)
            ):
                raise ValueError("sparse repair target is below mandatory token cost")

            def _unique_sorted(indices: Iterable[int]) -> List[int]:
                return sorted({int(i) for i in indices if 0 <= int(i) < requested_span_tokens})

            def _kivi_qdq(
                tensor: torch.Tensor,
                *,
                bits: int,
                group_size: int,
                residual_length: int,
                per_token: bool,
            ) -> torch.Tensor:
                """Apply KIVI-style asymmetric fake quantize/dequantize."""
                if tensor.numel() == 0:
                    return tensor
                levels = float((1 << bits) - 1)
                residual = max(0, min(residual_length, int(tensor.shape[0])))
                main = tensor[:-residual] if residual else tensor
                tail = tensor[-residual:] if residual else None
                if main.numel() == 0:
                    return tensor.clone()
                source = main.float()
                restored_parts = []
                if per_token:
                    # KIVI values: quantize each token independently in groups
                    # along the head dimension.
                    for start in range(0, int(source.shape[-1]), group_size):
                        chunk = source[..., start : start + group_size]
                        minimum = chunk.amin(dim=-1, keepdim=True)
                        maximum = chunk.amax(dim=-1, keepdim=True)
                        scale = (maximum - minimum).clamp_min(1e-6) / levels
                        quantized = torch.round((chunk - minimum) / scale).clamp_(0, levels)
                        restored_parts.append(quantized * scale + minimum)
                    restored = torch.cat(restored_parts, dim=-1)
                else:
                    # KIVI keys: per-channel groups over the token dimension.
                    for start in range(0, int(source.shape[0]), group_size):
                        chunk = source[start : start + group_size]
                        minimum = chunk.amin(dim=0, keepdim=True)
                        maximum = chunk.amax(dim=0, keepdim=True)
                        scale = (maximum - minimum).clamp_min(1e-6) / levels
                        quantized = torch.round((chunk - minimum) / scale).clamp_(0, levels)
                        restored_parts.append(quantized * scale + minimum)
                    restored = torch.cat(restored_parts, dim=0)
                restored = restored.to(dtype=tensor.dtype)
                if tail is not None:
                    restored = torch.cat([restored, tail.clone()], dim=0)
                return restored.contiguous()

            if history_kv_selectable_relative_indices is not None:
                selected_by_layer, sparse_metadata = select_sparse_repair_indices(
                    history_method,
                    history_scores,
                    sparse_selectable,
                    sparse_mandatory,
                    target_tokens,
                    recent_window=int(history_kv_recent_window),
                    kernel_size=int(history_kv_kernel_size),
                    pooling=history_kv_pooling.strip().lower(),
                    h2o_recent_fraction=float(history_kv_h2o_recent_fraction),
                    num_layers=len(raw_key_values),
                    device=device,
                )
                if sparse_metadata["per_head_selection"]:
                    raw_key_values = [
                        gather_paired_kv(key, value, indices)
                        for (key, value), indices in zip(
                            raw_key_values, selected_by_layer
                        )
                    ]
                    length = selected_by_layer[0].shape[1]
                    repair_positions = full_repair_positions[-length:].contiguous().clone()
                    repair_positions[-1] = full_repair_positions[-1]
                    sparse_metadata.update(summarize_headwise_indices(selected_by_layer))
                    sparse_metadata["repair_positions_semantics"] = "ledger_only_recent_suffix"
                else:
                    selected_tensor = selected_by_layer[0][0]
                    raw_key_values = [
                        (
                            key.index_select(0, selected_tensor).contiguous().clone(),
                            value.index_select(0, selected_tensor).contiguous().clone(),
                        )
                        for key, value in raw_key_values
                    ]
                    repair_positions = repair_positions.index_select(
                        0, selected_tensor
                    ).contiguous()
                history_meta = {
                    "history_kv_method": history_method,
                    "history_boundary_adaptation": True,
                    "requested_span_tokens": requested_span_tokens,
                    "selected_token_count": int(repair_positions.numel()),
                    "selection_reason": "global_schema_candidates_with_mandatory_protocol",
                    "selection_indices_coordinate_space": "span_relative",
                    **sparse_metadata,
                }
            elif history_method == "kivi":
                bits = max(1, int(os.environ.get("C2KV_KIVI_BITS", "2")))
                group_size = max(1, int(os.environ.get("C2KV_KIVI_GROUP_SIZE", "32")))
                residual_length = max(
                    0, int(os.environ.get("C2KV_KIVI_RESIDUAL_LENGTH", "32"))
                )
                raw_key_values = [
                    (
                        _kivi_qdq(
                            key,
                            bits=bits,
                            group_size=group_size,
                            residual_length=residual_length,
                            per_token=False,
                        ),
                        _kivi_qdq(
                            value,
                            bits=bits,
                            group_size=group_size,
                            residual_length=residual_length,
                            per_token=True,
                        ),
                    )
                    for key, value in raw_key_values
                ]
                selected_rel = list(range(requested_span_tokens))
                history_meta = {
                    "history_kv_method": history_method,
                    "algorithm_version": "kivi_2bit_qdq_v1",
                    "history_boundary_adaptation": True,
                    "requested_span_tokens": requested_span_tokens,
                    "target_tokens": requested_span_tokens,
                    "selected_token_count": requested_span_tokens,
                    "selected_relative_indices": selected_rel,
                    "selection_reason": "kivi_qdq_full_history_no_token_eviction",
                    "per_head_selection": False,
                    "kivi_bits": bits,
                    "kivi_group_size": group_size,
                    "kivi_residual_length": residual_length,
                }
            elif history_method == "streamingllm":
                selected_tensor = select_streamingllm_indices(
                    requested_span_tokens,
                    target_tokens=target_tokens,
                    device=device,
                )
                selected_rel = [int(index) for index in selected_tensor.tolist()]
                raw_key_values = [
                    (
                        key.index_select(0, selected_tensor).contiguous().clone(),
                        value.index_select(0, selected_tensor).contiguous().clone(),
                    )
                    for key, value in raw_key_values
                ]
                repair_positions = repair_positions.index_select(
                    0, selected_tensor
                ).contiguous()
                kept_sinks = min(
                    4,
                    max(0, target_tokens - 1),
                    max(0, requested_span_tokens - 1),
                )
                history_meta = {
                    "history_kv_method": history_method,
                    "algorithm_version": "streamingllm_history_boundary_v1",
                    "history_boundary_adaptation": True,
                    "requested_span_tokens": requested_span_tokens,
                    "target_tokens": target_tokens,
                    "selected_token_count": len(selected_rel),
                    "selected_relative_indices": selected_rel,
                    "selection_reason": "attention_sinks_plus_recent_suffix",
                    "sink_tokens": kept_sinks,
                    "recent_tokens": target_tokens - kept_sinks,
                    "per_head_selection": False,
                }
            elif history_method == "pyramidkv":
                if history_scores:
                    layer_scores = history_scores
                else:
                    layer_scores = [
                        torch.zeros(
                            requested_span_tokens,
                            dtype=torch.float32,
                            device=device,
                        )
                        for _ in range(len(self.model.layers))
                ]
                num_layers = max(1, len(layer_scores))
                budget_scale = float(
                    os.environ.get("C2KV_PYRAMIDKV_BUDGET_SCALE", "0.66")
                )
                scaled_target = max(1, int(round(target_tokens * budget_scale)))
                low_budget = max(1, min(requested_span_tokens, int(round(scaled_target * 1.5))))
                high_budget = max(1, min(requested_span_tokens, int(round(scaled_target * 0.5))))
                per_layer_budgets = []
                per_layer_selected_counts = []
                union_selected: set[int] = set()
                for layer_idx, scores in enumerate(layer_scores):
                    if num_layers == 1:
                        layer_budget = target_tokens
                    else:
                        # PyramidKV-style funnel: lower layers retain a larger
                        # history cache, upper layers retain a smaller cache.
                        frac = layer_idx / float(num_layers - 1)
                        layer_budget = int(round(low_budget * (1.0 - frac) + high_budget * frac))
                    layer_budget = max(1, min(requested_span_tokens, layer_budget))
                    per_layer_budgets.append(layer_budget)
                    recent_budget = min(
                        layer_budget,
                        max(1, min(int(history_kv_recent_window or 64), requested_span_tokens)),
                    )
                    recent_rel = list(
                        range(requested_span_tokens - recent_budget, requested_span_tokens)
                    )
                    past_budget = max(0, layer_budget - len(recent_rel))
                    layer_selected = set(recent_rel)
                    past_len = max(0, requested_span_tokens - len(recent_rel))
                    if past_budget > 0 and past_len > 0:
                        _, top_idx = torch.topk(
                            scores[:past_len],
                            k=min(past_budget, past_len),
                            largest=True,
                        )
                        layer_selected.update(int(i) for i in top_idx.tolist())
                    per_layer_selected_counts.append(len(layer_selected))
                    union_selected.update(layer_selected)
                selected_rel = sorted(union_selected)
                selected_tensor = torch.tensor(
                    selected_rel, dtype=torch.long, device=device
                )
                raw_key_values = [
                    (
                        key.index_select(0, selected_tensor).contiguous().clone(),
                        value.index_select(0, selected_tensor).contiguous().clone(),
                    )
                    for key, value in raw_key_values
                ]
                repair_positions = repair_positions.index_select(
                    0, selected_tensor
                ).contiguous()
                history_meta = {
                    "history_kv_method": history_method,
                    "algorithm_version": "pyramidkv_shared_page_table_approximation_v0",
                    "history_boundary_adaptation": True,
                    "official_algorithm_implemented": False,
                    "requested_span_tokens": requested_span_tokens,
                    "target_tokens": target_tokens,
                    "selected_token_count": len(selected_rel),
                    "selected_relative_indices": selected_rel,
                    "selection_reason": "layer_budget_union_shared_page_table_approximation",
                    "per_head_selection": False,
                    "budget_preserved": len(selected_rel) == target_tokens,
                    "shared_page_table_approximation": True,
                    "pyramidkv_budget_scale": budget_scale,
                    "scaled_target_tokens": scaled_target,
                    "per_layer_budget_tokens": per_layer_budgets,
                    "per_layer_selected_counts": per_layer_selected_counts,
                }
            else:
                if len(history_scores) != len(raw_key_values):
                    raise RuntimeError(
                        "history KV scoring did not produce exactly one score tensor "
                        f"per layer: {len(history_scores)} != {len(raw_key_values)}"
                    )
                selected_by_layer: List[torch.Tensor] = []
                compressed_key_values = []
                for (key, value), layer_scores in zip(
                    raw_key_values, history_scores
                ):
                    if history_method == "h2o":
                        selected = select_h2o_prefill_indices(
                            layer_scores,
                            target_tokens=target_tokens,
                            recent_fraction=float(history_kv_h2o_recent_fraction),
                        )
                    else:
                        selected = select_snapkv_indices(
                            layer_scores,
                            target_tokens=target_tokens,
                            recent_window=snap_recent_window,
                            kernel_size=snap_kernel_size,
                            pooling=snap_pooling,
                        )
                    selected_by_layer.append(selected)
                    compressed_key_values.append(
                        gather_paired_kv(key, value, selected)
                    )
                raw_key_values = compressed_key_values

                # A rotated headwise entry has no single true token position per
                # physical slot.  The shared vector is ledger-only; its final
                # value preserves the original history boundary used by in_place.
                original_span_end = repair_positions[-1].clone()
                repair_positions = repair_positions[-target_tokens:].contiguous()
                repair_positions[-1] = original_span_end

                if history_method == "h2o":
                    recent_budget = max(
                        1,
                        min(
                            target_tokens,
                            int(
                                round(
                                    target_tokens
                                    * float(history_kv_h2o_recent_fraction)
                                )
                            ),
                        ),
                    )
                    algorithm_version = "h2o_prefill_gqa_v1"
                    reason = "prefill_heavy_hitter_plus_recent_per_kv_head"
                    scoring_query_tokens = seq_len
                    algorithm_fields = {
                        "online_decode_updates": False,
                        "scope": "prefill_history_boundary",
                        "heavy_tokens_per_head": target_tokens - recent_budget,
                        "recent_tokens_per_head": recent_budget,
                    }
                else:
                    observation_window = min(snap_recent_window, seq_len)
                    recent_budget = min(
                        target_tokens,
                        snap_recent_window,
                        requested_span_tokens,
                    )
                    algorithm_version = "snapkv_gqa_headwise_v1"
                    reason = "observation_pooling_plus_recent_per_kv_head"
                    scoring_query_tokens = observation_window
                    algorithm_fields = {
                        "observation_window": observation_window,
                        "pooling": snap_pooling,
                        "kernel_size": snap_kernel_size,
                        "past_tokens_per_head": target_tokens - recent_budget,
                        "recent_tokens_per_head": recent_budget,
                    }

                history_meta = {
                    "history_kv_method": history_method,
                    "algorithm_version": algorithm_version,
                    "history_boundary_adaptation": True,
                    "requested_span_tokens": requested_span_tokens,
                    "target_tokens": target_tokens,
                    "selected_token_count": target_tokens,
                    # No single source-token set is true across layers/heads.
                    "selected_relative_indices": None,
                    "selection_reason": reason,
                    "per_head_selection": True,
                    "query_group_reduction": "sum",
                    "scoring_query_tokens": scoring_query_tokens,
                    "selection_indices_coordinate_space": "span_relative",
                    "repair_positions_semantics": "ledger_only_recent_suffix",
                    **algorithm_fields,
                    **summarize_headwise_indices(selected_by_layer),
                }
            recovery_mode = (history_kv_recovery_mode or "").strip().lower()
            if recovery_mode:
                if recovery_mode not in {"append", "replace"}:
                    raise ValueError(
                        f"Unsupported history_kv_recovery_mode: {recovery_mode!r}")
                recovery_rel = _unique_sorted(
                    history_kv_recovery_relative_indices or [])
                if not recovery_rel:
                    raise ValueError(
                        "history_kv recovery requires non-empty source-token indices")
                if history_method in HEADWISE_HISTORY_KV_METHODS:
                    restored, recovery_accounting = dense_headwise_recovery_indices(
                        selected_by_layer, recovery_rel, seq_len=requested_span_tokens)
                    raw_key_values = [
                        gather_paired_kv(key, value, indices)
                        for (key, value), indices in zip(full_raw_key_values, restored)
                    ]
                    length = recovery_accounting["after_recovery_active_tokens"]
                    # Keys already carry each source token's original RoPE.
                    # Shared positions remain ledger-only and preserve span end.
                    repair_positions = full_repair_positions[-length:].contiguous().clone()
                    history_meta.update({
                        "recovery_mode": recovery_mode,
                        "recovery_semantics": "headwise_raw_union_dense_completion_v1",
                        "operator_equivalent_for_raw_token_eviction": True,
                        **recovery_accounting,
                        "recovery_relative_indices": recovery_rel,
                        "selected_token_count": length,
                        "selected_relative_indices": None,
                        **summarize_headwise_indices(restored),
                    })
                    return raw_key_values, repair_positions.view(1, -1), history_meta
                retained_before = _unique_sorted(selected_rel)
                merged_rel, recovery_accounting = deduplicated_recovery_indices(
                    retained_before, recovery_rel,
                    seq_len=requested_span_tokens)
                selected_tensor = torch.tensor(
                    merged_rel, dtype=torch.long, device=device)
                raw_key_values = [
                    (
                        key.index_select(0, selected_tensor).contiguous().clone(),
                        value.index_select(0, selected_tensor).contiguous().clone(),
                    )
                    for key, value in full_raw_key_values
                ]
                repair_positions = full_repair_positions.index_select(
                    0, selected_tensor).contiguous()
                history_meta.update({
                    "recovery_mode": recovery_mode,
                    "recovery_semantics": "deduplicated_raw_token_union",
                    "operator_equivalent_for_raw_token_eviction": True,
                    "recovery_target_coverage": 1.0,
                    **recovery_accounting,
                    "recovery_relative_indices": recovery_rel,
                    "selected_token_count": len(merged_rel),
                    "selected_relative_indices": merged_rel,
                })
        repair_positions = repair_positions.view(1, -1).contiguous()
        if history_meta is not None:
            return raw_key_values, repair_positions, history_meta
        return raw_key_values, repair_positions


    @torch.no_grad()
    def generate_cacheblend_kv(
        self,
        input_ids: torch.Tensor,
        span_start: int,
        span_end: int,
        *,
        position_offset: int = 0,
        raw_kv_position_mode: str = "rotated",
        history_kv_method: Optional[str] = None,
        cacheblend: Optional[Dict[str, Any]] = None,
    ):
        """CacheBlend repair extraction: the span's KV = per-chunk standalone KV
        with the highest-deviation ``recomp_ratio`` of its tokens recomputed
        in context (mem_cache/cacheblend.py, EuroSys artifact semantics).

        Same contract as ``generate_raw_repair_kv``: ``(raw_key_values,
        repair_positions, meta)`` with K post-RoPE at the span's absolute
        positions (``rotated``), base projections only (paper 2607.17715
        section 3.3.2 original-token invariance holds for the recomputed rows;
        the reused rows are the base model's out-of-context KV by design).
        The entry can only be placed ``in_place`` at those positions.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"generate_cacheblend_kv expects input_ids shape (1, L), got {input_ids.shape}."
            )
        seq_len = int(input_ids.shape[1])
        if not (0 <= span_start < span_end <= seq_len):
            raise ValueError(
                f"Invalid cacheblend span: {span_start=}, {span_end=}, {seq_len=}."
            )
        if raw_kv_position_mode != "rotated":
            raise ValueError(
                "cacheblend entries are post-RoPE at their native positions; "
                f"raw_kv_position_mode must be 'rotated', got {raw_kv_position_mode!r}."
            )
        if history_kv_method:
            raise ValueError(
                "cacheblend is exclusive with history_kv_method "
                f"(got {history_kv_method!r})."
            )
        config = CacheBlendConfig.from_request(cacheblend)
        device = input_ids.device
        positions = torch.arange(
            position_offset,
            position_offset + seq_len,
            dtype=torch.long,
            device=device,
        )
        ops = _Qwen3CacheBlendOps(self)
        if not hasattr(self, "_cacheblend_chunk_cache"):
            # Process/model-local cache: never shared across checkpoints or TP
            # ranks. CPU LRU is bounded independently of the accelerator pool.
            self._cacheblend_chunk_cache = ChunkKVCache(
                int(os.environ.get("SGLANG_CACHEBLEND_CHUNK_CACHE_BYTES", str(256 * 1024 * 1024)))
            )
        out_kv, meta = cacheblend_blend(
            ops, input_ids.view(-1), positions, span_start, span_end, config,
            chunk_cache=self._cacheblend_chunk_cache,
        )
        span_len = span_end - span_start
        raw_key_values = [
            (
                k.reshape(span_len, -1).contiguous(),
                v.reshape(span_len, -1).contiguous(),
            )
            for k, v in out_kv
        ]
        repair_positions = positions[span_start:span_end].view(1, -1).contiguous()
        meta["position_offset"] = int(position_offset)
        return raw_key_values, repair_positions, meta

    @torch.no_grad()
    def generate_pic(self, input_ids, attention_mask, ratio=1, **kwargs):
        """Extract full-length residual-QKV PIC states for one document."""
        if not self.full_length_pic:
            raise ValueError("generate_pic requires a checkpoint with pic_enabled=True.")
        if ratio != 1:
            raise ValueError("Full-length PIC storage requires compression_ratio=1.")

        pic_mask, position_ids = prepare_pic_input(input_ids, attention_mask)
        hidden_states = self.model.embed_tokens(input_ids)
        pic_key_values = []
        for layer in self.model.layers:
            hidden_states, layer_kv = layer.forward_with_pic(
                hidden_states,
                positions=position_ids.squeeze(0),
            )
            pic_key_values.append(layer_kv)
        return pic_key_values, pic_mask, position_ids

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        if hasattr(self, "_cacheblend_chunk_cache"):
            self._cacheblend_chunk_cache.clear()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        if self.enable_c2kv:
            if self.full_length_pic:
                stacked_params_mapping += [
                    ("residual_qkv_proj", "residual_q_proj", "q"),
                    ("residual_qkv_proj", "residual_k_proj", "k"),
                    ("residual_qkv_proj", "residual_v_proj", "v"),
                ]
            else:
                stacked_params_mapping += [
                    ("gist_qkv_proj", "gist_q_proj", "q"),
                    ("gist_qkv_proj", "gist_k_proj", "k"),
                    ("gist_qkv_proj", "gist_v_proj", "v"),
                ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("gist_embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if name == "model.embed_tokens.weight":
                if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                    if "lm_head.weight" in params_dict:
                        param = params_dict["lm_head.weight"]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def load_c2kv_tool_gist_weights(self) -> Dict[str, Any]:
        """Load the second ("tool") gist set from --c2kv-tool-gist-weights.

        Reads only the ``gist_*`` tensors of the source (a full checkpoint or a
        ``c2kv-gist.safetensors`` export package) and maps them onto
        ``tool_gist_qkv_proj`` / ``tool_gist_embed_tokens``.  Every fused
        projection must receive all three shards, otherwise loading fails
        instead of serving a partially initialised encoder.
        """
        from safetensors import safe_open

        from sglang.srt.mem_cache.c2kv_semantics import c2kv_tool_gist_identity

        source = getattr(self, "c2kv_tool_gist_source", None)
        if not source or getattr(self, "tool_gist_cfg", None) is None:
            raise RuntimeError("No C2KV tool gist set is configured on this model")
        if getattr(self, "c2kv_tool_gist_uses_served_t0", False):
            self.c2kv_tool_gist_identity = c2kv_tool_gist_identity(source)
            summary = {
                "source": source,
                "identity": self.c2kv_tool_gist_identity,
                "variant": "T0",
                "compression_domain": "tool",
                "parameter_source": "served_checkpoint",
            }
            logger.info("C2KV tool gist set aliases served T0: %s", summary)
            return summary
        params_dict = dict(self.named_parameters())
        stacked = [
            ("tool_gist_qkv_proj", "gist_q_proj", "q"),
            ("tool_gist_qkv_proj", "gist_k_proj", "k"),
            ("tool_gist_qkv_proj", "gist_v_proj", "v"),
        ]
        expected = {("model.tool_gist_embed_tokens.weight", None)}
        for name in params_dict:
            if ".tool_gist_qkv_proj." in name:
                expected.update((name, shard) for shard in "qkv")
        loaded = set()
        files = _c2kv_gist_weight_files(source)
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if "gist_" not in name:
                        continue
                    target = name if name.startswith("model.") else add_prefix(name, "model")
                    if target.startswith("model.gist_embed_tokens."):
                        target = target.replace(
                            "model.gist_embed_tokens.", "model.tool_gist_embed_tokens."
                        )
                        param = params_dict[target]
                        default_weight_loader(param, handle.get_tensor(name))
                        loaded.add((target, None))
                        continue
                    layer_id = get_layer_id(target)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    for param_name, weight_name, shard_id in stacked:
                        if weight_name not in target:
                            continue
                        target = target.replace(weight_name, param_name)
                        if target not in params_dict:
                            raise ValueError(
                                f"C2KV tool gist tensor {name} has no parameter {target}"
                            )
                        param = params_dict[target]
                        param.weight_loader(param, handle.get_tensor(name), shard_id)
                        loaded.add((target, shard_id))
                        break
                    else:
                        raise ValueError(
                            f"Unexpected gist tensor {name} in C2KV tool gist source {source}"
                        )
        missing = sorted(f"{name}[{shard}]" for name, shard in expected - loaded)
        if missing:
            raise ValueError(
                "C2KV tool gist source is incomplete; missing "
                f"{len(missing)} shards, e.g. {missing[:3]}"
            )
        self.c2kv_tool_gist_identity = c2kv_tool_gist_identity(source)
        summary = {
            "source": source,
            "files": [os.path.basename(path) for path in files],
            "shards": len(loaded),
            "identity": self.c2kv_tool_gist_identity,
            "variant": (self.c2kv_tool_gist_metadata or {}).get("history_memory_variant"),
            "compression_domain": (self.c2kv_tool_gist_metadata or {}).get(
                "history_memory_compression_domain"
            ),
        }
        logger.info("C2KV tool gist set loaded: %s", summary)
        return summary

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]


EntryClass = Qwen3ForCausalLM


class _Qwen3CacheBlendOps:
    """``cacheblend.LayerOps`` over this model's BASE projections.

    The primitives are the repair-extract ones of ``generate_raw_repair_kv``
    (qkv_proj -> qk_norm -> rope, explicit causal attention, o_proj +
    all-reduce + residual + MLP), so a CacheBlend row is computed with exactly
    the arithmetic every other repair entry is; only the token set differs.
    Never touches ``gist_qkv_proj`` (original-token invariance).
    """

    def __init__(self, model: "Qwen3ForCausalLM"):
        self.model = model
        self.layers = model.model.layers
        self.num_layers = len(self.layers)
        self.cache_dtype = model.model.embed_tokens.weight.dtype

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(input_ids.view(1, -1)).squeeze(0)

    def input_norm(self, layer_index: int, hidden_rows: torch.Tensor) -> torch.Tensor:
        return self.layers[layer_index].input_layernorm(hidden_rows)

    def qkv(self, layer_index: int, attn_input: torch.Tensor):
        attn = self.layers[layer_index].self_attn
        qkv, _ = attn.qkv_proj(attn_input)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q, k = apply_qk_norm(
            q=q,
            k=k,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
            head_dim=attn.head_dim,
            alt_stream=attn.alt_stream,
        )
        n = int(attn_input.shape[0])
        return (
            q.reshape(n, attn.num_heads, attn.head_dim),
            k.reshape(n, attn.num_kv_heads, attn.head_dim),
            v.reshape(n, attn.num_kv_heads, attn.head_dim),
        )

    def rope(self, layer_index: int, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        attn = self.layers[layer_index].self_attn
        n = int(q.shape[0])
        q_flat, k_flat = attn.rotary_emb(
            positions, q.reshape(n, -1).contiguous(), k.reshape(n, -1).contiguous()
        )
        return (
            q_flat.reshape(n, attn.num_heads, attn.head_dim),
            k_flat.reshape(n, attn.num_kv_heads, attn.head_dim),
        )

    def rotate_k(
        self, layer_index: int, positions: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        attn = self.layers[layer_index].self_attn
        n = int(k.shape[0])
        # Qwen3's rotary adapter reshapes Q with num_heads and K with
        # num_kv_heads. GQA therefore needs a throwaway Q with query-head shape,
        # rather than zeros_like(K).
        fake_q = k.new_zeros((n, attn.num_heads, attn.head_dim))
        _, k_rot = self.rope(layer_index, positions, fake_q, k)
        return k_rot

    def attention(
        self,
        layer_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        blocked: torch.Tensor,
    ) -> torch.Tensor:
        attn = self.layers[layer_index].self_attn
        num_q = int(q.shape[0])
        num_k = int(k.shape[0])
        q_b = q.reshape(1, num_q, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()
        k_b = k.reshape(1, num_k, attn.num_kv_heads, attn.head_dim).transpose(1, 2).contiguous()
        v_b = v.reshape(1, num_k, attn.num_kv_heads, attn.head_dim).transpose(1, 2).contiguous()
        mask = blocked.reshape(1, 1, num_q, num_k)
        if _is_npu:
            # same ops and the same env switch as generate_raw_repair_kv
            if os.environ.get(
                "C2KV_REPAIR_EXTRACT_ATTN_IMPL",
                "prompt_flash",
            ) == "prompt_flash" and hasattr(torch_npu, "npu_prompt_flash_attention"):
                attn_output = torch_npu.npu_prompt_flash_attention(
                    q_b,
                    k_b,
                    v_b,
                    num_heads=q_b.shape[1],
                    num_key_value_heads=k_b.shape[1],
                    input_layout="BNSD",
                    atten_mask=mask,
                    scale_value=attn.scaling,
                    sparse_mode=0,
                )
            else:
                attn_output = torch_npu.npu_fusion_attention(
                    q_b,
                    k_b,
                    v_b,
                    q_b.shape[1],
                    input_layout="BNSD",
                    atten_mask=mask,
                    scale=attn.scaling,
                    keep_prob=1.0,
                    sparse_mode=0,
                )
            attn_output = _npu_fusion_attention_output(attn_output, q_b.shape)
        else:
            if attn.num_heads != attn.num_kv_heads:
                groups = attn.num_heads // attn.num_kv_heads
                k_run = k_b.repeat_interleave(groups, dim=1)
                v_run = v_b.repeat_interleave(groups, dim=1)
            else:
                k_run, v_run = k_b, v_b
            scores = torch.matmul(q_b.float(), k_run.transpose(-2, -1).float()) * attn.scaling
            scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v_run.dtype)
            attn_output = torch.matmul(probs, v_run)
        return (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(num_q, attn.num_heads * attn.head_dim)
        )

    def post_attention(
        self, layer_index: int, attn_output: torch.Tensor, residual_rows: torch.Tensor
    ) -> torch.Tensor:
        layer = self.layers[layer_index]
        projected, _ = layer.self_attn.o_proj(attn_output)
        projected = tensor_model_parallel_all_reduce(projected)
        hidden = residual_rows + projected
        mlp_input = layer.post_attention_layernorm(hidden)
        return hidden + layer.mlp(mlp_input)

    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        # the deviation must be summed over ALL kv heads (artifact), which
        # under tensor parallelism are sharded across ranks
        if get_tensor_model_parallel_world_size() > 1:
            return tensor_model_parallel_all_reduce(value)
        return value
