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
    gather_paired_kv,
    require_rotated_headwise_storage,
    select_h2o_prefill_indices,
    select_snapkv_indices,
    select_streamingllm_indices,
    summarize_headwise_indices,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.mem_cache.cacheblend import CacheBlendConfig
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
                quant_config=quant_config,
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
                prefix=add_prefix(c2kv_proj_name, prefix),
            )
            setattr(self, c2kv_proj_name, c2kv_proj)
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
        qkv_gist, _ = self.gist_qkv_proj(hidden_states)
        sizes = [self.q_size, self.kv_size, self.kv_size]
        base_parts = qkv.split(sizes, dim=-1)
        gist_parts = qkv_gist.split(sizes, dim=-1)
        sel = mask.to(qkv.device).view(-1, 1)
        merged = [
            torch.where(sel, gist_t, base_t) if name in parts else base_t
            for name, base_t, gist_t in zip("qkv", base_parts, gist_parts)
        ]
        return torch.cat(merged, dim=-1)

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
            # The first runtime-eviction implementation scores a full history
            # prefill round. Prefix scoring against cached K/V can be added
            # later, but silently mixing the two would corrupt token indices.
            if prefix_len != 0:
                continue

            history_start = int(config.get("history_start") or 0)
            history_end = int(config.get("history_end") or 0)
            if not (0 <= history_start < history_end <= extend_len):
                continue

            method = str(config.get("method") or "").strip().lower()
            if method in {"", "streamingllm"}:
                continue
            recent_window = max(1, int(config.get("history_kv_recent_window") or 64))
            q_end = history_end
            q_start = max(0, q_end - recent_window)
            if q_start >= q_end:
                continue

            q_req = q[token_start:token_end].view(
                extend_len, self.num_heads, self.head_dim
            ).transpose(0, 1).contiguous()
            k_req = k[token_start:token_end].view(
                extend_len, self.num_kv_heads, self.head_dim
            ).transpose(0, 1).contiguous()
            if self.num_heads != self.num_kv_heads:
                groups = self.num_heads // self.num_kv_heads
                k_score = k_req.repeat_interleave(groups, dim=0)
            else:
                k_score = k_req

            q_window = q_req[:, q_start:q_end, :]
            k_all = k_score[:, :history_end, :]
            logits = torch.matmul(
                q_window.float(),
                k_all.transpose(-2, -1).float(),
            ) * self.scaling
            q_pos = flat_positions[token_start + q_start : token_start + q_end].to(
                logits.device
            ).view(1, -1, 1)
            k_pos = flat_positions[token_start : token_start + history_end].to(
                logits.device
            ).view(1, 1, -1)
            logits = logits.masked_fill(k_pos > q_pos, float("-inf"))
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            layer_score = probs[:, :, history_start:history_end].sum(dim=(0, 1))

            req_pool_idx = int(forward_batch.req_pool_indices[batch_idx].item())
            entry = score_store.setdefault(
                req_pool_idx,
                {
                    "method": method,
                    "history_start": history_start,
                    "history_end": history_end,
                    "history_len": history_end - history_start,
                    "layers": [],
                },
            )
            entry["layers"].append(layer_score.detach().cpu())

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
        use_aiter_fused = (
            self.use_fused_qk_norm_mrope
            and forward_batch.forward_mode.is_decode()
            and getattr(forward_batch, "c2kv_use_gist_projection", None) is None
            and get_global_server_args().rl_on_policy_target is None
        )

        if use_aiter_fused:
            q, k, v = self.forward_prepare_aiter_fused_mrope(
                positions, hidden_states, forward_batch
            )
            save_kv_cache = False
        elif (
            getattr(forward_batch, "c2kv_use_gist_projection", None) is not None
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

        qkv_gist, _ = self.gist_qkv_proj(gist_hidden)
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
            gist_cfg.gist_extra_embed_num, config.hidden_size
        )
        self.prepare_gist_input = get_prepare_gist_input_func(gist_cfg)
        return gist_cfg


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
    def generate_gist(self, input_ids, attention_mask, ratio=4, **kwargs):
        """
        Run the gist extraction pass for one document.

        Args:
            input_ids:       (1, seq_len) int64 on GPU
            attention_mask:  (1, seq_len) bool on GPU
            ratio:           compression ratio; gist_len = ceil(seq_len / ratio)

        Returns:
            gist_key_values: List[(K, V)] per layer, each (gist_len, kv_size) float,
                             pre-RoPE. TP-local.
            gist_mask:       (1, gist_len) bool
            gist_position_ids: (1, gist_len) int64
        """
        block_mask, gist_mask, position_ids = self.model.prepare_gist_input(
            input_ids, attention_mask, ratio=ratio
        )
        gist_len = gist_mask.shape[1]
        device = input_ids.device

        gist_embed = self.model.gist_embed_tokens(
            torch.zeros((1, gist_len), dtype=torch.long, device=device)
        )
        inputs_embeds = torch.cat(
            [self.model.embed_tokens(input_ids), gist_embed], dim=1
        )

        hidden_states = inputs_embeds
        gist_key_values = []
        for layer_idx, layer in enumerate(self.model.layers):
            layer_residual = get_apply_gist_residual_func(self.gist_cfg, layer_idx)
            hidden_states, layer_kv = layer.forward_with_gist(
                hidden_states,
                gist_mask,
                positions=position_ids.squeeze(0),
                attention_mask=block_mask,
                apply_gist_residual=layer_residual,
                ratio=ratio,
            )
            gist_key_values.append(layer_kv)

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
        history_method = (history_kv_method or "").strip().lower()
        if history_method == "snapkv":
            history_method = "snapkv_persistent"
        if history_method == "pyramid":
            history_method = "pyramidkv"
        require_rotated_headwise_storage(history_method, raw_kv_position_mode)
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

            if history_method in HEADWISE_HISTORY_KV_METHODS:
                if history_method == "h2o":
                    score_query_start = 0
                else:
                    observation_window = min(snap_recent_window, seq_len)
                    score_query_start = seq_len - observation_window
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

        history_meta = None
        if history_method:
            if history_method not in {
                "streamingllm",
                "h2o",
                "snapkv_persistent",
                "snapkv_refresh",
                "pyramidkv",
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

            def _unique_sorted(indices: Iterable[int]) -> List[int]:
                return sorted({int(i) for i in indices if 0 <= int(i) < requested_span_tokens})

            if history_method == "streamingllm":
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
                low_budget = max(1, min(requested_span_tokens, int(round(target_tokens * 1.5))))
                high_budget = max(1, min(requested_span_tokens, int(round(target_tokens * 0.5))))
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
        out_kv, meta = cacheblend_blend(
            ops, input_ids.view(-1), positions, span_start, span_end, config
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
