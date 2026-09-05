"""
C2KV KV injection helper.

Reads a stored gist entry, applies RoPE at the correct absolute positions,
and writes K/V tensors into the engine's KV pool.
"""

import os
from typing import List, Optional

import torch

from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
from sglang.srt.mem_cache.c2kv_pool import C2KVEntry, C2KVPool
from sglang.srt.mem_cache.c2kv_semantics import validate_rope_position_range


def _validate_rope_positions(position_ids: torch.Tensor, table_size: int) -> None:
    if position_ids.numel() == 0:
        raise ValueError("C2KV_ROPE_POSITION_OUT_OF_RANGE: no positions supplied")
    validate_rope_position_range(
        int(position_ids.min().item()),
        int(position_ids.max().item()),
        int(table_size),
    )


def inject_c2kv_gist(
    entry: C2KVEntry,
    c2kv_pool: C2KVPool,
    position_cursor: int,
    loc: torch.Tensor,
    token_to_kv_pool,
    attn_layers: List,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool = True,
) -> None:

    gist_len = entry.gist_len

    if loc.numel() != gist_len:
        raise ValueError(
            f"C2KV loc length mismatch: "
            f"loc.numel()={loc.numel()} != {gist_len=}"
        )

    if c2kv_pool.num_layers != len(attn_layers):
        raise ValueError(
            "C2KV layer count mismatch: "
            f"{c2kv_pool.num_layers=} != {len(attn_layers)=}"
        )

    # ---------------------------------------------------------
    # Position information
    # ---------------------------------------------------------

    gist_pos = c2kv_pool.get_position_ids(entry)

    abs_pos = position_cursor + gist_pos
    _validate_rope_positions(abs_pos, cos_sin_cache.shape[0])

    rotary_dim = cos_sin_cache.shape[1]
    half_dim = rotary_dim // 2

    cos = cos_sin_cache[
        abs_pos,
        :half_dim,
    ]

    sin = cos_sin_cache[
        abs_pos,
        half_dim:,
    ]

    head_dim = half_dim * 2

    # ---------------------------------------------------------
    # Debug dump
    # ---------------------------------------------------------

    dump_path = os.environ.get(
        "C2KV_DEBUG_INJECT_DUMP"
    )

    debug_obj = None

    if dump_path:
        debug_obj = {
            "position_cursor": int(position_cursor),
            "gist_len": int(gist_len),
            "original_seq_len": int(entry.original_seq_len),
            "gist_pos": gist_pos.detach().cpu(),
            "abs_pos": abs_pos.detach().cpu(),
            "loc": loc.detach().cpu(),
            "layers": {},
        }

    # ---------------------------------------------------------
    # Inject each layer
    # ---------------------------------------------------------

    for layer_idx in range(
        c2kv_pool.num_layers
    ):

        # IMPORTANT:
        # This is already after C2KVPool.store/get.
        k_pre, v_pre = c2kv_pool.get_layer_kv(
            entry,
            layer_idx,
        )

        if k_pre.shape[2] != head_dim:
            raise ValueError(
                f"C2KV head_dim mismatch "
                f"at layer {layer_idx}: "
                f"{k_pre.shape[2]} != {head_dim}"
            )

        # Apply absolute-position RoPE.
        k_rotated = apply_rotary_emb(
            k_pre,
            cos,
            sin,
            is_neox_style,
        )

        layer = attn_layers[layer_idx]

        # Write into SGLang main KV cache.
        token_to_kv_pool.set_kv_buffer(
            layer=layer,
            loc=loc,
            cache_k=k_rotated,
            cache_v=v_pre,
        )

        # -----------------------------------------------------
        # Read the actual main KV cache back.
        # -----------------------------------------------------

        if dump_path:

            if (
                hasattr(torch, "npu")
                and torch.npu.is_available()
            ):
                torch.npu.synchronize()

            layer_id = layer.layer_id

            k_buffer = (
                token_to_kv_pool.get_key_buffer(
                    layer_id
                )
            )

            v_buffer = (
                token_to_kv_pool.get_value_buffer(
                    layer_id
                )
            )

            # Handles normal Ascend layout:
            #
            # [page, page_size, Hkv, D]
            #
            # and FIA-like layouts by flattening all
            # physical token dimensions.
            k_flat = k_buffer.reshape(
                -1,
                k_buffer.shape[-2],
                k_buffer.shape[-1],
            )

            v_flat = v_buffer.reshape(
                -1,
                v_buffer.shape[-2],
                v_buffer.shape[-1],
            )

            loc_long = loc.long()

            k_readback = (
                k_flat[loc_long]
                .contiguous()
                .clone()
            )

            v_readback = (
                v_flat[loc_long]
                .contiguous()
                .clone()
            )

            debug_obj["layers"][layer_idx] = {
                # after C2KVPool.store/get
                "k_pre": (
                    k_pre.detach()
                    .cpu()
                    .clone()
                ),
                "v_pre": (
                    v_pre.detach()
                    .cpu()
                    .clone()
                ),

                # immediately before main KV cache write
                "k_rotated": (
                    k_rotated.detach()
                    .cpu()
                    .clone()
                ),

                # actual main KV cache contents
                "k_readback": (
                    k_readback.detach()
                    .cpu()
                ),
                "v_readback": (
                    v_readback.detach()
                    .cpu()
                ),
            }

    # Diagnostic: make all injected KV writes globally visible
    # before the scheduler can launch the next prefill round.
    #
    # If enabling this fixes C2KV attention, the bug is a
    # scheduler/forward-stream ordering issue rather than KV values.
    if (
        os.environ.get("C2KV_DEBUG_FORCE_SYNC") == "1"
        and hasattr(torch, "npu")
        and torch.npu.is_available()
    ):
        torch.npu.synchronize()
        print(
            "[C2KV FORCE SYNC]",
            {
                "gist_len": int(gist_len),
                "position_cursor": int(position_cursor),
            },
            flush=True,
        )

    if dump_path:
        torch.save(
            debug_obj,
            dump_path,
        )

        print(
            "[C2KV DEBUG INJECT] "
            f"saved injection dump to {dump_path}",
            flush=True,
        )


def inject_c2kv_stored_kv(
    entry: C2KVEntry,
    c2kv_pool: C2KVPool,
    *,
    loc: torch.Tensor,
    token_to_kv_pool,
    attn_layers: List,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool = True,
    position_ids: Optional[torch.Tensor] = None,
) -> None:
    """Inject a generic stored KV entry into the active paged KV cache.

    Repair entries may already contain K with the original absolute RoPE phase
    (`entry.already_rotated=True`). In that case K is copied verbatim and
    `position_ids` must be None. Entries stored pre-RoPE (the default for
    model_prefill repair extraction and for neutral/sham entries) get RoPE
    applied exactly once, at `position_ids` when given (append_tail placement)
    or at the stored absolute position ids otherwise.
    """

    token_len = entry.token_len
    if loc.numel() != token_len:
        raise ValueError(
            f"C2KV repair loc length mismatch: loc.numel()={loc.numel()} != {token_len=}"
        )
    if c2kv_pool.num_layers != len(attn_layers):
        raise ValueError(
            "C2KV repair layer count mismatch: "
            f"{c2kv_pool.num_layers=} != {len(attn_layers)=}"
        )

    if position_ids is not None:
        if entry.already_rotated:
            raise ValueError(
                "C2KV repair entry was stored post-RoPE (already_rotated=True); "
                "it cannot be re-placed at new positions. Re-extract it through "
                "the model_prefill path for append_tail placement."
            )
        if position_ids.numel() != token_len:
            raise ValueError(
                f"C2KV repair position override length mismatch: "
                f"{position_ids.numel()} != {token_len=}"
            )
        abs_pos = position_ids.to(device=cos_sin_cache.device, dtype=torch.long)
    else:
        abs_pos = c2kv_pool.get_position_ids(entry)
    rotary_dim = cos_sin_cache.shape[1]
    half_dim = rotary_dim // 2
    head_dim = half_dim * 2
    if entry.already_rotated:
        cos = sin = None
    else:
        _validate_rope_positions(abs_pos, cos_sin_cache.shape[0])
        cos = cos_sin_cache[abs_pos, :half_dim]
        sin = cos_sin_cache[abs_pos, half_dim:]

    for layer_idx in range(c2kv_pool.num_layers):
        k_stored, v_stored = c2kv_pool.get_layer_kv(entry, layer_idx)
        if k_stored.shape[2] != head_dim:
            raise ValueError(
                f"C2KV repair head_dim mismatch at layer {layer_idx}: "
                f"{k_stored.shape[2]} != {head_dim}"
            )
        cache_k = (
            k_stored
            if entry.already_rotated
            else apply_rotary_emb(k_stored, cos, sin, is_neox_style)
        )
        layer = attn_layers[layer_idx]
        token_to_kv_pool.set_kv_buffer(
            layer=layer,
            loc=loc,
            cache_k=cache_k,
            cache_v=v_stored,
        )

    if (
        os.environ.get("C2KV_DEBUG_FORCE_SYNC") == "1"
        and hasattr(torch, "npu")
        and torch.npu.is_available()
    ):
        torch.npu.synchronize()
