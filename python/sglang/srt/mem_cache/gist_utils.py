"""
Gist utility functions for C2KV (Concatenable and Compressible KV Cache).

Builds the custom attention mask, position IDs, and optional residual
connections used during the gist extraction forward pass.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch.nn.attention.flex_attention import create_block_mask

# PyTorch 2.8 accepts kernel_options as a plain dict.
C2KV_KERNEL_OPTIONS = {
    "FORCE_USE_FLEX_ATTENTION": True,
}


def resolve_c2kv_compression_ratio(
    requested_ratio: int, *, full_length_pic: bool
) -> int:
    """Return the storage ratio used by the active C2KV representation."""
    if requested_ratio <= 0:
        raise ValueError("compression_ratio must be greater than 0.")
    return 1 if full_length_pic else requested_ratio


def prepare_pic_input(input_ids, attention_mask):
    """Build token and position metadata for full-length PIC extraction."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "PIC extraction currently requires input_ids with shape (1, seq_len), "
            f"got {tuple(input_ids.shape)}."
        )
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            "PIC attention_mask shape must match input_ids: "
            f"{tuple(attention_mask.shape)} != {tuple(input_ids.shape)}."
        )
    if not bool(attention_mask.bool().all().item()):
        raise ValueError("PIC extraction currently requires an unpadded document.")

    seq_len = input_ids.shape[1]
    if seq_len == 0:
        raise ValueError("PIC extraction requires at least one token.")

    pic_mask = torch.ones((1, seq_len), dtype=torch.bool, device=input_ids.device)
    position_ids = torch.arange(
        seq_len, dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)
    return pic_mask, position_ids


@dataclass
class GistConfig:
    gist_type: str = "dynamic-interleave"
    gist_param: str = "qkv"
    gist_extra_embed_num: int = 1
    gist_token_id: Optional[int] = None
    gist_residual_type: str = "none"
    gist_overlap: int = 0
    hidden_size: int = 4096
    attention_bias: bool = False


def get_prepare_gist_input_func(gist_cfg: GistConfig) -> Callable:
    """
    Returns prepare_gist_input(input_ids, attention_mask, ratio)
    -> (new_attn_mask, gist_mask, position_ids).

    Attention mask layout (True = attend):
        input tokens see each other causally; cannot see gist tokens.
        gist tokens attend their own input chunk plus the first-ratio sink;
        they see gist tokens causally.

    Each gist token attends to its own chunk plus `gist_overlap` preceding
    tokens (clamped to 0), i.e. [max(j*ratio - gist_overlap, 0), (j+1)*ratio).

    Position IDs:
        input token i  -> position i
        gist token j   -> min((j+1)*ratio - 1, seq_len - 1)
    """

    gist_overlap = gist_cfg.gist_overlap

    def prepare_gist_input(input_ids, attention_mask, ratio=4):
        device = input_ids.device
        seq_len = input_ids.shape[1]
        gist_len = math.ceil(seq_len / ratio)
        total_len = seq_len + gist_len

        # --- block_mask for flex_attention ---
        # Mask logic (True = attend):
        #   input-to-input: causal (q_idx >= kv_idx)
        #   input-to-gist: never (input tokens cannot see gist tokens)
        #   gist-to-input: its chunk & sink tokens
        #   gist-to-gist: causal (q_idx >= kv_idx)
        def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            is_q_input = q_idx < seq_len
            is_kv_input = kv_idx < seq_len

            # input query attending input key: causal
            input_to_input = is_q_input & is_kv_input & (q_idx >= kv_idx)
            # input query attending gist key: never
            # gist query attending input key: its chunk & sink tokens
            gist_j = q_idx - seq_len
            # extend the chunk backward by gist_overlap tokens; kv_idx >= 0
            # naturally clamps the lower bound to 0.
            chunk_begin = gist_j * ratio - gist_overlap
            chunk_end = (gist_j + 1) * ratio
            gist_to_input = (~is_q_input) & is_kv_input & (
                ((kv_idx >= chunk_begin) & (kv_idx < chunk_end)) | (kv_idx < ratio)
            )
            # gist query attending gist key: causal
            gist_to_gist = (~is_q_input) & (~is_kv_input) & (q_idx >= kv_idx)

            return input_to_input | gist_to_input | gist_to_gist

        if device.type == "npu":
            # Correctness-first NPU fallback:
            # build a dense boolean attention mask directly and avoid
            # FlexAttention BlockMask / stable argsort on Ascend.
            idx = torch.arange(total_len, device=device, dtype=torch.long)
            q_idx = idx[:, None]
            kv_idx = idx[None, :]

            is_q_input = q_idx < seq_len
            is_kv_input = kv_idx < seq_len

            # input -> input: causal
            input_to_input = is_q_input & is_kv_input & (q_idx >= kv_idx)

            # gist -> input: own chunk + sink tokens
            gist_j = q_idx - seq_len
            chunk_begin = gist_j * ratio - gist_overlap
            chunk_end = (gist_j + 1) * ratio

            gist_to_input = (~is_q_input) & is_kv_input & (
                ((kv_idx >= chunk_begin) & (kv_idx < chunk_end)) | (kv_idx < ratio)
            )

            # gist -> gist: causal
            gist_to_gist = (~is_q_input) & (~is_kv_input) & (q_idx >= kv_idx)

            # Shape: (1, 1, total_len, total_len)
            block_mask = (input_to_input | gist_to_input | gist_to_gist).unsqueeze(
                0
            ).unsqueeze(0)

        else:
            block_mask = create_block_mask(
                mask_mod,
                B=1,
                H=None,
                Q_LEN=total_len,
                KV_LEN=total_len,
                device=device,
            )

        # --- gist_mask (1, gist_len) ---
        gist_mask = torch.ones((1, gist_len), dtype=torch.bool, device=device)

        # --- position_ids (1, total_len) ---
        input_pos = torch.arange(seq_len, dtype=torch.long, device=device)
        gist_pos = torch.tensor(
            [min((j + 1) * ratio - 1, seq_len - 1) for j in range(gist_len)],
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.cat([input_pos, gist_pos], dim=0).unsqueeze(0)

        return block_mask, gist_mask, position_ids

    return prepare_gist_input


def _apply_gist_residual_interleave(
    tokens_tensor: torch.Tensor, gist_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    ratio = kwargs["ratio"]
    batch_size, seq_length, hidden_size = tokens_tensor.shape
    pad_length = seq_length % ratio
    nopad_length = seq_length - pad_length
    mean_tensor = tokens_tensor[:, :nopad_length].reshape(
        batch_size, -1, ratio, hidden_size
    ).mean(dim=2)
    if pad_length != 0:
        pad_mean = tokens_tensor[:, nopad_length:].mean(dim=1, keepdim=True)
        mean_tensor = torch.cat([mean_tensor, pad_mean], dim=1)
    return mean_tensor + gist_tensor


def _apply_none(input_hidden, gist_hidden, **kwargs):
    return gist_hidden


def get_apply_gist_residual_func(gist_cfg: GistConfig, layer_idx: int) -> Callable:
    """
    Returns apply_gist_residual(input_hidden, gist_hidden, **kwargs) -> gist_hidden.

    Residual types:
        "none"       -> identity on gist_hidden
        "mean"       -> chunk-mean of input + gist_hidden at every layer
        "embed-mean" -> chunk-mean of input + gist_hidden at layer 0 only
    """
    residual_type = gist_cfg.gist_residual_type

    if residual_type == "embed-mean":
        if layer_idx != 0: # only apply at layer 0
            return _apply_none
        return _apply_gist_residual_interleave

    if residual_type == "mean":
        return _apply_gist_residual_interleave

    return _apply_none
