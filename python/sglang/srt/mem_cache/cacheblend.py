"""CacheBlend (Yao et al., EuroSys 2025, arXiv 2405.16444) as a C2KV repair
extraction mode: chunk-KV reuse with selective recomputation.

This module is the model-agnostic engine.  ``models/qwen3.py`` supplies the
per-layer primitives (``LayerOps``) and the scheduler stores the result as ONE
repair entry (``C2KVPool.store_repair``) that a chat request places
``in_place`` at the history span's absolute positions, exactly like the
history-KV eviction entries.

The algorithm follows the official EuroSys artifact
(``vllm_blend/vllm/model_executor/models/llama.py`` +
``attention/backends/xformers.py``, LMCache/CacheBlend @ 55ad026), not the
paper prose, wherever the two differ:

1. Every chunk's KV is computed STANDALONE (the chunk's tokens alone, positions
   0..n-1, all layers) and its K is kept pre-RoPE (artifact ``collect`` mode,
   ``hack_kv`` taken before ``rotary_emb``).  At blend time the old K is
   rotated at the chunk's absolute positions in the concatenated prompt
   (artifact: ``rotary_emb(org_pos, fake_q, old_kv[0])``).
2. Layers ``0 .. check_layer`` (artifact ``check_layers=[1]``: layer 0 =
   status 0, layer 1 = status 1) run a FULL forward over every token of the
   input; their K/V are fresh for all tokens.
3. At ``check_layer`` the per-token deviation is ``sum((fresh - old) ** 2)``
   over (kv_heads, head_dim) of V (artifact) or K (``metric="k"``, the LMCache
   lineage), over the reused span only; the top ``int(span_len * ratio)``
   tokens (artifact ``topk_num``) plus every non-reused token (prologue before
   the span, suffix after it -- the artifact's ``last_len`` suffix) form the
   recompute set S.
4. Layers ``> check_layer`` (status 2) run only the rows in S: fresh q/k/v for
   S, K/V of the span = old chunk KV with the S rows overwritten by the fresh
   values (artifact ``key_old[imp_indices] = key``), attention of the S
   queries over the whole prefix.
5. Attention mask for the subset queries: ``causal`` (query at position p sees
   keys at positions <= p -- the paper's semantics) or ``bottom_right`` (the
   artifact's ``LowerTriangularFromBottomRightMask``, which lets a selected
   token attend to later keys; kept only to reproduce the artifact).

Deltas that are NOT the artifact's, all documented in
``c2kv/c2kv_serving_semantics.md`` section 10: chunks are the bench's history
docs (rendered chat messages) or a fixed token grid, never the artifact's
per-dataset passages; the system/tool prologue is prefilled fresh in the same
forward (the artifact caches it as chunk 0); the chunk cache is materialised
per request instead of being loaded from storage (identical values, no
wall-clock claim).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import torch

CACHEBLEND_METRICS = ("v", "k")
CACHEBLEND_MASKS = ("causal", "bottom_right")


@dataclass
class CacheBlendConfig:
    recomp_ratio: float = 0.16
    check_layer: int = 1
    metric: str = "v"
    mask: str = "causal"
    chunk_tokens: Optional[int] = None
    chunk_bounds: Optional[List[Tuple[int, int]]] = None

    @classmethod
    def from_request(cls, raw: Optional[Dict[str, Any]]) -> "CacheBlendConfig":
        raw = dict(raw or {})
        cfg = cls(
            recomp_ratio=float(
                raw["recomp_ratio"] if raw.get("recomp_ratio") is not None else 0.16
            ),
            check_layer=int(
                raw["check_layer"] if raw.get("check_layer") is not None else 1
            ),
            metric=str(raw.get("metric") or "v").lower(),
            mask=str(raw.get("mask") or "causal").lower(),
            chunk_tokens=(
                int(raw["chunk_tokens"]) if raw.get("chunk_tokens") is not None else None
            ),
            chunk_bounds=(
                [(int(a), int(b)) for a, b in raw["chunk_bounds"]]
                if raw.get("chunk_bounds") is not None
                else None
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not (0.0 <= self.recomp_ratio <= 1.0):
            raise ValueError(
                f"cacheblend recomp_ratio must be in [0, 1], got {self.recomp_ratio}"
            )
        if self.check_layer < 0:
            raise ValueError(
                f"cacheblend check_layer must be >= 0, got {self.check_layer}"
            )
        if self.metric not in CACHEBLEND_METRICS:
            raise ValueError(
                f"cacheblend metric must be one of {CACHEBLEND_METRICS}, got {self.metric!r}"
            )
        if self.mask not in CACHEBLEND_MASKS:
            raise ValueError(
                f"cacheblend mask must be one of {CACHEBLEND_MASKS}, got {self.mask!r}"
            )
        if self.chunk_tokens is not None and self.chunk_tokens < 1:
            raise ValueError(
                f"cacheblend chunk_tokens must be >= 1, got {self.chunk_tokens}"
            )

    def as_meta(self) -> Dict[str, Any]:
        return {
            "recomp_ratio": self.recomp_ratio,
            "check_layer": self.check_layer,
            "metric": self.metric,
            "mask": self.mask,
            "chunk_tokens": self.chunk_tokens,
        }


def resolve_chunk_bounds(
    span_len: int,
    chunk_bounds: Optional[Sequence[Tuple[int, int]]] = None,
    chunk_tokens: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Chunk boundaries RELATIVE to the span, contiguous and covering it.

    ``chunk_bounds`` (explicit, e.g. one per history doc) wins; otherwise a
    fixed grid of ``chunk_tokens``; otherwise the whole span is one chunk.
    """
    if span_len <= 0:
        raise ValueError("cacheblend span must be non-empty")
    if chunk_bounds:
        bounds = [(int(a), int(b)) for a, b in chunk_bounds]
        expected = 0
        for start, end in bounds:
            if start != expected:
                raise ValueError(
                    f"cacheblend chunk_bounds must be contiguous from 0: got {bounds}"
                )
            if end <= start:
                raise ValueError(f"cacheblend chunk {(start, end)} is empty")
            expected = end
        if expected != span_len:
            raise ValueError(
                f"cacheblend chunk_bounds cover {expected} tokens, span has {span_len}"
            )
        return bounds
    if chunk_tokens:
        size = int(chunk_tokens)
        return [(s, min(s + size, span_len)) for s in range(0, span_len, size)]
    return [(0, span_len)]


def deviation_scores(fresh: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """Per-token squared deviation, summed over every trailing dim (artifact:
    ``torch.sum((value - value_old) ** 2, dim=[1, 2])``).  float32."""
    if fresh.shape != old.shape:
        raise ValueError(
            "cacheblend deviation: shape mismatch "
            f"{tuple(fresh.shape)} vs {tuple(old.shape)}"
        )
    diff = fresh.float() - old.float()
    return diff.pow(2).flatten(1).sum(dim=1)


def recompute_budget(span_len: int, ratio: float) -> int:
    """Artifact: ``topk_num = int(total_len * recomp_ratio)``; floored at 1 for
    a non-empty span so ratio > 0 always recomputes something, and 0 when the
    ratio is exactly 0 (pure reuse)."""
    if span_len <= 0:
        return 0
    if ratio <= 0.0:
        return 0
    return max(1, min(span_len, int(span_len * ratio)))


def select_recompute_tokens(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Indices (sorted, int64, on ``scores.device``) of the ``budget`` highest
    deviation tokens (artifact: ``torch.topk`` then ``torch.sort``)."""
    n = int(scores.numel())
    budget = max(0, min(n, int(budget)))
    if budget == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    top = torch.topk(scores, k=budget, largest=True).indices
    return torch.sort(top).values.to(torch.long)


def subset_causal_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    mode: str = "causal",
) -> torch.Tensor:
    """Boolean (S, L) mask, True = BLOCKED, for S subset queries over L keys.

    ``causal``: key blocked when key_position > query_position (exact
    causality for scattered queries).  ``bottom_right``: the artifact's
    LowerTriangularFromBottomRightMask -- query i (0-based in the subset) sees
    keys with index <= L - S + i, which for a scattered subset admits keys
    AFTER the query's own position.
    """
    if mode not in CACHEBLEND_MASKS:
        raise ValueError(f"unknown cacheblend mask mode {mode!r}")
    q = query_positions.to(torch.long).view(-1, 1)
    if mode == "causal":
        k = key_positions.to(torch.long).view(1, -1)
        return k > q
    s = int(q.shape[0])
    total = int(key_positions.numel())
    key_index = torch.arange(total, device=query_positions.device).view(1, -1)
    limit = (torch.arange(s, device=query_positions.device) + (total - s)).view(-1, 1)
    return key_index > limit


def blend_rows(
    old: torch.Tensor, fresh_rows: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """``old`` with rows ``indices`` replaced by ``fresh_rows`` (artifact status
    2: ``key_old[imp_indices] = key``).  Returns a new tensor."""
    out = old.clone()
    if int(indices.numel()):
        out[indices] = fresh_rows.to(out.dtype)
    return out


class LayerOps(Protocol):
    """Per-layer primitives the engine needs; supplied by the model."""

    num_layers: int

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor: ...

    def input_norm(self, layer_index: int, hidden_rows: torch.Tensor) -> torch.Tensor: ...

    def qkv(
        self, layer_index: int, attn_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (q (n, Hq, D), k_pre_rope (n, Hkv, D), v (n, Hkv, D)), after the
        QK norm, BEFORE rotary.  Base projections only."""
        ...

    def rope(
        self, layer_index: int, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate (q, k) at ``positions``; the engine passes clones."""
        ...

    def rotate_k(
        self, layer_index: int, positions: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        """Rotate K without assuming that query and KV head counts match."""
        ...

    def attention(
        self,
        layer_index: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        blocked: torch.Tensor,
    ) -> torch.Tensor:
        """q (S, Hq, D), k/v (L, Hkv, D), blocked (S, L) bool -> (S, Hq*D)."""
        ...

    def post_attention(
        self, layer_index: int, attn_output: torch.Tensor, residual_rows: torch.Tensor
    ) -> torch.Tensor:
        """o_proj (+ TP all-reduce) + residual + post-norm + MLP -> hidden rows."""
        ...

    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Sum across tensor-parallel ranks (identity when TP == 1)."""
        ...


def _rotate_k_only(
    ops: LayerOps, layer_index: int, positions: torch.Tensor, k_pre: torch.Tensor
) -> torch.Tensor:
    """Rotate pre-RoPE K through the model-owned GQA-aware adapter."""
    return ops.rotate_k(layer_index, positions, k_pre.clone())


@torch.no_grad()
def chunk_kv(
    ops: LayerOps, chunk_ids: torch.Tensor
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Standalone prefill of one chunk (positions 0..n-1).  Returns per layer
    ``(k_pre_rope (n, Hkv, D), v (n, Hkv, D))`` -- the artifact's chunk cache."""
    n = int(chunk_ids.numel())
    positions = torch.arange(n, dtype=torch.long, device=chunk_ids.device)
    blocked = subset_causal_mask(positions, positions, "causal")
    hidden = ops.embed(chunk_ids)
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for li in range(ops.num_layers):
        attn_input = ops.input_norm(li, hidden)
        q, k_pre, v = ops.qkv(li, attn_input)
        out.append((k_pre.clone(), v.clone()))
        q, k = ops.rope(li, positions, q.clone(), k_pre.clone())
        attn_out = ops.attention(li, q, k, v, blocked)
        hidden = ops.post_attention(li, attn_out, hidden)
    return out


@torch.no_grad()
def blend(
    ops: LayerOps,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    span_start: int,
    span_end: int,
    config: CacheBlendConfig,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    """The CacheBlend forward over ``input_ids`` (1-D, L tokens) at absolute
    ``positions`` (L).  ``[span_start, span_end)`` is the reused span; tokens
    before it (system/tool prologue) and after it (suffix) are always fresh.

    Returns per layer ``(k_rotated (span_len, Hkv, D), v (span_len, Hkv, D))``
    for the span -- K post-RoPE at its absolute positions, i.e. the
    ``rotated`` storage form -- plus the accounting meta.
    """
    if (
        input_ids.ndim != 1
        or positions.ndim != 1
        or input_ids.numel() != positions.numel()
    ):
        raise ValueError(
            "cacheblend blend expects 1-D input_ids and positions of equal length"
        )
    total = int(input_ids.numel())
    if not (0 <= span_start < span_end <= total):
        raise ValueError(
            f"invalid cacheblend span [{span_start}, {span_end}) for {total} tokens"
        )
    span_len = span_end - span_start
    num_layers = int(ops.num_layers)
    if config.check_layer >= num_layers:
        raise ValueError(
            f"cacheblend check_layer {config.check_layer} >= num_layers {num_layers}"
        )
    bounds = resolve_chunk_bounds(span_len, config.chunk_bounds, config.chunk_tokens)
    device = input_ids.device

    # 1. chunk cache: standalone KV per chunk, concatenated over the span
    per_chunk = [
        chunk_kv(ops, input_ids[span_start + a : span_start + b]) for a, b in bounds
    ]
    old_k: List[torch.Tensor] = []
    old_v: List[torch.Tensor] = []
    for li in range(num_layers):
        old_k.append(torch.cat([kv[li][0] for kv in per_chunk], dim=0))
        old_v.append(torch.cat([kv[li][1] for kv in per_chunk], dim=0))
    del per_chunk

    span_positions = positions[span_start:span_end]
    all_rows = torch.arange(total, dtype=torch.long, device=device)
    outside_rows = torch.cat([all_rows[:span_start], all_rows[span_end:]])

    hidden = ops.embed(input_ids)
    rows = all_rows  # rows of the FULL sequence currently carried forward
    selected_rel: Optional[torch.Tensor] = None
    scores_meta: Dict[str, Any] = {}
    out_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for li in range(num_layers):
        row_positions = positions[rows]
        attn_input = ops.input_norm(li, hidden)
        q, k_pre, v = ops.qkv(li, attn_input)
        q, k = ops.rope(li, row_positions, q.clone(), k_pre.clone())

        if li <= config.check_layer:
            # status 0 / 1: every token is fresh at this layer
            k_full, v_full = k, v
            out_kv.append(
                (
                    k_full[span_start:span_end].clone(),
                    v_full[span_start:span_end].clone(),
                )
            )
            if li == config.check_layer:
                old_k_rot = _rotate_k_only(ops, li, span_positions, old_k[li])
                if config.metric == "v":
                    scores = deviation_scores(v[span_start:span_end], old_v[li])
                else:
                    scores = deviation_scores(k[span_start:span_end], old_k_rot)
                scores = ops.all_reduce_sum(scores)
                budget = recompute_budget(span_len, config.recomp_ratio)
                selected_rel = select_recompute_tokens(scores, budget)
                scores_meta = {
                    "deviation_mean": float(scores.mean().item()),
                    "deviation_max": float(scores.max().item()),
                    "deviation_selected_min": (
                        float(scores[selected_rel].min().item())
                        if int(selected_rel.numel())
                        else None
                    ),
                }
                q_rows = torch.sort(
                    torch.cat([outside_rows, span_start + selected_rel])
                ).values
                q = q[q_rows]
                residual = hidden[q_rows]
                blocked = subset_causal_mask(positions[q_rows], positions, config.mask)
            else:
                q_rows = rows
                residual = hidden
                blocked = subset_causal_mask(row_positions, positions, "causal")
            attn_out = ops.attention(li, q, k_full, v_full, blocked)
            hidden = ops.post_attention(li, attn_out, residual)
            rows = q_rows
        else:
            # status 2: only the recompute rows are carried; the span's KV is
            # the chunk cache with the recomputed rows overwritten
            old_k_rot = _rotate_k_only(ops, li, span_positions, old_k[li])
            in_span = (rows >= span_start) & (rows < span_end)
            span_sel = rows[in_span] - span_start
            k_span = blend_rows(old_k_rot, k[in_span], span_sel)
            v_span = blend_rows(old_v[li], v[in_span], span_sel)
            before = rows < span_start
            after = rows >= span_end
            k_full = torch.cat([k[before], k_span, k[after]], dim=0)
            v_full = torch.cat([v[before], v_span, v[after]], dim=0)
            out_kv.append((k_span, v_span))
            blocked = subset_causal_mask(row_positions, positions, config.mask)
            attn_out = ops.attention(li, q, k_full, v_full, blocked)
            hidden = ops.post_attention(li, attn_out, hidden)

    if selected_rel is None:  # check_layer == num_layers - 1: no status-2 layer
        selected_rel = torch.empty(0, dtype=torch.long, device=device)
    meta: Dict[str, Any] = {
        "kv_reuse_method": "cacheblend",
        "requested_span_tokens": span_len,
        "chunk_count": len(bounds),
        "chunk_bounds": [list(b) for b in bounds],
        "recomputed_tokens": int(selected_rel.numel()),
        "recomputed_relative_indices": [int(i) for i in selected_rel.tolist()],
        "fresh_outside_tokens": int(outside_rows.numel()),
        "effective_recomp_ratio": (
            float(selected_rel.numel()) / float(span_len) if span_len else 0.0
        ),
    }
    meta.update(config.as_meta())
    meta.update(scores_meta)
    return out_kv, meta
