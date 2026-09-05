"""Unit tests for the CacheBlend engine (``srt/mem_cache/cacheblend.py``).

The engine is exercised with a tiny synthetic decoder that implements the
``LayerOps`` protocol in plain torch, so the algebra is checked without a
checkpoint or a serving stack:

* ``recomp_ratio = 1.0`` (every span token recomputed) must reproduce the
  dense full-prefill KV exactly at every layer -- selective recompute with
  a full budget IS the dense forward;
* ``recomp_ratio = 0.0`` (pure reuse) must return the chunk cache, rotated
  to the absolute positions, at every layer past the check layer, and the
  dense KV at the layers up to and including it;
* the recompute set has the artifact's size ``int(span_len * ratio)``, is
  sorted, and the prologue/suffix rows are never reused;
* helper semantics (chunk bounds, deviation, masks, budget).
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from typing import List, Tuple

import pytest
import torch

try:  # installed sglang
    from sglang.srt.mem_cache import cacheblend as cb
except Exception:  # bare checkout: load the module by path (no sglang deps)
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PATH = os.path.normpath(
        os.path.join(_HERE, "..", "..", "..", "python", "sglang", "srt", "mem_cache", "cacheblend.py")
    )
    _spec = importlib.util.spec_from_file_location("cacheblend_under_test", _PATH)
    cb = importlib.util.module_from_spec(_spec)
    sys.modules["cacheblend_under_test"] = cb
    _spec.loader.exec_module(cb)


torch.manual_seed(20260905)


def _rope_tables(head_dim: int, max_pos: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv)  # (max_pos, D/2)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x (n, H, D) rotate-half RoPE at ``positions`` (n)."""
    half = x.shape[-1] // 2
    c = cos[positions].unsqueeze(1)  # (n, 1, D/2)
    s = sin[positions].unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class TinyDecoder:
    """A 3-layer GQA decoder implementing ``cacheblend.LayerOps``."""

    def __init__(self, num_layers=3, hidden=16, num_heads=4, num_kv_heads=2, head_dim=4, vocab=32, max_pos=256):
        self.num_layers = num_layers
        self.hidden = hidden
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.embedding = torch.randn(vocab, hidden) * 0.5
        self.wqkv = [torch.randn(hidden, (num_heads + 2 * num_kv_heads) * head_dim) * 0.3 for _ in range(num_layers)]
        self.wo = [torch.randn(num_heads * head_dim, hidden) * 0.3 for _ in range(num_layers)]
        self.w1 = [torch.randn(hidden, 2 * hidden) * 0.3 for _ in range(num_layers)]
        self.w2 = [torch.randn(2 * hidden, hidden) * 0.3 for _ in range(num_layers)]
        self.cos, self.sin = _rope_tables(head_dim, max_pos)
        self.attention_calls: List[Tuple[int, int, int]] = []

    # ---- LayerOps ----
    def embed(self, input_ids):
        return self.embedding[input_ids]

    def input_norm(self, li, hidden_rows):
        return torch.nn.functional.layer_norm(hidden_rows, (self.hidden,))

    def qkv(self, li, attn_input):
        proj = attn_input @ self.wqkv[li]
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = proj.split([q_size, kv_size, kv_size], dim=-1)
        n = attn_input.shape[0]
        return (
            q.view(n, self.num_heads, self.head_dim),
            k.view(n, self.num_kv_heads, self.head_dim),
            v.view(n, self.num_kv_heads, self.head_dim),
        )

    def rope(self, li, positions, q, k):
        assert q.shape[1:] == (self.num_heads, self.head_dim)
        assert k.shape[1:] == (self.num_kv_heads, self.head_dim)
        return _apply_rope(q, positions, self.cos, self.sin), _apply_rope(k, positions, self.cos, self.sin)

    def rotate_k(self, li, positions, k):
        fake_q = k.new_zeros((k.shape[0], self.num_heads, self.head_dim))
        return self.rope(li, positions, fake_q, k)[1]

    def attention(self, li, q, k, v, blocked):
        self.attention_calls.append((li, int(q.shape[0]), int(k.shape[0])))
        groups = self.num_heads // self.num_kv_heads
        k_run = k.repeat_interleave(groups, dim=1)  # (L, Hq, D)
        v_run = v.repeat_interleave(groups, dim=1)
        scores = torch.einsum("shd,lhd->hsl", q, k_run) * self.scaling
        scores = scores.masked_fill(blocked.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hsl,lhd->shd", probs, v_run)
        return out.reshape(q.shape[0], self.num_heads * self.head_dim)

    def post_attention(self, li, attn_output, residual_rows):
        hidden = residual_rows + attn_output @ self.wo[li]
        mlp_in = torch.nn.functional.layer_norm(hidden, (self.hidden,))
        return hidden + torch.nn.functional.gelu(mlp_in @ self.w1[li]) @ self.w2[li]

    def all_reduce_sum(self, value):
        return value

    # ---- reference: dense full prefill ----
    def dense_kv(self, input_ids, positions):
        hidden = self.embed(input_ids)
        blocked = cb.subset_causal_mask(positions, positions, "causal")
        out = []
        for li in range(self.num_layers):
            q, k_pre, v = self.qkv(li, self.input_norm(li, hidden))
            q, k = self.rope(li, positions, q, k_pre)
            out.append((k.clone(), v.clone()))
            hidden = self.post_attention(li, self.attention(li, q, k, v, blocked), hidden)
        return out


def _inputs(prologue=5, chunks=(6, 7, 4), suffix=2, offset=0):
    total = prologue + sum(chunks) + suffix
    ids = torch.randint(0, 32, (total,))
    positions = torch.arange(offset, offset + total)
    bounds, cursor = [], 0
    for n in chunks:
        bounds.append((cursor, cursor + n))
        cursor += n
    return ids, positions, prologue, prologue + sum(chunks), bounds


# ------------------------------------------------------------------ helpers

def test_resolve_chunk_bounds_explicit_grid_and_errors():
    assert cb.resolve_chunk_bounds(10, [(0, 4), (4, 10)]) == [(0, 4), (4, 10)]
    assert cb.resolve_chunk_bounds(10, None, 4) == [(0, 4), (4, 8), (8, 10)]
    assert cb.resolve_chunk_bounds(10) == [(0, 10)]
    with pytest.raises(ValueError):
        cb.resolve_chunk_bounds(10, [(0, 4), (5, 10)])  # gap
    with pytest.raises(ValueError):
        cb.resolve_chunk_bounds(10, [(0, 4), (4, 9)])  # short
    with pytest.raises(ValueError):
        cb.resolve_chunk_bounds(10, [(0, 4), (4, 4), (4, 10)])  # empty chunk
    with pytest.raises(ValueError):
        cb.resolve_chunk_bounds(0)


def test_deviation_and_budget():
    fresh = torch.zeros(3, 2, 4)
    old = torch.zeros(3, 2, 4)
    old[1] = 1.0  # 8 entries differ by 1 -> squared sum 8
    scores = cb.deviation_scores(fresh, old)
    assert scores.tolist() == [0.0, 8.0, 0.0]
    assert scores.dtype == torch.float32
    # artifact: int(span_len * ratio); floored at 1 for ratio > 0, 0 for ratio == 0
    assert cb.recompute_budget(100, 0.16) == 16
    assert cb.recompute_budget(5, 0.16) == 1
    assert cb.recompute_budget(100, 0.0) == 0
    assert cb.recompute_budget(100, 1.0) == 100
    assert cb.recompute_budget(0, 0.5) == 0
    sel = cb.select_recompute_tokens(torch.tensor([0.1, 5.0, 0.2, 4.0]), 2)
    assert sel.tolist() == [1, 3]  # sorted, not by score order
    assert cb.select_recompute_tokens(torch.tensor([1.0, 2.0]), 0).numel() == 0


def test_subset_causal_mask_modes():
    qpos = torch.tensor([1, 4])
    kpos = torch.arange(6)
    causal = cb.subset_causal_mask(qpos, kpos, "causal")
    # query at position 1 sees keys 0,1; query at 4 sees 0..4
    assert causal.tolist() == [
        [False, False, True, True, True, True],
        [False, False, False, False, False, True],
    ]
    bottom_right = cb.subset_causal_mask(qpos, kpos, "bottom_right")
    # artifact mask: row i sees keys <= L - S + i = 4 + i -> the FIRST query
    # (position 1) is allowed to see keys 2..4, i.e. the future
    assert bottom_right.tolist() == [
        [False, False, False, False, False, True],
        [False, False, False, False, False, False],
    ]
    full = torch.arange(6)
    assert torch.equal(
        cb.subset_causal_mask(full, full, "bottom_right"),
        cb.subset_causal_mask(full, full, "causal"),
    )
    with pytest.raises(ValueError):
        cb.subset_causal_mask(qpos, kpos, "diagonal")


def test_config_from_request_validation():
    cfg = cb.CacheBlendConfig.from_request({"recomp_ratio": 0.2, "check_layer": 0, "metric": "K"})
    assert cfg.metric == "k" and cfg.check_layer == 0 and cfg.recomp_ratio == 0.2
    assert cb.CacheBlendConfig.from_request(None).recomp_ratio == 0.16
    for bad in ({"recomp_ratio": 1.5}, {"check_layer": -1}, {"metric": "q"}, {"mask": "x"}, {"chunk_tokens": 0}):
        with pytest.raises(ValueError):
            cb.CacheBlendConfig.from_request(bad)


# ------------------------------------------------------------------ engine

def test_k_only_rotation_respects_gqa_head_counts():
    model = TinyDecoder(num_heads=4, num_kv_heads=2)
    positions = torch.tensor([3, 7, 11])
    k_pre = torch.randn(3, model.num_kv_heads, model.head_dim)

    actual = cb._rotate_k_only(model, 0, positions, k_pre)
    expected = _apply_rope(k_pre, positions, model.cos, model.sin)
    torch.testing.assert_close(actual, expected)


def test_full_budget_reproduces_dense_prefill():
    model = TinyDecoder()
    ids, positions, start, end, bounds = _inputs()
    dense = model.dense_kv(ids, positions)
    cfg = cb.CacheBlendConfig(recomp_ratio=1.0, check_layer=1, chunk_bounds=bounds)
    out, meta = cb.blend(model, ids, positions, start, end, cfg)
    assert len(out) == model.num_layers
    for li, (k, v) in enumerate(out):
        torch.testing.assert_close(k, dense[li][0][start:end], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(v, dense[li][1][start:end], atol=1e-5, rtol=1e-5)
    assert meta["recomputed_tokens"] == end - start
    assert meta["effective_recomp_ratio"] == 1.0
    assert meta["chunk_count"] == len(bounds)
    assert meta["kv_reuse_method"] == "cacheblend"


def test_zero_budget_returns_rotated_chunk_cache_past_check_layer():
    model = TinyDecoder()
    ids, positions, start, end, bounds = _inputs(offset=7)
    dense = model.dense_kv(ids, positions)
    cfg = cb.CacheBlendConfig(recomp_ratio=0.0, check_layer=1, chunk_bounds=bounds)
    out, meta = cb.blend(model, ids, positions, start, end, cfg)
    assert meta["recomputed_tokens"] == 0
    # layers <= check_layer: fresh == dense
    for li in range(cfg.check_layer + 1):
        torch.testing.assert_close(out[li][0], dense[li][0][start:end], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[li][1], dense[li][1][start:end], atol=1e-5, rtol=1e-5)
    # layers > check_layer: the standalone chunk cache, K rotated at the
    # absolute positions, V untouched
    for li in range(cfg.check_layer + 1, model.num_layers):
        k_parts, v_parts = [], []
        for a, b in bounds:
            chunk = cb.chunk_kv(model, ids[start + a : start + b])
            k_pre, v = chunk[li]
            k_parts.append(_apply_rope(k_pre, positions[start + a : start + b], model.cos, model.sin))
            v_parts.append(v)
        torch.testing.assert_close(out[li][0], torch.cat(k_parts), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out[li][1], torch.cat(v_parts), atol=1e-5, rtol=1e-5)
        # and it is NOT the dense KV (the chunks were computed out of context)
        assert not torch.allclose(out[li][1], dense[li][1][start:end], atol=1e-3)


def test_partial_budget_selection_and_row_bookkeeping():
    model = TinyDecoder()
    ids, positions, start, end, bounds = _inputs(prologue=4, chunks=(8, 8, 9), suffix=3)
    span_len = end - start
    cfg = cb.CacheBlendConfig(recomp_ratio=0.16, check_layer=1, chunk_bounds=bounds)
    out, meta = cb.blend(model, ids, positions, start, end, cfg)
    sel = meta["recomputed_relative_indices"]
    assert len(sel) == int(span_len * 0.16) == meta["recomputed_tokens"]
    assert sel == sorted(set(sel))
    assert all(0 <= i < span_len for i in sel)
    assert meta["fresh_outside_tokens"] == 4 + 3
    # after the check layer every attention call carries prologue + selected +
    # suffix queries over ALL keys
    later = [call for call in model.attention_calls if call[0] > cfg.check_layer and call[2] == len(ids)]
    assert later, "no status-2 attention call recorded"
    assert all(call[1] == 4 + len(sel) + 3 for call in later)
    # selected rows are dense-fresh at the last layer, unselected are the cache
    dense = model.dense_kv(ids, positions)
    li = model.num_layers - 1
    k_last, v_last = out[li]
    sel_t = torch.tensor(sel, dtype=torch.long)
    # recomputed rows see the blended context, not the dense one, so they
    # differ from dense; the check is that they differ from the CACHE too
    cache_v = torch.cat([cb.chunk_kv(model, ids[start + a : start + b])[li][1] for a, b in bounds])
    unsel = torch.tensor([i for i in range(span_len) if i not in set(sel)], dtype=torch.long)
    torch.testing.assert_close(v_last[unsel], cache_v[unsel], atol=1e-5, rtol=1e-5)
    if len(sel):
        assert not torch.allclose(v_last[sel_t], cache_v[sel_t], atol=1e-4)
    assert meta["deviation_max"] >= meta["deviation_selected_min"] >= 0.0


def test_grid_chunking_and_single_chunk_agree_with_dense_at_full_budget():
    model = TinyDecoder()
    ids, positions, start, end, _ = _inputs(prologue=3, chunks=(10,), suffix=0)
    dense = model.dense_kv(ids, positions)
    for cfg in (
        cb.CacheBlendConfig(recomp_ratio=1.0, chunk_tokens=4),
        cb.CacheBlendConfig(recomp_ratio=1.0),
    ):
        out, meta = cb.blend(model, ids, positions, start, end, cfg)
        for li in range(model.num_layers):
            torch.testing.assert_close(out[li][0], dense[li][0][start:end], atol=1e-5, rtol=1e-5)
    assert cb.blend(model, ids, positions, start, end, cb.CacheBlendConfig(chunk_tokens=4))[1]["chunk_count"] == 3


def test_blend_rejects_bad_spans_and_layers():
    model = TinyDecoder()
    ids, positions, start, end, bounds = _inputs()
    with pytest.raises(ValueError):
        cb.blend(model, ids, positions, end, start, cb.CacheBlendConfig())
    with pytest.raises(ValueError):
        cb.blend(model, ids, positions, start, end, cb.CacheBlendConfig(check_layer=99))
    with pytest.raises(ValueError):
        cb.blend(model, ids.view(1, -1), positions, start, end, cb.CacheBlendConfig())
