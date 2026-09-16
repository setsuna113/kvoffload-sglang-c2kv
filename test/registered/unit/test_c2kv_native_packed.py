"""CPU-only tests for exact event-native C2KV packing contracts."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.normpath(
    os.path.join(
        _HERE,
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "srt",
        "mem_cache",
        "c2kv_native_packed.py",
    )
)
_SPEC = importlib.util.spec_from_file_location("c2kv_native_packed_under_test", _PATH)
native = importlib.util.module_from_spec(_SPEC)
sys.modules["c2kv_native_packed_under_test"] = native
_SPEC.loader.exec_module(native)


def _binding(**overrides):
    values = {
        "model_path": "/checkpoints/checkpoint-1000",
        "tokenizer_path": None,
        "weight_version": None,
        "dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "gist_parameter_dtype": "float32",
        "gist_compute_dtype": "bfloat16",
        "gist_type": "dynamic-interleave",
        "gist_param": "qkv",
        "gist_extra_embed_num": 1,
        "gist_residual_type": "none",
        "gist_overlap": 0,
        "pic_enabled": False,
        "pic_param": "qkv",
        "query_projection": "base",
    }
    values.update(overrides)
    return native.canonical_model_binding(**values)


def _chunk(
    chunk_id,
    token_ids,
    *,
    source_position_start=None,
    gist_position_ids=None,
    handle=None,
):
    return {
        "chunk_id": chunk_id,
        "event_id": f"event-{chunk_id}",
        "part_index": 0,
        "source_indices": [2, 3],
        "source_token_start": 0,
        "source_token_end": len(token_ids),
        "token_ids": list(token_ids),
        "handle": handle,
        "source_position_start": source_position_start,
        "gist_position_ids": gist_position_ids,
    }


def _plan(*, encoder_chunks, compression_chunks=()):
    return native.plan_native_packed_request(
        system_input_ids=[101, 102],
        workspace_input_ids=[901, 902],
        encoder_chunks=encoder_chunks,
        compression_chunks=compression_chunks,
        model_binding=_binding(),
        packing_version=native.PACKING_VERSION,
        raw_layout_profile=native.RAW_LAYOUT_PROFILE,
        encoding_scope="G:event-group-v1",
        compression_ratio=4,
    )


def test_builds_exact_raw_layout_and_source_span_rope_positions():
    first = _chunk(
        "a",
        range(10, 20),
        source_position_start=2,
        gist_position_ids=[5, 9, 11],
    )
    second = _chunk(
        "b",
        [20, 21, 22],
        source_position_start=12,
        gist_position_ids=[14],
    )

    plan = _plan(encoder_chunks=[first, second])

    assert plan.logical_input_ids == (
        101,
        102,
        *range(10, 20),
        20,
        21,
        22,
        901,
        902,
    )
    assert plan.segment_boundaries == ((2, 12), (12, 15))
    assert plan.costs == {
        "system_tokens": 2,
        "raw_tokens": 2,
        "presented_encoder_tokens": 13,
        "eligible_encoder_tokens": 13,
        "gist_tokens": 4,
        "gist_prefix_kv_tokens": 4,
        "raw_workspace_kv_tokens": 2,
        "resident_kv_tokens": 8,
    }


def test_extra_compression_chunks_are_materialized_but_not_injected():
    selected = _chunk(
        "selected",
        [10, 11, 12, 13],
        source_position_start=2,
        gist_position_ids=[5],
    )
    extra = _chunk("eligible", [30, 31, 32])

    plan = _plan(encoder_chunks=[selected], compression_chunks=[extra, selected])

    assert plan.logical_input_ids == (101, 102, 10, 11, 12, 13, 901, 902)
    assert len(plan.unique_chunks) == 2
    assert plan.selected_handles == (plan.compression_handles[1],)
    assert plan.costs["presented_encoder_tokens"] == 4
    assert plan.costs["eligible_encoder_tokens"] == 7


@pytest.mark.parametrize(
    "change",
    [
        {"compression_ratio": 8},
        {"encoding_scope": "current"},
        {"model_binding": _binding(model_path="/different-checkpoint")},
    ],
)
def test_handle_binds_ratio_scope_and_checkpoint(change):
    chunk = _chunk("a", [10, 11, 12])
    kwargs = {
        "model_binding": _binding(),
        "packing_version": native.PACKING_VERSION,
        "encoding_scope": "G:event-group-v1",
        "compression_ratio": 4,
    }
    reference = native.canonical_chunk_handle(chunk, **kwargs)
    kwargs.update(change)
    assert native.canonical_chunk_handle(chunk, **kwargs) != reference


def test_handle_binds_full_chunk_metadata_and_tokens():
    base = _chunk("a", [10, 11, 12])
    kwargs = {
        "model_binding": _binding(),
        "packing_version": native.PACKING_VERSION,
        "encoding_scope": "G:event-group-v1",
        "compression_ratio": 4,
    }
    reference = native.canonical_chunk_handle(base, **kwargs)
    for changed in (
        {**base, "chunk_id": "b"},
        {**base, "event_id": "different"},
        {**base, "source_indices": [4]},
        {**base, "token_ids": [10, 11, 13]},
    ):
        assert native.canonical_chunk_handle(changed, **kwargs) != reference


def test_supplied_handle_and_placement_mismatches_fail_closed():
    chunk = _chunk(
        "a",
        [10, 11, 12, 13],
        source_position_start=3,
        gist_position_ids=[5],
        handle="not-the-canonical-handle",
    )
    with pytest.raises(ValueError, match="C2KV_NATIVE_HANDLE_MISMATCH"):
        _plan(encoder_chunks=[chunk])

    chunk["handle"] = None
    with pytest.raises(ValueError, match="C2KV_NATIVE_SOURCE_POSITION_MISMATCH"):
        _plan(encoder_chunks=[chunk])

    chunk["source_position_start"] = 2
    chunk["gist_position_ids"] = [4]
    with pytest.raises(ValueError, match="C2KV_NATIVE_GIST_POSITION_MISMATCH"):
        _plan(encoder_chunks=[chunk])


def test_chunk_bounds_and_frozen_token_ids_fail_closed():
    chunk = _chunk("a", [10, 11])
    chunk["source_token_end"] = 3
    with pytest.raises(ValueError, match="source token bounds"):
        _plan(encoder_chunks=[chunk])

    chunk = _chunk("a", [10, True])
    with pytest.raises(ValueError, match="nonnegative integer token ids"):
        _plan(encoder_chunks=[chunk])


def test_shadow_values_use_ieee_binary16_storage_roundtrip():
    assert native.float16_roundtrip([1.0001, -2.5]) == [1.0, -2.5]
