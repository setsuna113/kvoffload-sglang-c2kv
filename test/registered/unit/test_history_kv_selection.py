"""CPU contracts for history-boundary KV selection."""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch


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
        "history_kv_selection.py",
    )
)
_SPEC = importlib.util.spec_from_file_location("history_kv_selection_under_test", _PATH)
history = importlib.util.module_from_spec(_SPEC)
sys.modules["history_kv_selection_under_test"] = history
_SPEC.loader.exec_module(history)


def test_gqa_scores_reduce_contiguous_query_groups_to_native_kv_heads():
    query = torch.zeros(1, 4, 4, 2)
    key = torch.zeros(1, 2, 4, 2)
    query[:, :2, -1, 0] = 10
    query[:, 2:, -1, 1] = 10
    key[:, 0, 1, 0] = 1
    key[:, 1, 2, 1] = 1

    scores = history.attention_scores_by_kv_head(
        query,
        key,
        scale=1.0,
        query_start=3,
        query_end=4,
        key_start=0,
        key_end=4,
        query_chunk_size=1,
    )

    assert scores.shape == (2, 4)
    assert scores.argmax(dim=-1).tolist() == [1, 2]


def test_h2o_scoring_accumulates_queries_before_the_last_64():
    seq_len = 70
    query = torch.zeros(1, 2, seq_len, 2)
    key = torch.zeros(1, 1, seq_len, 2)
    query[:, :, :6, 0] = 10
    key[:, :, 0, 0] = 1

    all_prefill = history.attention_scores_by_kv_head(
        query,
        key,
        scale=1.0,
        query_start=0,
        query_end=seq_len,
        key_start=0,
        key_end=seq_len,
        query_chunk_size=7,
    )
    last_64 = history.attention_scores_by_kv_head(
        query,
        key,
        scale=1.0,
        query_start=6,
        query_end=seq_len,
        key_start=0,
        key_end=seq_len,
        query_chunk_size=7,
    )

    assert all_prefill[0, 0] > last_64[0, 0] + 5


def test_chunked_scoring_has_exact_causal_softmax_without_chunk_padding():
    query = torch.zeros(1, 1, 5, 2)
    key = torch.zeros(1, 1, 5, 2)
    chunked = history.attention_scores_by_kv_head(
        query,
        key,
        scale=1.0,
        query_start=0,
        query_end=3,
        key_start=0,
        key_end=5,
        query_chunk_size=2,
    )
    single_chunk = history.attention_scores_by_kv_head(
        query,
        key,
        scale=1.0,
        query_start=0,
        query_end=3,
        key_start=0,
        key_end=5,
        query_chunk_size=5,
    )

    expected = torch.tensor([[1 + 1 / 2 + 1 / 3, 1 / 2 + 1 / 3, 1 / 3, 0, 0]])
    torch.testing.assert_close(chunked, expected)
    torch.testing.assert_close(chunked, single_chunk)


def test_snapkv_selects_different_tokens_by_layer_and_head_and_pairs_kv():
    layer0_scores = torch.zeros(2, 10)
    layer0_scores[0, 0:2] = torch.tensor([10.0, 9.0])
    layer0_scores[1, 4:6] = torch.tensor([10.0, 9.0])
    layer1_scores = torch.zeros(2, 10)
    layer1_scores[0, 2:4] = torch.tensor([10.0, 9.0])
    layer1_scores[1, 5:7] = torch.tensor([10.0, 9.0])

    selected0 = history.select_snapkv_indices(
        layer0_scores,
        target_tokens=4,
        recent_window=2,
        kernel_size=1,
        pooling="avgpool",
    )
    selected1 = history.select_snapkv_indices(
        layer1_scores,
        target_tokens=4,
        recent_window=2,
        kernel_size=1,
        pooling="avgpool",
    )

    assert selected0.tolist() == [[0, 1, 8, 9], [4, 5, 8, 9]]
    assert selected1.tolist() == [[2, 3, 8, 9], [5, 6, 8, 9]]
    assert selected0.shape == selected1.shape == (2, 4)

    key = torch.arange(10 * 2 * 3).reshape(10, 2 * 3).float()
    value = (1000 + torch.arange(10 * 2 * 2)).reshape(10, 2 * 2).float()
    gathered_key, gathered_value = history.gather_paired_kv(
        key, value, selected0
    )
    gathered_key = gathered_key.view(4, 2, 3)
    gathered_value = gathered_value.view(4, 2, 2)
    key = key.view(10, 2, 3)
    value = value.view(10, 2, 2)

    for slot in range(4):
        for head in range(2):
            source = selected0[head, slot]
            torch.testing.assert_close(gathered_key[slot, head], key[source, head])
            torch.testing.assert_close(gathered_value[slot, head], value[source, head])


def test_snapkv_avgpool_and_maxpool_take_distinct_paths():
    scores = torch.tensor([[0.0, 10.0, 0.0, 0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0]])
    average = history.select_snapkv_indices(
        scores,
        target_tokens=3,
        recent_window=2,
        kernel_size=3,
        pooling="avgpool",
    )
    maximum = history.select_snapkv_indices(
        scores,
        target_tokens=3,
        recent_window=2,
        kernel_size=3,
        pooling="maxpool",
    )

    assert average[0, 0].item() == 5
    assert maximum[0, 0].item() in {0, 1, 2}
    assert average[0, -2:].tolist() == maximum[0, -2:].tolist() == [8, 9]


def test_h2o_recent_fraction_is_not_capped_at_64_tokens():
    scores = torch.arange(200, dtype=torch.float32).repeat(2, 1)
    selected = history.select_h2o_prefill_indices(
        scores,
        target_tokens=160,
        recent_fraction=0.75,
    )

    assert selected.shape == (2, 160)
    assert selected[:, -120:].tolist() == [list(range(80, 200))] * 2
    assert torch.unique(selected[0]).numel() == 160


@pytest.mark.parametrize("target_tokens", [1, 2, 3, 10])
def test_h2o_ties_and_small_budgets_still_return_exactly_k_unique_slots(
    target_tokens,
):
    selected = history.select_h2o_prefill_indices(
        torch.zeros(3, 10),
        target_tokens=target_tokens,
        recent_fraction=0.5,
    )

    assert selected.shape == (3, target_tokens)
    assert selected[:, -1].tolist() == [9, 9, 9]
    for head_indices in selected:
        assert torch.unique(head_indices).numel() == target_tokens


@pytest.mark.parametrize(
    "target_tokens, expected",
    [
        (1, [9]),
        (3, [0, 1, 9]),
        (6, [0, 1, 2, 3, 8, 9]),
    ],
)
def test_streamingllm_reserves_the_final_token_after_attention_sinks(
    target_tokens, expected
):
    selected = history.select_streamingllm_indices(
        10, target_tokens=target_tokens
    )
    assert selected.tolist() == expected
    assert selected.numel() == target_tokens


def test_headwise_selection_rejects_shared_pre_rope_positions():
    for method in sorted(history.HEADWISE_HISTORY_KV_METHODS):
        with pytest.raises(ValueError, match="requires raw_kv_position_mode='rotated'"):
            history.require_rotated_headwise_storage(method, "pre_rope")
        history.require_rotated_headwise_storage(method, "rotated")


def test_headwise_audit_metadata_reports_shape_and_bounded_previews():
    indices = [
        torch.tensor([[0, 1, 8, 9], [4, 5, 8, 9]]),
        torch.tensor([[2, 3, 8, 9], [5, 6, 8, 9]]),
    ]
    metadata = history.summarize_headwise_indices(indices, preview_tokens=2)

    assert metadata["selection_index_shape"] == [2, 2, 4]
    assert metadata["selection_indices_preview_tokens"] == 2
    assert metadata["selection_indices_preview"][1]["head_prefix"] == [[2, 3], [5, 6]]
    assert metadata["selection_indices_preview"][0]["head_suffix"] == [[8, 9], [8, 9]]
