"""CPU contracts for CommitKV's paper-faithful tensor core."""

from __future__ import annotations

import importlib.util
import math
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
        "commitkv.py",
    )
)
_SPEC = importlib.util.spec_from_file_location("commitkv_under_test", _PATH)
commitkv = importlib.util.module_from_spec(_SPEC)
sys.modules["commitkv_under_test"] = commitkv
_SPEC.loader.exec_module(commitkv)


def _window(query, key, value, query_positions=None, key_positions=None):
    return commitkv.build_deletion_effect_window(
        query,
        key,
        value,
        query_positions=(
            query_positions
            if query_positions is not None
            else range(key.shape[1] - query.shape[1], key.shape[1])
        ),
        key_positions=(key_positions if key_positions is not None else range(key.shape[1])),
        scale=1.0,
    )


class _EffectWindow:
    def __init__(self, effects):
        self.effects = effects

    def effect(self, indices):
        return torch.tensor(self.effects[frozenset(int(i) for i in indices)])


def test_deletion_effect_matches_direct_renormalized_attention():
    query = torch.tensor([[[1.0, 0.0], [0.5, 1.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]])
    value = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]])
    window = _window(query, key, value, [1, 2], [0, 1, 2])

    effect = window.effect([1])
    kept = torch.tensor([0, 2])
    logits = torch.matmul(query.float(), key[:, kept].float().transpose(-2, -1))
    causal = torch.tensor([0, 2]).view(1, 1, -1) <= torch.tensor([1, 2]).view(
        1, -1, 1
    )
    direct_weights = torch.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)
    direct_output = torch.matmul(direct_weights, value[:, kept].float())
    expected = (
        torch.linalg.vector_norm(window.outputs - direct_output, dim=-1)
        / torch.linalg.vector_norm(window.outputs, dim=-1)
    ).amax()

    torch.testing.assert_close(effect, expected)


def test_deletion_effect_uses_each_querys_causal_page_subset():
    weights = torch.tensor([[[1.0, 0.0], [0.4, 0.6]]])
    values = torch.tensor([[[1.0], [4.0]]])
    outputs = torch.matmul(weights, values)
    window = commitkv.DeletionEffectWindow(
        weights, values, outputs, torch.tensor([0, 1]), torch.tensor([0, 1])
    )

    assert window.effect([1]).item() > 0
    # The first query cannot see the deleted token at absolute position 1.
    first_query = commitkv.DeletionEffectWindow(
        weights[:, :1],
        values,
        outputs[:, :1],
        torch.tensor([0]),
        torch.tensor([0, 1]),
    )
    assert first_query.effect([1]).item() == 0


def test_gqa_expands_paired_keys_and_values_to_query_heads():
    query = torch.tensor(
        [[[2.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 2.0]]]
    )
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
    value = torch.tensor([[[10.0, 0.0], [0.0, 1.0]], [[0.0, 20.0], [2.0, 0.0]]])

    window = _window(query, key, value, [1], [0, 1])

    assert window.values.shape == (4, 2, 2)
    torch.testing.assert_close(window.values[0], window.values[1])
    torch.testing.assert_close(window.values[2], window.values[3])
    assert window.effect([0]).item() > 0


def test_event_partition_is_contiguous_and_bounded():
    pages = commitkv.partition_event_span("tool-3", 10, 47, page_size=16)

    assert [(page.start, page.end) for page in pages] == [
        (10, 26),
        (26, 42),
        (42, 47),
    ]
    assert all(len(page) <= 16 for page in pages)
    assert [page.page_id for page in pages] == [
        ("tool-3", 0),
        ("tool-3", 1),
        ("tool-3", 2),
    ]


def test_only_fully_resident_pages_are_eligible_for_a_measurement_window():
    pages = commitkv.partition_event_span("tool-3", 10, 15, page_size=3)

    mapped = commitkv.resident_page_indices(pages, [8, 10, 11, 12, 14, 20])

    assert mapped == {("tool-3", 0): (1, 2, 3)}


def test_percentile_midrank_and_paired_lifecycle_states():
    config = commitkv.CommitKVConfig()
    pre = {
        "completed": 0.20,
        "dormant": 0.001,
        "active": 0.20,
        "uncertain": 0.02,
        "missing": 1.0,
    }
    post = {
        "completed": 0.001,
        "dormant": 0.001,
        "active": 0.20,
        "uncertain": 0.02,
    }

    evidence = commitkv.pair_lifecycle_evidence(pre, post, config=config)

    assert set(evidence) == {"completed", "dormant", "active", "uncertain"}
    assert evidence["completed"].state == commitkv.LifecycleState.COMPLETION_CANDIDATE
    assert evidence["dormant"].state == commitkv.LifecycleState.DORMANT
    assert evidence["active"].state == commitkv.LifecycleState.STILL_ACTIVE
    tied = commitkv.percentile_ranks({"a": 1.0, "b": 1.0})
    assert tied == {"a": 0.5, "b": 0.5}


def test_joint_retirement_recomputes_union_and_rejects_accumulated_effect():
    weights = torch.tensor([[[0.004, 0.004, 0.992]]])
    values = torch.tensor([[[0.0], [0.0], [1.0]]])
    outputs = torch.matmul(weights, values)
    window = commitkv.DeletionEffectWindow(
        weights, values, outputs, torch.tensor([2]), torch.tensor([0, 1, 2])
    )
    candidate = commitkv.LifecycleEvidence(
        0.2,
        0.004,
        0.9,
        0.1,
        commitkv.LifecycleState.COMPLETION_CANDIDATE,
    )

    accepted, tested = commitkv.greedy_joint_retirement(
        {"a": candidate, "b": candidate},
        {"a": [0], "b": [1]},
        window,
        joint_threshold=0.006,
    )

    assert accepted == ("a",)
    assert tested["a"] <= 0.006
    assert tested["b"] > 0.006


def test_joint_retirement_orders_post_ascending_then_pre_descending():
    weights = torch.tensor([[[0.001, 0.001, 0.998]]])
    values = torch.ones(1, 3, 1)
    window = commitkv.DeletionEffectWindow(
        weights,
        values,
        torch.matmul(weights, values),
        torch.tensor([2]),
        torch.tensor([0, 1, 2]),
    )
    base = dict(pre_percentile=0.9, post_percentile=0.1,
                state=commitkv.LifecycleState.COMPLETION_CANDIDATE)
    evidence = {
        "lower_pre": commitkv.LifecycleEvidence(0.1, 0.001, **base),
        "higher_pre": commitkv.LifecycleEvidence(0.2, 0.001, **base),
    }

    accepted, _ = commitkv.greedy_joint_retirement(
        evidence,
        {"lower_pre": [0], "higher_pre": [1]},
        window,
        joint_threshold=1.0,
    )

    assert accepted == ("higher_pre", "lower_pre")


def test_pending_budget_uses_whole_pages_pre_effect_priority_and_page_cap():
    pages = {"high": range(0, 4), "medium": range(4, 8), "low": range(8, 10)}
    selected_pages, selected_tokens = commitkv.protect_pending_pages(
        pages,
        {"low": 0.1, "medium": 0.2, "high": 0.3},
        total_budget=32,
        pending_fraction=0.25,
        max_pages=2,
    )

    assert selected_pages == ("high", "medium")
    assert selected_tokens == tuple(range(8))


def test_retention_excludes_retired_protects_pending_and_applies_one_index_set():
    retained = commitkv.compose_retained_indices(
        [5, 4, 3, 2, 1, 0],
        resident_token_count=6,
        budget=4,
        retired_indices=[4],
        pending_indices=[1, 2],
    )
    keys = torch.arange(2 * 6 * 3).reshape(2, 6, 3)
    values = 100 + torch.arange(2 * 6 * 3).reshape(2, 6, 3)
    positions = torch.tensor([10, 20, 30, 40, 50, 60])

    selected_k, selected_v, selected_p = commitkv.apply_retained_indices(
        keys, values, positions, retained, token_dim=1
    )

    assert retained.tolist() == [1, 2, 3, 5]
    torch.testing.assert_close(selected_k, keys[:, retained])
    torch.testing.assert_close(selected_v, values[:, retained])
    torch.testing.assert_close(selected_p, positions[retained])
    assert 4 not in retained.tolist()


def test_reference_selector_repeats_one_common_index_set_across_layers_and_heads():
    selected, metadata = commitkv.select_commitkv_headwise(
        [5, 4, 3, 2, 1, 0],
        resident_token_count=6,
        target_tokens=4,
        retired_indices=[4],
        pending_indices=[1],
        num_layers=3,
        num_kv_heads=2,
    )

    assert len(selected) == 3
    assert all(indices.tolist() == [[1, 2, 3, 5], [1, 2, 3, 5]]
               for indices in selected)
    assert metadata["algorithm_version"] == "commitkv_arxiv_2608_07855_v1"
    assert metadata["common_retained_indices"] is True
    assert metadata["reference_attention_backend"] == "torch_sdpa"


def test_runtime_state_pairs_windows_retires_jointly_and_builds_checkpoint():
    config = commitkv.CommitKVConfig(
        measurement_layer_id=7,
        pending_fraction=0.5,
        joint_threshold=0.01,
    )
    state = commitkv.CommitKVRuntimeState(config)
    pages = tuple(
        commitkv.EventPage(name, 0, index, index + 1)
        for index, name in enumerate(("completed", "dormant", "active", "uncertain"))
    )
    pre = _EffectWindow({
        frozenset({0}): 0.20,
        frozenset({1}): 0.001,
        frozenset({2}): 0.20,
        frozenset({3}): 0.02,
    })
    pre_receipt = state.record_pre(
        "commit-1", pages, pre, range(6), total_budget=8
    )
    assert pre_receipt["measurement_layer_id"] == 7
    assert pre_receipt["protected_pending_pages"] == 4

    pending_selected, pending_meta = state.checkpoint(
        reversed(range(6)),
        range(6),
        target_tokens=8,
        num_layers=2,
        num_kv_heads=2,
    )
    assert pending_meta["protected_pending_page_count"] == 4
    assert all({0, 1, 2, 3}.issubset(set(row.tolist()))
               for layer in pending_selected for row in layer)

    post = _EffectWindow({
        frozenset({0}): 0.001,
        frozenset({1}): 0.001,
        frozenset({2}): 0.20,
        frozenset({3}): 0.02,
        # Exact union needed by greedy joint validation.
        frozenset({0, 1}): 0.02,
        frozenset({0, 2}): 0.20,
        frozenset({0, 3}): 0.02,
    })
    post_receipt = state.record_post("commit-1", post, range(6))
    assert post_receipt["accepted_page_ids"] == [("completed", 0)]
    assert post_receipt["lifecycle_states"][("dormant", 0)] == "dormant"

    selected, metadata = state.checkpoint(
        reversed(range(6)),
        range(6),
        target_tokens=4,
        num_layers=2,
        num_kv_heads=2,
    )
    assert all(indices.tolist() == [[2, 3, 4, 5], [2, 3, 4, 5]]
               for indices in selected)
    assert metadata["retired_page_count"] == 1
    assert metadata["completed_transitions"] == 1

    # A later commit re-scans every message page, including the one already
    # retired; it must be skipped, not protected as pending (first AppWorld run
    # failed the checkpoint with "a token cannot be both retired and pending").
    pre2 = _EffectWindow({
        frozenset({0}): 0.30,
        frozenset({1}): 0.001,
        frozenset({2}): 0.20,
        frozenset({3}): 0.02,
    })
    receipt2 = state.record_pre("commit-2", pages, pre2, range(6), total_budget=4)
    assert ("completed", 0) not in receipt2["protected_page_ids"]
    assert receipt2["scanned_pages"] == 3
    selected2, metadata2 = state.checkpoint(
        reversed(range(6)),
        range(6),
        target_tokens=4,
        num_layers=2,
        num_kv_heads=2,
    )
    assert all(0 not in row.tolist() for layer in selected2 for row in layer)
    assert metadata2["retired_page_count"] == 1


@pytest.mark.parametrize("observed_query_count", [0, 7])
def test_incomplete_post_releases_protection_without_retiring_pages(
    observed_query_count,
):
    state = commitkv.CommitKVRuntimeState(
        commitkv.CommitKVConfig(
            measurement_layer_id=0,
            window_size=8,
            page_size=1,
            pending_fraction=0.5,
        )
    )
    retired_page = commitkv.EventPage("retired", 0, 0, 1)
    state.retired_pages[retired_page.page_id] = retired_page
    pending_page = commitkv.EventPage("pending", 0, 1, 2)
    state.record_pre(
        "commit",
        [pending_page],
        _EffectWindow({frozenset({1}): 0.2}),
        range(4),
        total_budget=2,
    )
    protected, protected_meta = state.checkpoint(
        [3, 2, 1, 0], range(4), target_tokens=2, num_layers=1, num_kv_heads=1
    )
    assert protected[0][0].tolist() == [1, 3]
    assert protected_meta["protected_pending_page_count"] == 1

    receipt = state.record_incomplete_post(
        "commit", observed_query_count=observed_query_count
    )

    assert receipt["commit_id"] == "commit"
    assert receipt["measurement_phase"] == "post_commit_unavailable"
    assert receipt["reason"] == "next_turn_ended_before_window"
    assert receipt["observed_query_count"] == observed_query_count
    assert receipt["required_query_count"] == 8
    assert receipt["accepted_page_ids"] == []
    assert receipt["incomplete_transition_policy"] == (
        "full_window_or_unclassified_project_convention"
    )
    assert state.pending is None
    assert state.retired_pages == {retired_page.page_id: retired_page}
    assert state.completed_transitions == 0
    assert state.incomplete_transitions == 1

    selected, metadata = state.checkpoint(
        [3, 2, 1, 0], range(4), target_tokens=2, num_layers=1, num_kv_heads=1
    )
    assert selected[0][0].tolist() == [2, 3]
    assert metadata["protected_pending_page_count"] == 0
    assert metadata["retired_page_count"] == 1
    assert metadata["completed_transitions"] == 0
    assert metadata["incomplete_transitions"] == 1
    assert metadata["incomplete_transition_policy"] == (
        "full_window_or_unclassified_project_convention"
    )


def test_incomplete_post_rejects_wrong_commit_and_invalid_window_without_mutation():
    state = commitkv.CommitKVRuntimeState(
        commitkv.CommitKVConfig(
            measurement_layer_id=0,
            window_size=8,
            page_size=1,
            pending_fraction=0.5,
        )
    )
    page = commitkv.EventPage("pending", 0, 1, 2)
    state.record_pre(
        "commit",
        [page],
        _EffectWindow({frozenset({1}): 0.2}),
        range(4),
        total_budget=2,
    )
    pending = state.pending

    with pytest.raises(RuntimeError):
        state.record_incomplete_post("other", observed_query_count=1)
    with pytest.raises(ValueError):
        state.record_incomplete_post("commit", observed_query_count=8)
    with pytest.raises(ValueError):
        state.record_incomplete_post("commit", observed_query_count=-1)

    assert state.pending is pending
    assert state.retired_pages == {}
    assert state.completed_transitions == 0
    assert state.incomplete_transitions == 0
    selected, metadata = state.checkpoint(
        [3, 2, 1, 0], range(4), target_tokens=2, num_layers=1, num_kv_heads=1
    )
    assert selected[0][0].tolist() == [1, 3]
    assert metadata["protected_pending_page_count"] == 1


def test_runtime_state_requires_explicit_layer_and_preserves_pending_page():
    with pytest.raises(ValueError, match="measurement_layer_id"):
        commitkv.CommitKVRuntimeState(commitkv.CommitKVConfig())

    state = commitkv.CommitKVRuntimeState(
        commitkv.CommitKVConfig(measurement_layer_id=0, pending_fraction=1.0)
    )
    page = commitkv.EventPage("event", 0, 4, 5)
    state.record_pre(
        "commit", [page], _EffectWindow({frozenset({1}): 0.2}), [0, 4],
        total_budget=2,
    )
    with pytest.raises(RuntimeError, match="evicted before post"):
        state.checkpoint(
            [0], [0], target_tokens=2, num_layers=1, num_kv_heads=1
        )
    with pytest.raises(RuntimeError, match="missing from the post window"):
        state.record_post(
            "commit", _EffectWindow({frozenset({0}): 0.1}), [0]
        )


def test_deleting_all_attention_mass_fails_closed():
    weights = torch.ones(1, 1, 1)
    values = torch.ones(1, 1, 2)
    window = commitkv.DeletionEffectWindow(
        weights,
        values,
        values.clone(),
        torch.tensor([0]),
        torch.tensor([0]),
    )

    effect = window.effect([0]).item()
    assert math.isinf(effect)
    evidence = commitkv.pair_lifecycle_evidence(
        {"page": effect}, {"page": effect}, config=commitkv.CommitKVConfig()
    )
    assert evidence["page"].state == commitkv.LifecycleState.UNCERTAIN


def test_invalid_budget_and_index_contracts_fail_closed():
    with pytest.raises(ValueError, match="pending tokens exceed"):
        commitkv.compose_retained_indices(
            range(3), resident_token_count=3, budget=1, pending_indices=[0, 1]
        )
    with pytest.raises(ValueError, match="both retired and pending"):
        commitkv.compose_retained_indices(
            range(3),
            resident_token_count=3,
            budget=2,
            retired_indices=[1],
            pending_indices=[1],
        )

    exactly_full = commitkv.compose_retained_indices(
        [2], resident_token_count=3, budget=2, pending_indices=[0, 1]
    )
    assert exactly_full.tolist() == [0, 1]


def test_v2_admitted_complete_page_vetoes_retirement_before_joint_test_and_expires():
    import copy

    state = commitkv.CommitKVRuntimeState(commitkv.CommitKVConfig(
        measurement_layer_id=0, joint_threshold=0.01,
    ))
    pages = tuple(commitkv.EventPage(name, 0, index, index + 1)
                  for index, name in enumerate(("source", "dormant", "active", "uncertain")))
    page = pages[0]
    pre = _EffectWindow({frozenset({0}): 0.2, frozenset({1}): 0.001,
                         frozenset({2}): 0.2, frozenset({3}): 0.02})
    post = _EffectWindow({frozenset({0}): 0.001, frozenset({1}): 0.001,
                          frozenset({2}): 0.2, frozenset({3}): 0.02,
                          frozenset({0, 1}): 0.02, frozenset({0, 2}): 0.2,
                          frozenset({0, 3}): 0.02})
    state.record_pre("first", pages, pre, range(6), total_budget=8)
    state.retirement_veto_positions = frozenset({0})
    held = copy.deepcopy(state)
    assert held.retirement_veto_positions == frozenset({0})
    receipt = state.record_post("first", post, range(6))
    assert receipt["retirement_vetoed_page_ids"] == [page.page_id]
    assert receipt["accepted_page_ids"] == []
    assert state.retired_pages == {}

    state.retirement_veto_positions = frozenset()
    state.record_pre("second", pages, pre, range(6), total_budget=8)
    receipt = state.record_post("second", post, range(6))
    assert receipt.get("retirement_vetoed_page_ids", []) == []
    assert receipt["accepted_page_ids"] == [page.page_id]
