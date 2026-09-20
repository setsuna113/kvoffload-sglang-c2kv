from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.agentkv import (
    AGENTKV_ALGORITHM_VERSION,
    AGENTKV_QUERY_RING_CAPACITY,
    AGENTKV_STAGE_ACT,
    AGENTKV_STAGE_OTHERS,
    AGENTKV_STAGE_THINK,
    AGENTKV_STAGE_TOOL,
    AgentKVQueryRing,
    refine_agentkv_stages_with_markers,
    resolve_agentkv_message_stages,
    select_agentkv_headwise,
    select_agentkv_layer_indices,
)


def test_event_hints_preserve_appworld_act_and_tool_semantics() -> None:
    stages = resolve_agentkv_message_stages(
        total_tokens=12,
        message_prefix_token_counts=[1, 4, 7, 10],
        event_messages=[
            {"role": "system", "phase": "others", "message_index": 0},
            {"role": "assistant", "phase": "act", "message_index": 1},
            {"role": "user", "phase": "tool", "message_index": 2},
        ],
    )

    assert stages.tolist() == [
        AGENTKV_STAGE_OTHERS,
        AGENTKV_STAGE_OTHERS,
        AGENTKV_STAGE_OTHERS,
        AGENTKV_STAGE_OTHERS,
        AGENTKV_STAGE_ACT,
        AGENTKV_STAGE_ACT,
        AGENTKV_STAGE_ACT,
        AGENTKV_STAGE_TOOL,
        AGENTKV_STAGE_TOOL,
        AGENTKV_STAGE_TOOL,
        AGENTKV_STAGE_THINK,
        AGENTKV_STAGE_THINK,
    ]


def test_native_assistant_markers_refine_stage_until_message_boundary() -> None:
    base = torch.tensor(
        [
            AGENTKV_STAGE_OTHERS,
            AGENTKV_STAGE_THINK,
            AGENTKV_STAGE_THINK,
            AGENTKV_STAGE_THINK,
            AGENTKV_STAGE_THINK,
            AGENTKV_STAGE_TOOL,
        ],
        dtype=torch.int32,
    )

    refined = refine_agentkv_stages_with_markers(
        [9, 10, 11, 12, 13, 14],
        base,
        {(11, 12): AGENTKV_STAGE_ACT},
        reset_offsets=[5],
    )

    assert refined.tolist() == [
        AGENTKV_STAGE_OTHERS,
        AGENTKV_STAGE_THINK,
        AGENTKV_STAGE_ACT,
        AGENTKV_STAGE_ACT,
        AGENTKV_STAGE_ACT,
        AGENTKV_STAGE_TOOL,
    ]


def test_marker_stage_resets_between_adjacent_assistant_messages() -> None:
    base = torch.full((6,), AGENTKV_STAGE_THINK, dtype=torch.int32)
    refined = refine_agentkv_stages_with_markers(
        [10, 11, 12, 20, 21, 22],
        base,
        {(11,): AGENTKV_STAGE_ACT},
        reset_offsets=[0, 3],
    )
    assert refined.tolist() == [
        AGENTKV_STAGE_THINK,
        AGENTKV_STAGE_ACT,
        AGENTKV_STAGE_ACT,
        AGENTKV_STAGE_THINK,
        AGENTKV_STAGE_THINK,
        AGENTKV_STAGE_THINK,
    ]


def test_query_ring_keeps_last_eight_queries_per_stage() -> None:
    ring = AgentKVQueryRing()
    positions = torch.arange(40, dtype=torch.long)
    stages = torch.arange(40, dtype=torch.int32) % 4
    query = positions.float().view(40, 1, 1)

    ring.write_layer(
        layer_id=0,
        query=query,
        positions=positions,
        stage_ids=stages,
    )
    observed_query, observed_positions = ring.read_layer(0)

    assert ring.rows_by_stage(0) == [AGENTKV_QUERY_RING_CAPACITY] * 4
    assert observed_query.shape == (32, 1, 1)
    assert observed_positions.tolist() == [
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        9,
        13,
        17,
        21,
        25,
        29,
        33,
        37,
        10,
        14,
        18,
        22,
        26,
        30,
        34,
        38,
        11,
        15,
        19,
        23,
        27,
        31,
        35,
        39,
    ]


def test_query_ring_single_token_writes_match_batched_stage_contents() -> None:
    positions = torch.arange(45, dtype=torch.long)
    stages = torch.tensor([i % 5 - 1 for i in range(45)], dtype=torch.int32)
    query = positions.float().view(45, 1, 1)
    batched = AgentKVQueryRing()
    batched.write_layer(
        layer_id=0, query=query, positions=positions, stage_ids=stages
    )
    sequential = AgentKVQueryRing()
    for index in range(len(positions)):
        sequential.write_layer(
            layer_id=0,
            query=query[index : index + 1],
            positions=positions[index : index + 1],
            stage_ids=stages[index : index + 1],
        )
    assert sequential.rows_by_stage(0) == batched.rows_by_stage(0)
    for actual, expected in zip(sequential.read_layer(0), batched.read_layer(0)):
        torch.testing.assert_close(actual, expected)


def test_layer_selector_matches_stageq_snapkv_gqa_mean_equation() -> None:
    key = torch.zeros((6, 2, 2), dtype=torch.float32)
    key[1:5, 0] = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 0.0]])
    key[1:5, 1] = torch.tensor([[0.0, 1.0], [0.0, 3.0], [2.0, 0.0], [0.0, -1.0]])
    query = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )

    selected = select_agentkv_layer_indices(
        key,
        query,
        target_tokens=4,
        sink_tokens=1,
        recent_tokens=1,
    )

    candidates = torch.arange(1, 5)
    grouped_query = query.reshape(1, 2, 2, 2)
    candidate_key = key[candidates].permute(1, 0, 2)
    logits = torch.einsum("ohgd,hcd->ohgc", grouped_query, candidate_key)
    scores = torch.softmax(logits / math.sqrt(2), dim=-1).mean(dim=(0, 2))
    expected_middle = candidates[torch.topk(scores, k=2, dim=-1).indices]
    expected = torch.sort(
        torch.cat(
            [torch.tensor([[0, 5], [0, 5]]), expected_middle],
            dim=1,
        ),
        dim=1,
    ).values
    assert torch.equal(selected, expected)
    assert selected.tolist() == [[0, 1, 3, 5], [0, 1, 2, 5]]


def test_selector_returns_reference_runtime_contract_and_provenance() -> None:
    ring = AgentKVQueryRing()
    positions = torch.arange(4, dtype=torch.long)
    stages = torch.arange(4, dtype=torch.int32)
    for layer_id in range(2):
        query = torch.randn(
            (4, 4, 3), generator=torch.Generator().manual_seed(layer_id)
        )
        ring.write_layer(
            layer_id=layer_id,
            query=query,
            positions=positions,
            stage_ids=stages,
        )
    keys = [
        torch.randn((40, 2, 3), generator=torch.Generator().manual_seed(10 + layer_id))
        for layer_id in range(2)
    ]

    selected, metadata = select_agentkv_headwise(
        keys,
        ring,
        target_tokens=28,
    )

    assert [tuple(item.shape) for item in selected] == [(2, 28), (2, 28)]
    assert metadata["method"] == "agentkv"
    assert metadata["algorithm_version"] == AGENTKV_ALGORITHM_VERSION
    assert metadata["source_commit"] == "254c57bc84e4a7895159ba1062bd46c83626f511"
    assert metadata["reference_attention_backend"] == "torch_sdpa"
    assert metadata["per_layer_budget_tokens"] == [28, 28]
    assert metadata["per_layer_query_rows_by_stage"] == [[1, 1, 1, 1], [1, 1, 1, 1]]


def test_missing_query_observation_matches_upstream_identity_fallback() -> None:
    key = torch.randn(40, 2, 4)
    selected = select_agentkv_layer_indices(
        key,
        key.new_empty((0, 2, 4)),
        target_tokens=28,
    )
    assert selected.shape == (2, 40)
    assert torch.equal(selected[0], torch.arange(40))
