import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.history_kv_reference import (  # noqa: E402
    ReferenceLayerKV,
    gather_reference_layer,
    pyramidkv_layer_budgets,
    reference_sdpa,
    select_pyramidkv_headwise,
)


def test_pyramid_schedule_matches_official_funnel_and_average_budget():
    budgets = pyramidkv_layer_budgets(4096, 320, 32, recent_window=64, beta=20)
    assert budgets[0] > budgets[-1]
    assert budgets == sorted(budgets, reverse=True)
    # Integer layer steps in the official implementation can leave a small
    # remainder, but the schedule remains centered on the requested budget.
    assert abs(sum(budgets) / len(budgets) - 320) < 16


def test_pyramid_selection_is_layer_and_head_specific_and_keeps_recent_window():
    scores = []
    for layer in range(4):
        tensor = torch.stack(
            [
                torch.arange(128, dtype=torch.float32),
                torch.arange(127, -1, -1, dtype=torch.float32),
            ]
        )
        tensor[0] += layer * 0.01
        scores.append(tensor)
    selected, meta = select_pyramidkv_headwise(
        scores, target_tokens=40, recent_window=8, kernel_size=1
    )
    assert len({item.shape[1] for item in selected}) > 1
    assert not torch.equal(selected[0][0], selected[0][1])
    for indices in selected:
        torch.testing.assert_close(
            indices[:, -8:], torch.arange(120, 128).expand(2, -1)
        )
    assert meta["per_head_selection"] is True
    assert meta["reference_attention_backend"] == "torch_sdpa"


def test_pyramid_absolute_target_uses_largest_fitting_official_schedule():
    scores = [torch.arange(4096, dtype=torch.float32).expand(2, -1) for _ in range(32)]
    selected, meta = select_pyramidkv_headwise(
        scores, target_tokens=320, recent_window=64, kernel_size=1
    )
    realized = math.ceil(sum(item.shape[1] for item in selected) / len(selected))
    assert realized <= 320
    assert meta["requested_target_tokens"] == 320
    assert meta["nominal_schedule_target_tokens"] <= 320
    assert meta["realized_full_token_equivalent"] == realized
    assert meta["flat_schedule_fallback"] is False


def test_pyramid_one_token_target_keeps_the_hard_bound():
    # RACER can leave one history token after charging recovered evidence; the
    # funnel's two-token layer minimum must not abort the request.
    scores = [torch.arange(300, dtype=torch.float32).expand(2, -1) for _ in range(36)]
    selected, meta = select_pyramidkv_headwise(scores, target_tokens=1)
    for indices in selected:
        torch.testing.assert_close(indices, torch.tensor([[299], [299]]))
    assert meta["flat_schedule_fallback"] is True
    assert meta["realized_full_token_equivalent"] == 1


def test_reference_attention_matches_explicit_headwise_attention():
    torch.manual_seed(0)
    source_k = torch.randn(6, 2, 4)
    source_v = torch.randn(6, 2, 4)
    indices = torch.tensor([[0, 2, 4], [1, 3, 5]])
    history = gather_reference_layer(
        source_k, source_v, torch.arange(10, 16), indices
    )
    normal_k = torch.randn(2, 2, 4)
    normal_v = torch.randn(2, 2, 4)
    query = torch.randn(2, 4, 4)
    q_pos = torch.tensor([22, 23])
    out = reference_sdpa(
        query, history, normal_k, normal_v, torch.tensor([20, 21]), q_pos,
        scale=0.5,
    )

    expected = []
    for head in range(4):
        kv_head = head // 2
        keys = torch.cat([history.key[kv_head], normal_k[:, kv_head]], dim=0)
        values = torch.cat([history.value[kv_head], normal_v[:, kv_head]], dim=0)
        logits = query[:, head].float() @ keys.float().T * 0.5
        probs = torch.softmax(logits, dim=-1)
        expected.append(probs.to(values.dtype) @ values)
    expected = torch.stack(expected, dim=1)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_reference_layer_reports_real_tensor_bytes():
    layer = ReferenceLayerKV(
        key=torch.zeros(2, 3, 4, dtype=torch.bfloat16),
        value=torch.zeros(2, 3, 4, dtype=torch.bfloat16),
        positions=torch.zeros(2, 3, dtype=torch.long),
    )
    assert layer.resident_bytes == 2 * 2 * 3 * 4 * 2 + 2 * 3 * 8
