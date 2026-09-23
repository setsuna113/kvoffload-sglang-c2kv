"""CPU integration contracts for CommitKV's serving capture and builder."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List

import torch
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.commitkv import (  # noqa: E402
    CommitKVConfig,
    CommitKVRuntimeState,
    EventPage,
)
from sglang.srt.mem_cache.history_kv_reference import (  # noqa: E402
    CommitKVServingState,
)


def _extract_method(path: Path, class_name: str, method_name: str):
    """Load one production method without importing the full serving stack."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "List": List,
        "math": math,
        "torch": torch,
        "ForwardBatch": object,
        "paper_telemetry": SimpleNamespace(
            bind_request=lambda *args, **kwargs: None,
            sample=lambda *args, **kwargs: None,
        ),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


QWEN_CAPTURE = _extract_method(
    ROOT / "python" / "sglang" / "srt" / "models" / "qwen3.py",
    "Qwen3Attention",
    "_capture_history_kv_runtime_queries",
)
QWEN_NORMAL_POSITIONS = _extract_method(
    ROOT / "python" / "sglang" / "srt" / "models" / "qwen3.py",
    "Qwen3Attention",
    "_reference_normal_positions",
)
SCHEDULER_BUILD = _extract_method(
    ROOT / "python" / "sglang" / "srt" / "managers" / "scheduler.py",
    "Scheduler",
    "_build_commitkv_reference_state",
)
SCHEDULER_INIT = _extract_method(
    ROOT / "python" / "sglang" / "srt" / "managers" / "scheduler.py",
    "Scheduler",
    "_init_c2kv_kv_memory_report",
)
SERVING_RESOLVE_RANGE = _extract_method(
    ROOT / "python" / "sglang" / "srt" / "entrypoints" / "openai" / "serving_chat.py",
    "OpenAIServingChat",
    "_resolve_history_kv_eviction_range",
)


class _ForwardMode:
    def __init__(self, mode: str):
        self.mode = mode

    def is_decode(self) -> bool:
        return self.mode == "decode"

    def is_extend_or_draft_extend_or_mixed(self) -> bool:
        return self.mode == "extend"


class _KVPool:
    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor]):
        self.start_layer = 0
        self.layer_num = len(keys)
        self._keys = keys
        self._values = values

    def get_kv_buffer(self, layer_id: int):
        return self._keys[layer_id], self._values[layer_id]


def _resolved_commitkv_hint(history_tokens: int, *, total_budget: int = 2048):
    hint = {
        "history_kv_eviction": {
            "method": "commitkv",
            "history_start_message_count": 1,
            "history_message_count": 2,
            "target_tokens": total_budget,
        },
        "history_kv_reference_config": {
            "method": "commitkv",
            "target_tokens": total_budget,
            "measurement_layer_id": 0,
        },
    }
    request = SimpleNamespace(messages=[object(), object()], c2kv_kv_memory_hint=hint)
    completed_ids = list(range(history_tokens + 1))
    owner = SimpleNamespace(
        _chat_template_tools=lambda request: None,
        _c2kv_chat_template_input_ids=(
            lambda request, messages, tools: (
                [0] if len(messages) == 1 else completed_ids
            )
        ),
        _find_token_subsequence=lambda haystack, needle: 0,
    )
    SERVING_RESOLVE_RANGE(owner, request, completed_ids + [history_tokens + 1])
    return hint


def test_long_history_scans_latest_fully_resident_pages_under_project_cap():
    config = CommitKVConfig(
        window_size=1,
        page_size=1,
        max_scanned_pages=64,
        measurement_layer_id=0,
    )
    policy = CommitKVRuntimeState(config)
    state = CommitKVServingState(policy=policy, target_tokens=128)
    existing = tuple(
        (index, "tool", "tool", index, index + 1) for index in range(70)
    )
    state.event_signature = existing
    state.event_pages = tuple(
        EventPage(index, 0, index, index + 1) for index in range(70)
    )
    state.record_decode_window(
        torch.ones(1, 1, 1),
        torch.tensor([70]),
        torch.ones(1, 70, 1),
        torch.arange(70, dtype=torch.float32).reshape(1, 70, 1),
        torch.arange(70),
        scale=1.0,
    )
    assert len(state.pre_pages) == 64
    assert [page.start for page in state.pre_pages] == list(range(6, 70))
    assert state.pre_scan_metadata["scan_truncated_pages"] == 6

    next_spans = [
        {
            "message_index": index,
            "role": "tool",
            "phase": "tool",
            "start": index,
            "end": index + 1,
        }
        for index in range(71)
    ]
    state.configure_events(next_spans)
    receipt = state.receipts[-1]
    assert receipt["scanned_pages"] == 64
    assert receipt["scan_policy"] == (
        "latest_fully_resident_pages_project_convention"
    )


def _fake_attention(layer_id: int = 0):
    attention = SimpleNamespace(
        attn=SimpleNamespace(layer_id=layer_id),
        num_heads=1,
        num_kv_heads=1,
        head_dim=1,
        scaling=1.0,
    )
    attention._reference_normal_positions = QWEN_NORMAL_POSITIONS
    attention._capture_history_kv_runtime_queries = MethodType(
        QWEN_CAPTURE, attention
    )
    return attention


def _forward_batch(
    *,
    mode: str,
    config: dict,
    state: CommitKVServingState,
    position: int | list[int],
    seq_len: int,
    req_to_token: torch.Tensor,
    kv_pool: _KVPool,
):
    positions = [position] if isinstance(position, int) else list(position)
    batch = SimpleNamespace(
        history_kv_reference_configs=[config],
        history_kv_runtime_states=[state],
        history_kv_reference_states=[None],
        history_kv_resident_positions=[list(range(seq_len))],
        forward_mode=_ForwardMode(mode),
        batch_size=1,
        extend_seq_lens_cpu=([len(positions)] if mode == "extend" else None),
        input_ids=torch.arange(len(positions), dtype=torch.long),
        seq_lens=torch.tensor([seq_len], dtype=torch.long),
        req_pool_indices=torch.tensor([0], dtype=torch.long),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=kv_pool,
    )
    return torch.tensor(positions, dtype=torch.long), batch


def test_qwen_commitkv_capture_excludes_prompt_and_pairs_cross_turn_windows():
    policy = CommitKVRuntimeState(
        CommitKVConfig(
            measurement_layer_id=0,
            window_size=2,
            page_size=1,
            pending_fraction=0.5,
            max_scanned_pages=8,
            max_pending_pages=2,
        )
    )
    state = CommitKVServingState(policy=policy, target_tokens=4)
    attention = _fake_attention()
    req_to_token = torch.arange(16, dtype=torch.long).view(1, -1)
    key = torch.linspace(0.1, 1.6, 16).view(16, 1, 1)
    value = torch.linspace(1.0, 16.0, 16).view(16, 1, 1)
    kv_pool = _KVPool([key], [value])
    config = {
        "method": "commitkv",
        "event_token_spans": [
            {
                "message_index": 1,
                "role": "assistant",
                "phase": "act",
                "start": 0,
                "end": 1,
            }
        ],
    }

    # Prompt/extend queries configure event boundaries but must never enter
    # either CommitKV measurement window.
    positions, batch = _forward_batch(
        mode="extend",
        config=config,
        state=state,
        position=[0, 1, 2],
        seq_len=3,
        req_to_token=req_to_token,
        kv_pool=kv_pool,
    )
    attention._capture_history_kv_runtime_queries(
        torch.ones(3, 1), torch.ones(3, 1), torch.ones(3, 1), positions, batch
    )
    assert state.pre_queries == []
    assert state.post_queries == []

    # The final W decoded queries become the rolling pre-commit window.
    for position in (3, 4):
        positions, batch = _forward_batch(
            mode="decode",
            config=config,
            state=state,
            position=position,
            seq_len=position + 1,
            req_to_token=req_to_token,
            kv_pool=kv_pool,
        )
        attention._capture_history_kv_runtime_queries(
            torch.tensor([[float(position)]]),
            torch.tensor([[float(position) / 10]]),
            torch.tensor([[float(position)]]),
            positions,
            batch,
        )
    assert torch.cat(state.pre_positions).tolist() == [3, 4]
    assert state.pre_window.query_positions.tolist() == [3, 4]

    # The next prompt exposes a newly completed tool event. It seals the old
    # rolling pre window but contributes no query to the post window.
    config["event_token_spans"] = [
        *config["event_token_spans"],
        {
            "message_index": 2,
            "role": "tool",
            "phase": "tool",
            "start": 5,
            "end": 7,
        },
    ]
    positions, batch = _forward_batch(
        mode="extend",
        config=config,
        state=state,
        position=[5, 6],
        seq_len=7,
        req_to_token=req_to_token,
        kv_pool=kv_pool,
    )
    attention._capture_history_kv_runtime_queries(
        torch.full((2, 1), 99.0),
        torch.ones(2, 1),
        torch.ones(2, 1),
        positions,
        batch,
    )
    assert state.pending_commit_id == 2
    assert state.post_queries == []
    assert policy.pending is not None
    assert policy.pending.commit_id == 2

    # Only the first W decoded queries after the prompt form the post window.
    for position in (7, 8):
        positions, batch = _forward_batch(
            mode="decode",
            config=config,
            state=state,
            position=position,
            seq_len=position + 1,
            req_to_token=req_to_token,
            kv_pool=kv_pool,
        )
        attention._capture_history_kv_runtime_queries(
            torch.tensor([[float(position)]]),
            torch.tensor([[float(position) / 10]]),
            torch.tensor([[float(position)]]),
            positions,
            batch,
        )
        if position == 7:
            assert torch.cat(state.post_positions).tolist() == [7]
            assert policy.pending is not None
            assert policy.completed_transitions == 0
    assert state.pending_commit_id is None
    assert state.post_queries == []
    assert policy.pending is None
    assert policy.completed_transitions == 1
    assert [receipt["measurement_phase"] for receipt in state.receipts] == [
        "pre_commit",
        "post_commit",
    ]
    assert torch.cat(state.pre_positions).tolist() == [7, 8]


@pytest.mark.parametrize("short_queries", [0, 1, 5, 7])
def test_qwen_short_turn_is_unmeasured_and_later_full_turns_still_complete(
    short_queries,
):
    """ACE-style short actions must not crash or borrow another turn's Qs."""

    policy = CommitKVRuntimeState(
        CommitKVConfig(measurement_layer_id=0, window_size=8, page_size=1)
    )
    state = CommitKVServingState(policy=policy, target_tokens=32)
    attention = _fake_attention()
    req_to_token = torch.arange(128).view(1, -1)
    key = torch.ones(128, 1, 1)
    kv_pool = _KVPool([key], [torch.arange(128).float().view(128, 1, 1)])
    config = {
        "method": "commitkv",
        "event_token_spans": [
            {
                "message_index": 0,
                "role": "assistant",
                "phase": "act",
                "start": 0,
                "end": 1,
            }
        ],
    }

    def capture(mode, positions):
        positions, batch = _forward_batch(
            mode=mode,
            config=config,
            state=state,
            position=positions,
            seq_len=max(positions) + 1,
            req_to_token=req_to_token,
            kv_pool=kv_pool,
        )
        q = torch.ones(len(positions), 1)
        attention._capture_history_kv_runtime_queries(q, q, q, positions, batch)

    def observe(index, start):
        config["event_token_spans"].append(
            {
                "message_index": index,
                "role": "tool",
                "phase": "tool",
                "start": start,
                "end": start + 2,
            }
        )
        capture("extend", [start, start + 1])

    capture("extend", [0, 1, 2])
    for pos in range(3, 11):
        capture("decode", [pos])
    observe(1, 11)
    assert policy.pending.commit_id == 1
    assert state.pre_window is None
    assert state.pre_queries == []

    for pos in range(13, 13 + short_queries):
        capture("decode", [pos])
    boundary = 13 + short_queries
    observe(2, boundary)
    assert policy.pending is None
    assert state.pending_commit_id is None
    assert policy.completed_transitions == 0
    assert policy.incomplete_transitions == 1
    assert policy.retired_pages == {}
    assert state.post_queries == state.pre_queries == []
    assert state.pre_window is None
    assert [r["measurement_phase"] for r in state.receipts] == [
        "pre_commit",
        "post_commit_unavailable",
        "pre_commit_unavailable",
    ]
    assert state.receipts[-2]["observed_query_count"] == short_queries
    assert state.receipts[-1]["observed_query_count"] == short_queries
    assert state.event_signature[-1][0] == 2

    # A repeated prefill chunk is idempotent and cannot consume the event twice.
    before = list(state.receipts)
    capture("extend", [boundary, boundary + 1])
    assert state.receipts == before

    # The next full segment supplies only its own queries for a fresh pre.
    start = boundary + 2
    for pos in range(start, start + 8):
        capture("decode", [pos])
    assert state.pre_window.query_positions.tolist() == list(range(start, start + 8))
    observe(3, start + 8)
    assert policy.pending.commit_id == 3
    for pos in range(start + 10, start + 18):
        capture("decode", [pos])
    assert policy.completed_transitions == 1
    assert policy.incomplete_transitions == 1
    assert policy.pending is None
    assert state.receipts[-1]["measurement_phase"] == "post_commit"


def test_batched_tool_observations_share_one_transition():
    policy = CommitKVRuntimeState(CommitKVConfig(measurement_layer_id=0))
    state = CommitKVServingState(policy=policy, target_tokens=128)
    state.pre_window = _FixedEffectWindow(list(range(16)))
    spans = [
        {"message_index": i, "role": "tool", "phase": "tool", "start": i, "end": i + 1}
        for i in (1, 2)
    ]
    state.configure_events(spans)
    state.configure_events(spans)
    assert policy.pending.commit_id == 2
    assert len(state.receipts) == 1
    assert len(state.event_pages) == 2


def test_new_user_input_closes_pending_without_creating_a_tool_commit():
    policy = CommitKVRuntimeState(CommitKVConfig(measurement_layer_id=0))
    state = CommitKVServingState(policy=policy, target_tokens=128)
    state.pre_window = _FixedEffectWindow(list(range(16)))
    spans = [
        {"message_index": 1, "role": "tool", "phase": "tool", "start": 16, "end": 20}
    ]
    state.configure_events(spans)
    spans.append(
        {"message_index": 2, "role": "user", "phase": "others", "start": 20, "end": 24}
    )
    state.configure_events(spans)
    assert policy.pending is None
    assert state.pending_commit_id is None
    assert policy.incomplete_transitions == 1
    assert policy.completed_transitions == 0
    assert [r["measurement_phase"] for r in state.receipts] == [
        "pre_commit",
        "post_commit_unavailable",
    ]


class _FixedEffectWindow:
    def __init__(self, key_positions: list[int]):
        self.key_positions = torch.tensor(key_positions, dtype=torch.long)
        self.query_positions = torch.tensor(key_positions[-8:], dtype=torch.long)

    def effect(self, indices):
        return torch.tensor(0.5 + 0.01 * len(tuple(indices)))


def test_scheduler_commitkv_builder_protects_pending_and_uses_common_indices():
    policy = CommitKVRuntimeState(
        CommitKVConfig(
            measurement_layer_id=1,
            page_size=1,
            pending_fraction=0.5,
            max_pending_pages=1,
        )
    )
    policy.record_pre(
        "commit",
        [EventPage("old-action", 0, 0, 1)],
        _FixedEffectWindow([0, 1, 2, 3]),
        [0, 1, 2, 3],
        total_budget=2,
    )
    serving_state = CommitKVServingState(policy=policy, target_tokens=2)
    keys = [
        (100 * layer + torch.arange(4, dtype=torch.float32)).view(4, 1, 1)
        for layer in range(2)
    ]
    values = [item + 1000 for item in keys]
    kv_pool = _KVPool(keys, values)
    scheduler = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(4, dtype=torch.long).view(1, -1)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: kv_pool
        ),
    )
    scheduler._build_commitkv_reference_state = MethodType(
        SCHEDULER_BUILD, scheduler
    )
    req = SimpleNamespace(
        req_pool_idx=0,
        history_kv_runtime_state=serving_state,
        history_kv_reference_state=None,
        history_kv_resident_positions=[0, 1, 2, 3],
    )
    config = {
        "history_start": 0,
        "history_end": 4,
        "target_tokens": 2,
    }

    state = scheduler._build_commitkv_reference_state(req, config)

    state.validate()
    assert state.expected_layer_ids == (0, 1)
    assert state.selection_metadata["baseline_policy"] == (
        "most_recent_first_project_convention"
    )
    assert state.selection_metadata["protected_pending_page_count"] == 1
    for layer_id, layer in state.layers.items():
        # Pending position 0 survives even though the explicit base policy is
        # most-recent-first. All heads and layers use the same [0, 3] indices.
        assert layer.positions.tolist() == [[0, 3]]
        torch.testing.assert_close(
            layer.key[:, :, 0],
            torch.tensor([[100.0 * layer_id, 100.0 * layer_id + 3.0]]),
        )
        torch.testing.assert_close(layer.value, layer.key + 1000)
    assert state.layers[0].positions.tolist() == state.layers[1].positions.tolist()
    assert config["commitkv_baseline_policy"] == (
        "most_recent_first_project_convention"
    )


@pytest.mark.parametrize("effective_target, expected_tokens", [(8, 8), (4, 4)])
def test_commitkv_builder_bounds_reference_only_recovery_state(
    effective_target, expected_tokens
):
    policy = CommitKVRuntimeState(
        CommitKVConfig(measurement_layer_id=0, page_size=1)
    )
    serving_state = CommitKVServingState(policy=policy, target_tokens=8)
    keys = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
    scheduler = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(8, dtype=torch.long).view(1, -1)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: _KVPool([keys], [keys + 100])
        ),
    )
    scheduler._build_commitkv_reference_state = MethodType(
        SCHEDULER_BUILD, scheduler
    )
    req = SimpleNamespace(
        req_pool_idx=0,
        history_kv_runtime_state=serving_state,
        history_kv_reference_state=None,
        history_kv_reference_config={
            "method": "commitkv",
            "target_tokens": 8,
            "racer_effective_target_tokens": 8,
        },
        history_kv_resident_positions=list(range(8)),
    )
    initial = scheduler._build_commitkv_reference_state(
        req,
        {"history_start": 0, "history_end": 8, "target_tokens": 8},
    )
    assert initial.layers[0].key.shape[1] == 8

    empty = keys[:0]
    scheduler.req_to_token_pool.req_to_token = torch.empty(
        (1, 0), dtype=torch.long
    )
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        get_kvcache=lambda: _KVPool([empty], [empty])
    )
    req.history_kv_reference_state = initial
    req.history_kv_reference_config["racer_effective_target_tokens"] = (
        effective_target
    )
    req.history_kv_resident_positions = []

    state = scheduler._build_commitkv_reference_state(
        req,
        {
            "history_start": 0,
            "history_end": 0,
            "target_tokens": effective_target,
        },
    )

    assert state.layers[0].key.shape[1] == expected_tokens
    assert state.selection_metadata["active_capacity_tokens"] == (
        effective_target
    )


def test_commitkv_absolute_budget_survives_resolver_clamps_across_turns():
    hints = [_resolved_commitkv_hint(size) for size in (137, 274, 2100)]
    assert [hint["history_kv_eviction"]["target_tokens"] for hint in hints] == [
        137,
        274,
        2048,
    ]
    assert [
        hint["history_kv_reference_config"]["target_tokens"] for hint in hints
    ] == [2048, 2048, 2048]

    scheduler = SimpleNamespace(model_config=SimpleNamespace(num_hidden_layers=1))
    scheduler._init_c2kv_kv_memory_report = MethodType(SCHEDULER_INIT, scheduler)
    req = SimpleNamespace(
        history_kv_runtime_state=None,
        history_kv_reference_config=None,
        history_kv_resident_positions=list(range(137)),
        history_kv_reference_state=None,
        req_pool_idx=0,
    )
    scheduler._init_c2kv_kv_memory_report(req, hints[0])
    serving_state = req.history_kv_runtime_state
    assert isinstance(serving_state, CommitKVServingState)
    assert serving_state.target_tokens == 2048

    serving_state.pre_window = _FixedEffectWindow(list(range(137)))
    serving_state.pre_pages = (EventPage("act-0", 0, 0, 16),)
    serving_state.configure_events(
        [
            {
                "message_index": 1,
                "role": "tool",
                "phase": "tool",
                "start": 16,
                "end": 32,
            }
        ]
    )
    assert serving_state.policy.pending.total_budget == 2048

    def build(normal_positions, effective_target, existing_state):
        count = len(normal_positions)
        keys = torch.tensor(normal_positions, dtype=torch.float32).view(count, 1, 1)
        kv_pool = _KVPool([keys], [keys + 10000])
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(count, dtype=torch.long).view(1, -1)
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: kv_pool
        )
        scheduler._build_commitkv_reference_state = MethodType(
            SCHEDULER_BUILD, scheduler
        )
        req.history_kv_resident_positions = list(normal_positions)
        req.history_kv_reference_state = existing_state
        config = {
            "history_start": 0,
            "history_end": count,
            "target_tokens": effective_target,
        }
        return scheduler._build_commitkv_reference_state(req, config), config

    state, config = build(range(137), 137, None)
    assert state.layers[0].positions.shape == (1, 137)
    assert state.selection_metadata["commitkv_total_budget_tokens"] == 2048
    assert state.selection_metadata["commitkv_request_effective_target_tokens"] == 137

    state, config = build(range(137, 274), 274, state)
    assert state.layers[0].positions.shape == (1, 274)
    assert state.selection_metadata["commitkv_total_budget_tokens"] == 2048
    assert state.selection_metadata["commitkv_request_effective_target_tokens"] == 274

    state, config = build(range(274, 2100), 2048, state)
    assert state.layers[0].positions.shape == (1, 2048)
    assert state.selection_metadata["commitkv_total_budget_tokens"] == 2048
    assert state.selection_metadata["commitkv_request_effective_target_tokens"] == 2048

    req.history_kv_reference_config = {
        **req.history_kv_reference_config,
        "target_tokens": 1024,
    }
    with pytest.raises(RuntimeError, match="COMMITKV_TOTAL_BUDGET_CHANGED"):
        build(range(2100, 2200), 1024, state)
