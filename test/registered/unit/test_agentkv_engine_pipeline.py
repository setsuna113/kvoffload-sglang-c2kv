"""CPU-only fakes around the actual AgentKV serving, model, and scheduler methods."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.agentkv import (
    AGENTKV_STAGE_ACT,
    AgentKVQueryRing,
)


def _method(path: Path, class_name: str, method_name: str, namespace: dict):
    """Compile one real method without importing its heavyweight engine module."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == class_name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method_name]


def test_serving_resolves_exact_events_and_agentkv_decode_markers() -> None:
    serving_path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    resolve = _method(
        serving_path,
        "OpenAIServingChat",
        "_resolve_history_kv_event_token_spans",
        {"ChatCompletionRequest": object, "List": list},
    )

    prefixes = {
        1: [10, 11],
        2: [10, 11, 20, 21, 22],
        3: [10, 11, 20, 21, 22, 30, 31],
    }
    tokenizer = SimpleNamespace(
        encode=lambda marker, add_special_tokens=False: {
            "<think>": [101],
            "</think>": [102],
            "<tool_call>": [201, 202],
            "</tool_call>": [203, 204],
        }[marker]
    )
    owner = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(tokenizer=tokenizer),
        _chat_template_tools=lambda request: ["tool"],
        _c2kv_contextual_prefix_ids=(
            lambda request, count, tools, prompt_ids: prefixes[count]
        ),
        _c2kv_first_message_start_offset=lambda request, message, tools: 1,
    )
    config = {"method": "agentkv"}
    request = SimpleNamespace(
        messages=[object(), object(), object()],
        c2kv_kv_memory_hint={
            "history_kv_event_messages": [
                {"role": "system", "phase": "others", "message_index": 0},
                {"role": "assistant", "phase": "act", "message_index": 1},
                {"role": "tool", "phase": "tool", "message_index": 2},
            ],
            "history_kv_reference_config": config,
        },
    )

    resolve(owner, request, prefixes[3] + [99])

    hint = request.c2kv_kv_memory_hint
    assert hint["history_kv_event_token_spans"] == [
        {"role": "system", "phase": "others", "start": 1, "end": 2,
         "message_index": 0},
        {"role": "assistant", "phase": "act", "start": 2, "end": 5,
         "message_index": 1},
        {"role": "tool", "phase": "tool", "start": 5, "end": 7,
         "message_index": 2},
    ]
    assert hint["history_kv_event_generation_suffix_start"] == 7
    assert {tuple(item["token_ids"]): item["stage"] for item in
            config["agentkv_marker_stage_sequences"]}[(201, 202)] == AGENTKV_STAGE_ACT


class _DecodeMode:
    @staticmethod
    def is_decode() -> bool:
        return True

    @staticmethod
    def is_extend_or_draft_extend_or_mixed() -> bool:
        return False


def _decode_batch(ring: AgentKVQueryRing, token_id: int, position: int):
    return SimpleNamespace(
        history_kv_reference_configs=[{
            "method": "agentkv",
            "event_token_spans": [],
            "agentkv_marker_stage_sequences": [
                {"token_ids": [201, 202], "stage": AGENTKV_STAGE_ACT}
            ],
        }],
        history_kv_runtime_states=[ring],
        forward_mode=_DecodeMode(),
        batch_size=1,
        input_ids=torch.tensor([token_id], dtype=torch.long),
    )


def _capture_decode(ring: AgentKVQueryRing, token_id: int, position: int) -> None:
    qwen_path = ROOT / "python/sglang/srt/models/qwen3.py"
    capture = _method(
        qwen_path,
        "Qwen3Attention",
        "_capture_history_kv_runtime_queries",
        {"torch": torch, "ForwardBatch": object},
    )
    owner = SimpleNamespace(num_heads=2, head_dim=2,
                            attn=SimpleNamespace(layer_id=0))
    query = torch.tensor([[position, 1.0, position + 0.5, 2.0]])
    capture(
        owner,
        query,
        torch.empty((0,)),
        torch.empty((0,)),
        torch.tensor([position], dtype=torch.long),
        _decode_batch(ring, token_id, position),
    )


def test_no_thinking_decode_marker_crosses_tokens_and_reassigns_to_act() -> None:
    ring = AgentKVQueryRing()

    _capture_decode(ring, 201, 40)
    assert ring.rows_by_stage(0) == [0, 1, 0, 0]

    # The marker completes on the next decode step. Both its prior token and
    # its current token move to act; subsequent action content stays there.
    _capture_decode(ring, 202, 41)
    _capture_decode(ring, 999, 42)

    assert ring.rows_by_stage(0) == [0, 0, 3, 0]
    _, positions = ring.read_layer(0, stages=[AGENTKV_STAGE_ACT])
    assert positions.tolist() == [40, 41, 42]


def _session_method(name: str):
    path = ROOT / "python/sglang/srt/mem_cache/session_aware_cache.py"
    return _method(path, "SessionSlot", name, {"Req": object})


def test_first_turn_query_ring_survives_session_and_drives_scheduler_selection() -> None:
    ring = AgentKVQueryRing()
    _capture_decode(ring, 201, 40)
    _capture_decode(ring, 202, 41)
    _capture_decode(ring, 999, 42)

    first = SimpleNamespace(
        req_pool_idx=0,
        kv_committed_len=40,
        kv_allocated_len=40,
        swa_evicted_seqlen=0,
        c2kv_position_correction=0,
        history_kv_resident_positions=list(range(40)),
        history_kv_score_state={},
        history_kv_reference_state=None,
        history_kv_reference_config={"method": "agentkv"},
        history_kv_runtime_state=ring,
        last_node=None,
        cache_protected_len=0,
        swa_uuid_for_lock=None,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        mamba_next_track_idx=None,
        mamba_last_track_seqlen=None,
        mamba_branching_seqlen=None,
    )
    slot = SimpleNamespace()
    _session_method("save_from_req")(slot, first, True)

    second = SimpleNamespace(history_kv_reference_config={"method": "agentkv"})
    _session_method("restore_to_req")(slot, second)
    assert second.history_kv_runtime_state is ring
    assert ring.rows_by_stage(0) == [0, 0, 3, 0]

    scheduler_path = ROOT / "python/sglang/srt/managers/scheduler.py"
    build = _method(
        scheduler_path,
        "Scheduler",
        "_build_agentkv_reference_state",
        {"Req": object},
    )
    key = torch.randn((64, 1, 2), generator=torch.Generator().manual_seed(1))
    value = torch.randn((64, 1, 2), generator=torch.Generator().manual_seed(2))
    cache = SimpleNamespace(
        start_layer=0,
        layer_num=1,
        get_kv_buffer=lambda layer_id: (key, value),
    )
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.long).view(1, 64)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: cache),
    )
    config = {"history_start": 0, "history_end": 40, "target_tokens": 28}

    state = build(owner, second, config)

    assert state.method == "agentkv"
    assert state.selection_metadata["per_layer_query_rows"] == [3]
    assert state.selection_metadata["per_layer_query_rows_by_stage"] == [[0, 0, 3, 0]]
    assert tuple(state.layer(0).key.shape) == (1, 28, 2)
    assert config["runtime_status_override"] == "reference_attention_ok"
