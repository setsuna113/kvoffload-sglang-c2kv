"""CPU-only regressions for periodic reference decode persistence."""

import ast
import math
import sys
import time
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch


ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / "python"
sys.path.insert(0, str(PYTHON))


def _method(path: Path, class_name: str, method_name: str, namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == class_name
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


def test_first_turn_decode_checkpoint_materializes_state_and_refreshes_report():
    scheduler_path = ROOT / "python/sglang/srt/managers/scheduler.py"
    checkpoint = _method(
        scheduler_path,
        "Scheduler",
        "_apply_reference_decode_checkpoint",
        {"math": math, "Req": object},
    )
    row = torch.zeros((1, 160), dtype=torch.int64)
    row[0, :130] = torch.arange(1, 131)
    key = torch.arange(400, dtype=torch.float32).reshape(200, 2, 1)
    value = key + 1000
    cache = SimpleNamespace(
        start_layer=0,
        layer_num=1,
        _get_key_buffer=lambda _: key,
        _get_value_buffer=lambda _: value,
    )
    freed = []
    allocator = SimpleNamespace(
        page_size=1,
        get_kvcache=lambda: cache,
        free=lambda slots: freed.extend(int(item) for item in slots.tolist()),
        available_size=lambda: 1000,
    )
    layer = SimpleNamespace(
        key=torch.zeros(2, 128, 1),
        value=torch.zeros(2, 128, 1),
        positions=torch.arange(128).expand(2, -1),
    )
    state = SimpleNamespace(
        layers={0: layer},
        resident_bytes=(layer.key.numel() + layer.value.numel()) * 4
        + layer.positions.numel() * 8,
        selection_metadata={"method": "agentkv"},
    )
    observed = {}

    def build(req, config):
        observed.update(config)
        assert req.history_kv_reference_state is None
        return state

    scheduler = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        token_to_kv_pool_allocator=allocator,
        _build_agentkv_reference_state=build,
        _build_commitkv_reference_state=lambda *_: None,
        _bytes_per_kv_token=lambda: 16,
    )
    req = SimpleNamespace(
        history_kv_reference_config={
            "method": "agentkv",
            "checkpoint_interval": 128,
            "target_tokens": 2048,
        },
        history_kv_reference_state=None,
        history_kv_eviction={},
        history_kv_resident_positions=[0, 1],
        persistent_decode_cache_locs=[torch.tensor(129)],
        decode_batch_idx=128,
        reference_decode_protected_len=2,
        reference_decode_logical_start=2,
        kv_committed_len=130,
        kv_allocated_len=130,
        already_computed=130,
        c2kv_position_correction=0,
        req_pool_idx=0,
        kv_memory_report={},
    )

    delta = checkpoint(scheduler, req)

    assert delta == -128
    assert observed["history_start"] == 2 and observed["history_end"] == 130
    assert req.history_kv_reference_state is state
    assert req.kv_committed_len == 2
    assert req.c2kv_position_correction == 128
    assert req.history_kv_resident_positions == [0, 1]
    assert req.persistent_decode_cache_locs == []
    assert req.kv_memory_report["active_history_kv_tokens"] == 128
    assert req.kv_memory_report["reference_history_token_slots"] == 256
    assert req.kv_memory_report["reference_decode_checkpoints"][0][
        "full_history_reprefill_performed"
    ] is False
    assert set(freed) == set(range(3, 131))


def test_checkpoint_finish_and_next_turn_append_reuse_only_exact_resident_stream():
    cache_path = ROOT / "python/sglang/srt/mem_cache/session_aware_cache.py"
    serving_path = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    controller_path = ROOT / "python/sglang/srt/managers/session_controller.py"

    discard = _method(
        cache_path,
        "SessionAwareCache",
        "_discard_persistent_decode_suffix",
        {"torch": torch, "Req": object},
    )
    cache_tree = ast.parse(cache_path.read_text(encoding="utf-8"))
    slot_nodes = [
        node
        for node in cache_tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"_VirtualNode", "SessionSlot"}
    ]
    slot_namespace = {
        "dataclass": __import__("dataclasses").dataclass,
        "field": __import__("dataclasses").field,
        "Any": __import__("typing").Any,
        "Optional": Optional,
        "Req": object,
    }
    exec(
        compile(ast.Module(body=slot_nodes, type_ignores=[]), str(cache_path), "exec"),
        slot_namespace,
    )
    SessionSlot = slot_namespace["SessionSlot"]

    row = torch.zeros((1, 16), dtype=torch.int64)
    row[0, :5] = torch.arange(40, 45)
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        page_size=1,
        token_to_kv_pool_allocator=SimpleNamespace(free=lambda _: None),
    )
    external_state = object()
    output_ids = list(range(100, 107))  # logical positions 5..11; 11 has no KV yet
    last_req = SimpleNamespace(
        origin_input_ids=[10, 12, 14],
        origin_input_ids_unpadded=[10, 12, 14],
        output_ids=output_ids,
        sampling_params=SimpleNamespace(max_new_tokens=64),
        req_pool_idx=0,
        kv_committed_len=5,
        kv_allocated_len=5,
        already_computed=5,
        swa_evicted_seqlen=0,
        c2kv_position_correction=6,
        history_kv_resident_positions=[0, 2, 4],
        history_kv_score_state={},
        history_kv_reference_state=external_state,
        history_kv_reference_config={"method": "agentkv"},
        history_kv_runtime_state=object(),
        reference_decode_logical_start=5,
        reference_decode_persistent_state=None,
        reference_decode_baseline_runtime_state=None,
        persistent_decode_cache_locs=[],
        kv_memory_report={},
        last_node=None,
        cache_protected_len=0,
        swa_uuid_for_lock=None,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        mamba_next_track_idx=None,
        mamba_last_track_seqlen=None,
        mamba_branching_seqlen=None,
        finished_reason=object(),
        multimodal_inputs=None,
    )

    discard(owner, last_req)
    assert last_req.history_kv_resident_positions == [0, 2, 4, 9, 10]
    assert last_req.persistent_session_active_output_ids == [104, 105, 106]
    assert last_req.kv_memory_report[
        "persistent_session_computed_logical_horizon"
    ] == 11
    assert last_req.kv_memory_report[
        "persistent_session_uncomputed_output_tokens"
    ] == 1

    slot = SessionSlot()
    slot.save_from_req(last_req, is_first=True)
    assert slot.history_kv_reference_state is external_state

    prepare = _method(
        serving_path,
        "OpenAIServingChat",
        "_prepare_persistent_history_delta",
        {"ChatCompletionRequest": object, "List": list, "Optional": Optional},
    )
    commit = _method(
        serving_path,
        "OpenAIServingChat",
        "_commit_persistent_history_session",
        {"GenerateReqInput": object, "List": list, "Dict": Dict, "Any": object},
    )
    serving = SimpleNamespace(
        _persistent_history_sessions={},
        _persistent_history_generation_prefixes={},
        _persistent_history_generation_bases={},
        _persistent_history_computed_prefixes={},
        _persistent_history_exact_output={},
        _is_persistent_history_request=lambda _: True,
        _translate_tool_session_coordinates=lambda *_: None,
    )
    canonical_prompt = [10, 11, 12, 13, 14]
    adapted = SimpleNamespace(
        _persistent_history_session_id="s",
        _persistent_history_canonical_prompt_ids=canonical_prompt,
        _persistent_history_generation_prefix_ids=[14],
        c2kv_kv_memory_hint={
            "history_kv_reference_config": {"method": "agentkv"}
        },
    )
    ret = [
        {
            "text": "raw actor output",
            "output_ids": output_ids,
            "meta_info": {"kv_memory_report": last_req.kv_memory_report},
        }
    ]
    serving._persistent_history_requests = {("s", id(adapted)): adapted}
    commit(serving, adapted, ret)
    receipt = ret[0]["meta_info"]["persistent_history_session"]
    assert receipt["continuation_mode"] == "exact_generated_prefix"
    assert receipt["recovery_append_supported"] is False
    assert receipt["generated_text"] == "raw actor output"
    assert receipt["computed_prefix_tokens"] == 11

    full_next_prompt = canonical_prompt + output_ids + [200, 201]
    request = SimpleNamespace(
        stream=False,
        session_params={"id": "s"},
        c2kv_kv_memory_hint={
            "persistent_history_session": {"enabled": True, "session_id": "s"},
            "history_kv_eviction": {
                "method": "agentkv",
                "history_start": 1,
                "history_end": 12,
            },
        },
    )
    delta, session_id, _ = prepare(serving, request, full_next_prompt)
    assert session_id == "s" and delta == [200, 201]
    assert request.session_params["drop_previous_output"] is False
    assert request.c2kv_kv_memory_hint[
        "persistent_session_computed_prefix_tokens"
    ] == 11

    class FakeReq:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def set_finish_with_abort(self, message):
            self.abort_message = message

    controller_tree = ast.parse(controller_path.read_text(encoding="utf-8"))
    controller_nodes = [
        node
        for node in controller_tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"SessionReqNode", "Session"}
    ]
    controller_namespace = {
        "Optional": Optional,
        "Dict": Dict,
        "Req": FakeReq,
        "TokenizedGenerateReqInput": object,
        "FINISH_ABORT": object,
        "logging": __import__("logging"),
        "time": time,
        "uuid": uuid,
    }
    exec(
        compile(
            ast.Module(body=controller_nodes, type_ignores=[]),
            str(controller_path),
            "exec",
        ),
        controller_namespace,
    )
    Session = controller_namespace["Session"]
    SessionReqNode = controller_namespace["SessionReqNode"]
    session = Session(1024, session_id="s", streaming=True)
    last_req.finished = lambda: True
    session.req_nodes["turn1"] = SessionReqNode(last_req)
    params = SimpleNamespace(
        replace=False,
        drop_previous_output=False,
        offset=0,
        rid=None,
    )
    incoming = SimpleNamespace(
        rid="turn2",
        input_ids=delta,
        session_params=params,
        c2kv_kv_memory_hint=request.c2kv_kv_memory_hint,
        mm_inputs=None,
        sampling_params=SimpleNamespace(max_new_tokens=64),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        priority=None,
        routing_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )
    next_req = session.create_req(incoming, tokenizer=None, vocab_size=1000)
    assert next_req.origin_input_ids == [10, 12, 14, 104, 105, 106, 200, 201]

    next_req.history_kv_reference_config = {"method": "agentkv"}
    slot.restore_to_req(next_req)
    assert next_req.kv_committed_len == 5
    assert next_req.history_kv_reference_state is external_state
    from sglang.srt.mem_cache.history_kv_lifecycle import append_resident_positions

    next_positions = append_resident_positions(
        next_req.history_kv_resident_positions,
        request.c2kv_kv_memory_hint["persistent_session_computed_prefix_tokens"],
        len(full_next_prompt),
    )
    assert next_positions == [0, 2, 4, 9, 10, 11, 12, 13]
    # Positions 5 and 7 remain only in the external state. Evicted positions
    # 6 and 8 never re-enter the normal token stream.
    assert {5, 7}.isdisjoint(next_positions)
    assert {6, 8}.isdisjoint(next_positions)
