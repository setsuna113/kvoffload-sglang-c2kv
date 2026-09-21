"""CPU execution tests for composed injection, selection and session coordinates."""

from __future__ import annotations

import ast
import copy
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional

import pytest
import torch


SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SRT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


composition = load("test_composition_contract", "mem_cache/c2kv_composition.py")
lifecycle = load("test_composition_lifecycle", "mem_cache/history_kv_lifecycle.py")
selection = load("test_composition_selection", "mem_cache/history_kv_selection.py")
native = load("test_composition_native", "mem_cache/c2kv_native_packed.py")


def extract_class(relative, name, methods=None, extra=None):
    tree = ast.parse((SRT / relative).read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    if methods is not None:
        node.bases = []
        node.body = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in methods]
    namespace = {"Optional": Optional, "List": list, "torch": torch,
                 "logger": logging.getLogger(__name__), "_is_npu": False,
                 "pool_snapkv_scores_by_position": selection.pool_snapkv_scores_by_position,
                 "_persistent_history_session_error": lambda *args: None,
                 **(extra or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRT / relative), "exec"), namespace)
    return namespace[name]


Round = extract_class("managers/schedule_batch.py", "C2KVPrefillRound")
Scheduler = extract_class("managers/scheduler.py", "Scheduler", {
    "_build_c2kv_prefill_rounds", "_compose_c2kv_history_rounds",
    "_select_history_kv_eviction_indices", "_build_pyramidkv_reference_state",
    "_advance_cached_c2kv_tool_rounds",
    "_add_c2kv_kv_memory_tokens",
})


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.schedule_batch", SimpleNamespace(C2KVPrefillRound=Round))
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_composition", composition)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.history_kv_lifecycle", lifecycle)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_pool", SimpleNamespace(c2kv_gist_token_ids=lambda key, count: [-100 - index for index in range(count)]))
    obj = Scheduler()
    obj.page_size = 1
    obj.max_req_input_len = 1000
    obj.max_req_len = 1000
    obj._log_c2kv_token_usage = lambda *args, **kwargs: None
    obj._release_c2kv_pins = lambda req: None
    obj._C2KV_REPAIR_PLACEMENTS = ("in_place", "append_keep_ledger", "append_tail")
    entries = {"tool": SimpleNamespace(entry_type="gist", gist_len=2, original_seq_len=8, positions=[3, 7])}
    obj.c2kv_pool = SimpleNamespace(get=lambda key: entries.get(key),
        get_position_ids=lambda entry: torch.tensor(entry.positions), pin_many=lambda keys: True)
    return obj


def request(*, start=2, end=2, history_start=2, history_end=8, region="tool", method="h2o", persistent=False):
    return SimpleNamespace(
        rid="composition", origin_input_ids=list(range(12)),
        c2kv_segments=[SimpleNamespace(token_start=start, token_end=end, key_hash="tool",
            source_token_count=8, source_token_end=None, expected_token_len=None,
            region=region, repair_placement=None, repair_key_hashes=[])],
        c2kv_use_gist_projection=False, prefix_indices=[], c2kv_position_correction=0,
        sampling_params=SimpleNamespace(max_new_tokens=8),
        history_kv_eviction={"method": method, "history_start": history_start,
            "history_end": history_end, "target_tokens": 2,
            "history_kv_recent_window": 2, "persistent_session": persistent},
        c2kv_kv_memory_hint={"persistent_history_session": {"enabled": persistent}},
        kv_memory_report={}, history_kv_score_state={},
    )


def test_off_legacy_builder_keeps_rounds_and_virtual_sequence(engine):
    req = request(region=None)
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert [(r.tokens, r.post_inject_seg_indices, r.post_history_kv_eviction) for r in req.c2kv_rounds] == [
        ([0, 1], [0], False), (list(range(2, 12)), [], False)]
    assert req.c2kv_virtual_input_ids == [0, 1, -100, -101] + list(range(2, 12))
    assert not hasattr(req, "history_kv_resident_positions")


@pytest.mark.parametrize("method", ["h2o", "snapkv", "pyramidkv"])
def test_initial_tools_and_history_share_rounds_and_expanded_positions(engine, method):
    req = request(method=method, persistent=True)
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert req.history_kv_resident_positions == [0, 1, 5, 9] + list(range(10, 20))
    assert req.history_kv_eviction["history_start"] == 4
    assert req.history_kv_eviction["history_end"] == 10
    assert req.history_kv_eviction["canonical_history_start"] == 10
    assert req.history_kv_eviction["canonical_history_end"] == 16
    assert req.c2kv_rounds[0].post_inject_seg_indices == [0]
    assert req.c2kv_rounds[-1].tokens == [10, 11]
    assert sum(r.post_history_kv_eviction for r in req.c2kv_rounds) == 1
    assert req.c2kv_persistent_active_input_ids == req.c2kv_virtual_input_ids


@pytest.mark.parametrize("method", ["agentkv", "commitkv"])
def test_reference_state_history_preserves_external_tool_carrier_and_event_coordinates(engine, method):
    req = request(method=method)
    req.history_kv_reference_config = {"method": method, "event_token_spans": [
        {"start": 2, "end": 10, "role": "system", "phase": "others"},
        {"start": 10, "end": 16, "role": "user", "phase": "others"},
    ]}
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert req.history_kv_eviction["history_start"] == 4
    assert req.history_kv_eviction["history_end"] == 10
    assert req.history_kv_eviction["protected_history_indices"] == []
    assert req.c2kv_tool_source_spans == [(2, 10)]
    assert req.history_kv_reference_config["event_token_spans"] == [
        {"start": 2, "end": 10, "role": "system", "phase": "others"},
        {"start": 10, "end": 16, "role": "user", "phase": "others"},
    ]
    assert sum(round_info.post_history_kv_eviction for round_info in req.c2kv_rounds) == 1
    assert next(round_info.tokens for round_info in req.c2kv_rounds
                if round_info.post_history_kv_eviction) == [7]
    assert req.c2kv_rounds[-1].tokens == [8, 9, 10, 11]


@pytest.mark.parametrize("method", ["agentkv", "commitkv"])
def test_reference_state_history_rejects_tool_carrier_inside_history(engine, method):
    req = request(start=5, end=5, history_start=2, history_end=10, method=method)
    assert engine._build_c2kv_prefill_rounds(req) == (
        "C2KV_REFERENCE_TOOL_OVERLAP_UNSUPPORTED: " + method
    )


def test_first_turn_without_eviction_stores_physical_session_view(engine):
    req = request(persistent=True)
    req.history_kv_eviction = None
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert req.c2kv_persistent_active_input_ids == [0, 1, -100, -101] + list(range(2, 12))
    assert req.history_kv_resident_positions[-1] == 19
    assert not any(r.post_history_kv_eviction for r in req.c2kv_rounds)


def test_tools_inside_history_are_protected_and_not_charged_to_history_budget(engine):
    req = request(start=5, end=5, history_start=2, history_end=10, persistent=True)
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert req.history_kv_eviction["protected_history_indices"] == [3, 4]
    req.history_kv_selection_scores = {"layers": [torch.tensor([1., 2., 3., 999., 999., 4., 5., 6., 7., 8.])]}
    selected = engine._select_history_kv_eviction_indices(req, req.history_kv_eviction)
    assert selected == [3, 4, 8, 9]
    tool_positions = {req.history_kv_resident_positions[5], req.history_kv_resident_positions[6]}
    assert not tool_positions & set(req.history_kv_score_state[0])


def test_query_collection_preserves_npu_bridge_and_evicts_once():
    descriptors = [{"positions": [3, 7]}]
    rounds = [Round([0, 1], [0]), Round([2, 3], []), Round([4, 5, 6, 7, 8, 9], [])]
    output = composition.split_rounds_at_query(rounds, descriptors, 4, 12, Round)
    assert [len(item.tokens) for item in output] == [2, 2, 6]
    assert [item.collect_history_kv_scores for item in output] == [False, True, True]
    assert [item.post_history_kv_eviction for item in output] == [False, False, True]


def test_query_window_excludes_gists_of_newly_appended_document(engine):
    req = request(start=10, end=10, history_start=2, history_end=8)
    req.history_kv_eviction["history_kv_recent_window"] = 64
    assert engine._build_c2kv_prefill_rounds(req) is None
    # Only the two real tail tokens execute Q; injected document KV cannot be
    # counted as query observations even when the requested window is longer.
    assert req.history_kv_eviction["selection_query_tokens"] == 2
    assert req.c2kv_rounds[-1].tokens == [10, 11]


def test_append_ledger_preserves_evicted_holes_and_anchors_new_tool():
    previous = [0, 1, 5, 9, 13, 18, 19]
    descriptors = [{"token_start": 9, "token_end": 9, "source_tokens": 8,
                    "positions": [3, 7]}]
    positions = composition.resident_positions(12, descriptors, previous, 20)
    assert positions == previous + [20, 21, 25, 29, 30, 31, 32]
    assert not set(range(10, 13)) & set(positions)
    assert len(positions) == 14


def test_source_horizon_after_tool_protocol_maps_to_rendered_prompt():
    segment = {"token_start": 2, "token_end": 2, "source_tokens": 8}
    assert composition.source_boundary(12, [segment]) == 20
    assert composition.trailing_source_horizon_to_input(21, 12, [segment]) == 13
    with pytest.raises(ValueError, match="HORIZON_BEFORE_SOURCE_PROMPT"):
        composition.trailing_source_horizon_to_input(19, 12, [segment])


def test_exact_session_commit_maps_computed_source_horizon_to_rendered_prefix(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_composition", composition)
    Chat = extract_class(
        "entrypoints/openai/serving_chat.py", "OpenAIServingChat",
        {"_commit_persistent_history_session"},
        extra={"GenerateReqInput": object, "Dict": dict, "Any": object},
    )
    chat = Chat()
    chat._persistent_history_tool_segments = {}
    chat._persistent_history_generation_bases = {}
    chat._persistent_history_sessions = {}
    chat._persistent_history_computed_prefixes = {}
    chat._persistent_history_exact_output = {}
    chat._persistent_history_generation_prefixes = {}
    hint = {
        "history_kv_reference_config": {"method": "agentkv"},
        "persistent_session_canonical_prompt_tokens": 20,
        "tool_memory_segments": [
            {"token_start": 2, "token_end": 2, "source_tokens": 8}
        ],
    }
    request = SimpleNamespace(
        _persistent_history_session_id="session",
        _persistent_history_canonical_prompt_ids=list(range(12)),
        _persistent_history_generation_prefix_ids=None,
        c2kv_kv_memory_hint=hint,
    )
    result = [{"output_ids": [50, 51], "text": "ok", "meta_info": {
        "kv_memory_report": {"persistent_session_computed_logical_horizon": 21}
    }}]
    chat._persistent_history_requests = {("session", id(request)): request}
    chat._commit_persistent_history_session(request, result)
    assert len(chat._persistent_history_sessions["session"]) == 14
    assert chat._persistent_history_computed_prefixes["session"] == 13
    assert result[0]["meta_info"]["kv_memory_report"]["persistent_session_computed_input_horizon"] == 13
    assert result[0]["meta_info"]["persistent_history_session"]["computed_prefix_tokens"] == 13


def test_message_boundaries_and_event_metadata_remove_same_carriers():
    hint = {"history_kv_eviction": {"history_start_message_count": 2, "history_message_count": 5},
            "history_kv_event_messages": list("abcdef")}
    composition.remap_message_metadata(hint, [1, 3], 6)
    assert hint["history_kv_eviction"] == {"history_start_message_count": 1, "history_message_count": 3}
    assert hint["history_kv_event_messages"] == list("acef")


def test_removed_carriers_renumber_events_for_real_token_span_resolver():
    events = load("test_composition_events", "mem_cache/history_kv_events.py")
    original = [
        {"message_index": 0, "role": "system", "phase": "others"},
        {"message_index": 1, "role": "user", "phase": "others"},
        {"message_index": 2, "role": "assistant", "phase": "act", "event_id": "call"},
        {"message_index": 3, "role": "user", "phase": "others"},
        {"message_index": 4, "role": "tool", "phase": "tool", "event_id": "result"},
    ]
    hint = {"history_kv_event_messages": original}
    composition.remap_message_metadata(hint, [1, 3], 5)
    retained = hint["history_kv_event_messages"]
    assert [item["message_index"] for item in retained] == [0, 1, 2]
    assert retained[-1]["event_id"] == "result"
    assert original[-1]["message_index"] == 4
    spans = events.resolve_history_kv_event_token_spans(
        total_tokens=13, message_prefix_token_counts=[0, 5, 8, 13],
        event_messages=retained,
    )
    assert spans == [
        {"role": "system", "phase": "others", "start": 0, "end": 5, "message_index": 0},
        {"role": "assistant", "phase": "act", "start": 5, "end": 8, "message_index": 1},
        {"role": "tool", "phase": "tool", "start": 8, "end": 13, "message_index": 2},
    ]


def test_session_boundary_at_new_carrier_does_not_count_it_as_cached(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_composition", composition)
    Chat = extract_class("entrypoints/openai/serving_chat.py", "OpenAIServingChat", {"_translate_tool_session_coordinates"})
    chat = Chat()
    previous = {"token_start": 2, "token_end": 2, "source_tokens": 8, "key_hash": "a"}
    fresh = {"token_start": 12, "token_end": 12, "source_tokens": 16, "key_hash": "b"}
    chat._persistent_history_tool_segments = {"session": [previous]}
    chat._persistent_history_tool_source_digests = {"session": "fixed-source"}
    hint = {"tool_memory_segments": [previous, fresh], "persistent_session_logical_prefix_tokens": 12,
            "persistent_session_canonical_prompt_tokens": 20,
            "joint_tool_memory": {"source_protocol_token_sha256": "fixed-source"},
            "history_kv_eviction": {"persistent_canonical_history_end": 16}}
    chat._translate_tool_session_coordinates(SimpleNamespace(c2kv_kv_memory_hint=hint, session_params={"id": "session"}), 12, 20)
    assert hint["persistent_session_logical_prefix_tokens"] == 20
    assert hint["persistent_session_canonical_prompt_tokens"] == 44
    assert hint["history_kv_eviction"]["persistent_canonical_history_end"] == 40


def test_tool_session_refresh_keeps_source_frame_and_maps_event_boundaries(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_composition", composition)
    Chat = extract_class("entrypoints/openai/serving_chat.py", "OpenAIServingChat", {"_translate_tool_session_coordinates"})
    chat = Chat()
    old = {"token_start": 2, "token_end": 2, "source_tokens": 8,
           "key_hash": "", "repair_key_hashes": ["old"]}
    new = {**old, "repair_key_hashes": ["new"]}
    chat._persistent_history_tool_segments = {"session": [old]}
    chat._persistent_history_tool_source_digests = {"session": "same-prefix"}
    hint = {"tool_memory_segments": [new],
            "joint_tool_memory": {"source_protocol_token_sha256": "same-prefix"},
            "persistent_session_logical_prefix_tokens": 12,
            "history_kv_event_token_spans": [
                {"start": 0, "end": 2}, {"start": 2, "end": 5},
                {"start": 5, "end": 12}],
            "history_kv_event_generation_suffix_start": 12}
    req = SimpleNamespace(c2kv_kv_memory_hint=hint, session_params={"id": "session"})
    chat._translate_tool_session_coordinates(req, 12, 12)
    assert hint["persistent_tool_refresh"]["previous_segment"] == old
    assert hint["persistent_tool_refresh"]["new_segment"] == new
    assert hint["persistent_session_logical_prefix_tokens"] == 20
    assert hint["history_kv_event_token_spans"] == [
        {"start": 0, "end": 10}, {"start": 10, "end": 13},
        {"start": 13, "end": 20}]
    assert hint["history_kv_event_generation_suffix_start"] == 20

    changed = copy.deepcopy(hint)
    changed["joint_tool_memory"]["source_protocol_token_sha256"] = "new-source"
    with pytest.raises(ValueError, match="PERSISTENT_HISTORY_TOOL_PREFIX_CHANGED"):
        chat._translate_tool_session_coordinates(
            SimpleNamespace(c2kv_kv_memory_hint=changed, session_params={"id": "session"}),
            12, 12,
        )


def test_unchanged_t0_gist_carriers_continue_without_raw_refresh_digest(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_composition", composition)
    Chat = extract_class("entrypoints/openai/serving_chat.py", "OpenAIServingChat", {"_translate_tool_session_coordinates"})
    chat = Chat()
    segments = [
        {"token_start": 2, "token_end": 2, "source_tokens": 8,
         "key_hash": "gist-a", "repair_key_hashes": []},
        {"token_start": 4, "token_end": 4, "source_tokens": 6,
         "key_hash": "gist-b", "repair_key_hashes": []},
    ]
    chat._persistent_history_tool_segments = {"session": copy.deepcopy(segments)}
    chat._persistent_history_tool_source_digests = {}
    hint = {"tool_memory_segments": copy.deepcopy(segments),
            "persistent_session_logical_prefix_tokens": 12}
    chat._translate_tool_session_coordinates(
        SimpleNamespace(c2kv_kv_memory_hint=hint, session_params={"id": "session"}),
        12, 12,
    )
    assert hint["persistent_session_logical_prefix_tokens"] == 26
    assert "persistent_tool_refresh" not in hint


def test_cached_prefix_injects_tool_without_zero_token_forward(engine):
    req = SimpleNamespace(c2kv_tool_source_spans=[(10, 18)], c2kv_rounds=[Round([0, 1], [0]), Round([2, 3], [])],
        c2kv_round_idx=0, extend_input_len=0, kv_committed_len=2, req_pool_idx=0)
    engine.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(20).reshape(1, 20))
    engine.tree_cache = object()
    injected = []
    def inject(current, index, start):
        injected.append((index, start))
        current.kv_committed_len += 2
        return True
    engine._inject_c2kv_gist_segment = inject
    req.prepare_c2kv_round_input = lambda cache: setattr(req, "extend_input_len", 2)
    assert engine._advance_cached_c2kv_tool_rounds(req)
    assert injected == [(0, 2)]
    assert req.c2kv_round_idx == 1
    assert req.c2kv_round_start_len == 4


def test_headwise_pyramid_excludes_tool_slots_from_reference_candidates(engine, monkeypatch):
    reference = load("test_composition_reference", "mem_cache/history_kv_reference.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.history_kv_reference", reference)
    telemetry = SimpleNamespace(sample=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "sglang.srt.observability", SimpleNamespace(paper_telemetry=telemetry))
    keys = torch.arange(24, dtype=torch.float32).reshape(12, 1, 2)
    engine.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: SimpleNamespace(get_kv_buffer=lambda layer: (keys, keys + 100)))
    engine.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(12).reshape(1, 12))
    req = SimpleNamespace(req_pool_idx=0, history_kv_resident_positions=list(range(12)), history_kv_reference_state=None)
    config = {"history_start": 2, "history_end": 10, "protected_history_indices": [2, 3],
              "target_tokens": 2, "history_kv_recent_window": 1, "history_kv_kernel_size": 1}
    score = {"headwise_layers": [torch.tensor([[1., 2., 999., 999., 3., 4., 5., 6.]])], "layer_ids": [0]}
    state = engine._build_pyramidkv_reference_state(req, config, score)
    assert not {4, 5} & set(state.layer(0).positions.flatten().tolist())
    assert state.layer(0).key.shape[1] == 2


def native_chunk(name, ids, start, ratio=4, tool=False):
    return {"chunk_id": name, "event_id": name, "part_index": 0,
            "source_token_start": 0, "source_token_end": len(ids), "token_ids": list(ids),
            "source_position_start": start,
            "gist_position_ids": [start + min(index + ratio, len(ids)) - 1 for index in range(0, len(ids), ratio)],
            **({"projection_set": "tool", "compression_ratio": ratio} if tool else {})}


def native_plan(**overrides):
    args = {"system_input_ids": [0, 1], "workspace_input_ids": list(range(6, 12)),
            "encoder_chunks": [native_chunk("history", list(range(2, 6)), 2)],
            "compression_chunks": [], "model_binding": {}, "packing_version": native.PACKING_VERSION,
            "raw_layout_profile": native.RAW_LAYOUT_PROFILE, "encoding_scope": "test",
            "compression_ratio": 4, "tool_binding": {"enabled": True, "identity": "checkpoint"}}
    args.update(overrides)
    return native.plan_native_packed_request(**args)


def test_anchored_t0_preserves_encoder_envelope_and_multiple_chunk_order():
    chunks = [native_chunk("tool-a", list(range(100, 108)), 6, tool=True),
              native_chunk("tool-b", list(range(200, 204)), 14, tool=True)]
    plan = native_plan(tool_gist_segments=[{"token_start": 6, "token_end": 8, "chunks": chunks}])
    assert list(plan.logical_input_ids) == list(range(12))
    assert plan.segment_boundaries == ((2, 6), (6, 8), (8, 8))
    assert len(plan.selected_handles) == 3
    assert list(plan.unique_chunks[1]["token_ids"]) == list(range(100, 108))
    assert plan.costs["canonical_position_tokens"] == 22
    assert plan.costs["anchored_tool_encoder_tokens"] == 12
    assert plan.costs["resident_kv_tokens"] == 10
    assert plan.costs["system_prefix_kv_tokens"] + plan.costs["workspace_resident_kv_tokens"] + plan.costs["gist_prefix_kv_tokens"] == 10


def test_raw_tool_workspace_slots_replace_only_named_span():
    plan = native_plan(raw_tool_segments=[{"token_start": 7, "token_end": 11,
        "token_len": 2, "repair_key_hashes": ["repair"]}])
    assert list(plan.logical_input_ids) == list(range(12))
    assert plan.costs["resident_kv_tokens"] == 7
    assert plan.costs["system_tokens"] == 2
    assert plan.costs["raw_tokens"] == 6
    assert plan.costs["workspace_resident_kv_tokens"] == 4


def test_raw_tool_cannot_overlap_compressed_history():
    with pytest.raises(ValueError, match="C2KV_NATIVE_RAW_TOOL_SPAN_OVERLAP"):
        native_plan(raw_tool_segments=[{"token_start": 3, "token_end": 5,
            "token_len": 1, "repair_key_hashes": ["repair"]}])


def test_tool_ratio_does_not_change_history_handles():
    history = native_chunk("history", [2, 3, 4, 5], 2)
    first = native_plan(encoder_chunks=[history])
    tool = native_chunk("tool", list(range(20, 28)), 6, ratio=2, tool=True)
    second = native_plan(tool_gist_segments=[{"token_start": 6, "token_end": 8, "chunk": tool}])
    assert first.selected_handles[0] == second.selected_handles[0]
    assert second.costs["anchored_tool_gist_tokens"] == 4


def test_session_restore_keeps_history_holes_and_appends_tool_rounds(engine, monkeypatch):
    Cache = extract_class("mem_cache/session_aware_cache.py", "SessionAwareCache", {"match_prefix", "_refresh_persistent_tool_prefix", "_is_persistent_history_req"},
        extra={"MatchPrefixParams": object, "MatchResult": SimpleNamespace, "_is_streaming": lambda req: True,
               "SessionSlot": object, "Req": object})
    req = request(start=7, end=7, persistent=True)
    # A previous turn keeps seven physical positions from a twenty-token
    # source. The new document is appended immediately after that prefix.
    previous = [0, 1, 5, 9, 13, 18, 19]
    req.origin_input_ids = [0, 1, -100, -101, 13, 18, 19, 40, 41, 42]
    req.history_kv_eviction.update(persistent_continuation=True,
        persistent_protected_prefix_tokens=10, persistent_canonical_history_end=29,
        persistent_delta_history_tokens=1)
    req.c2kv_kv_memory_hint.update(persistent_session_logical_prefix_tokens=20,
        persistent_session_canonical_prompt_tokens=31)
    req.session = SimpleNamespace(session_id="session")
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert req.c2kv_composition_pending
    state = {0: {13: 4.0}}
    def restore(current):
        current.kv_committed_len = 7
        current.req_pool_idx = 0
        current.c2kv_position_correction = 13
        current.c2kv_tool_source_spans = [(2, 10)]
        current.history_kv_score_state = state
    slot = SimpleNamespace(req_pool_idx=0, kv_committed_len=7,
        c2kv_position_correction=13, history_kv_resident_positions=previous,
        restore_to_req=restore, virtual_node=object(), cache_protected_len=0)
    cache = Cache()
    cache.slots = {"session": slot}
    cache.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(32).reshape(1, 32))
    cache.match_prefix(SimpleNamespace(req=req, key=SimpleNamespace(token_ids=req.origin_input_ids)))
    assert req.history_kv_resident_positions == previous + [23, 27, 28, 29, 30]
    assert req.history_kv_score_state is state
    assert req.c2kv_tool_source_spans == [(2, 10), (20, 28)]
    assert req.history_kv_eviction["protected_history_indices"] == [3, 4]
    assert req.c2kv_rounds[0].tokens == req.origin_input_ids[:7]
    assert req.c2kv_rounds[0].post_inject_seg_indices == [0]
    assert req.c2kv_rounds[-1].tokens == [41, 42]
    assert req.c2kv_rounds[-1].post_history_kv_eviction
    assert len(req.c2kv_virtual_input_ids) == len(req.history_kv_resident_positions)


def test_persistent_match_returns_entire_resized_prefix_not_pre_refresh_key():
    Cache = extract_class(
        "mem_cache/session_aware_cache.py", "SessionAwareCache",
        {"match_prefix", "_is_persistent_history_req"},
        extra={"MatchPrefixParams": object, "MatchResult": SimpleNamespace,
               "_is_streaming": lambda req: True, "Req": object},
    )
    cache = Cache()
    old_prefix = 1789
    grown_prefix = 2120
    old_ids = list(range(old_prefix + 1))
    req = SimpleNamespace(
        session=SimpleNamespace(session_id="session"),
        c2kv_kv_memory_hint={"persistent_history_session": {"enabled": True}},
        history_kv_eviction=None,
        origin_input_ids=list(old_ids), kv_memory_report={},
    )
    def restore(current):
        current.req_pool_idx = 0
        current.kv_committed_len = old_prefix
        current.c2kv_position_correction = 100
    slot = SimpleNamespace(
        req_pool_idx=0, history_kv_resident_positions=[0] * old_prefix,
        restore_to_req=restore, virtual_node=object(), cache_protected_len=0,
    )
    cache.slots = {"session": slot}
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(grown_prefix + 32).reshape(1, -1))
    def refresh(current_slot, current_req):
        current_req.kv_committed_len = grown_prefix
        current_req.origin_input_ids = list(range(grown_prefix + 2))
    cache._refresh_persistent_tool_prefix = refresh
    result = cache.match_prefix(SimpleNamespace(
        req=req, key=SimpleNamespace(token_ids=old_ids)))
    assert len(result.device_indices) == grown_prefix
    assert req.kv_committed_len == grown_prefix


def test_persistent_match_rejects_resident_prefix_beyond_refreshed_active_input():
    Cache = extract_class(
        "mem_cache/session_aware_cache.py", "SessionAwareCache",
        {"match_prefix", "_is_persistent_history_req"},
        extra={"MatchPrefixParams": object, "MatchResult": SimpleNamespace,
               "_is_streaming": lambda req: True, "Req": object},
    )
    cache = Cache()
    req = SimpleNamespace(
        session=SimpleNamespace(session_id="session"),
        c2kv_kv_memory_hint={"persistent_history_session": {"enabled": True}},
        history_kv_eviction=None, origin_input_ids=[1, 2], kv_memory_report={},
    )
    def restore(current):
        current.req_pool_idx = 0
        current.kv_committed_len = 2
        current.c2kv_position_correction = 0
    cache.slots = {"session": SimpleNamespace(
        req_pool_idx=0, history_kv_resident_positions=[0, 1],
        restore_to_req=restore, virtual_node=object(), cache_protected_len=0)}
    cache._refresh_persistent_tool_prefix = lambda slot, current: None
    with pytest.raises(RuntimeError, match="PREFIX_EXCEEDS_ACTIVE_INPUT"):
        cache.match_prefix(SimpleNamespace(
            req=req, key=SimpleNamespace(token_ids=[1, 2])))


@pytest.mark.parametrize("new_positions", [[2, 4, 5], [4]])
@pytest.mark.parametrize("page_size", [1, 128])
def test_persistent_tool_refresh_replaces_only_tool_kv_and_preserves_reference_state(
    monkeypatch, new_positions, page_size,
):
    Cache = extract_class(
        "mem_cache/session_aware_cache.py", "SessionAwareCache",
        {"_refresh_persistent_tool_prefix"}, extra={"SessionSlot": object, "Req": object},
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_pool",
                        SimpleNamespace(c2kv_gist_token_ids=lambda key, count: [-50 - i for i in range(count)]))
    cache = Cache()
    old_view = {"token_start": 2, "token_end": 2, "source_tokens": 6,
                "key_hash": "", "repair_key_hashes": ["old"]}
    new_view = {**old_view, "repair_key_hashes": ["new"]}
    entry = SimpleNamespace(entry_type="repair", already_rotated=True)
    new_width = len(new_positions)
    source_k = torch.arange(new_width * 2, dtype=torch.float32).reshape(new_width, 1, 2) + 300
    source_v = source_k + 100
    target_k = torch.zeros(512, 1, 2)
    target_v = torch.zeros(512, 1, 2)
    old_start = 10 if page_size == 1 else 128
    old_row = list(range(old_start, old_start + 7))
    target_k[old_row] = torch.arange(14, dtype=torch.float32).reshape(7, 1, 2)
    target_v[old_row] = target_k[old_row] + 100
    preserved_rows = [old_row[i] for i in (0, 1, 4, 5, 6)]
    old_non_tool = target_k[preserved_rows].clone()
    allocated = (torch.arange(30, 30 + new_width) if page_size == 1
                 else torch.arange(256, 384))
    frees = []
    cache.c2kv_pool = SimpleNamespace(
        pin_many=lambda keys: True, unpin_many=lambda keys: None,
        get=lambda key: entry, get_position_ids=lambda item: torch.tensor(new_positions),
        get_layer_kv=lambda item, layer: (source_k, source_v), num_layers=1,
        start_layer=0,
    )
    cache.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([old_row + [0] * 9]))
    def allocate(count):
        assert count == len(allocated)
        return allocated
    cache.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=page_size, alloc=allocate,
        free=lambda loc: frees.append(loc.tolist()),
        get_kvcache=lambda: SimpleNamespace(
            get_kv_buffer=lambda layer: (target_k, target_v)),
    )
    state = object()
    slot = SimpleNamespace(
        req_pool_idx=0, c2kv_tool_view=[old_view],
        c2kv_tool_source_digest="fixed-prefix", c2kv_tool_source_spans=[(2, 8)],
        history_kv_resident_positions=[0, 1, 2, 5, 8, 10, 11],
        history_kv_reference_state=state, history_kv_score_state={0: {8: 9.0}},
        kv_committed_len=7, kv_allocated_len=7, c2kv_position_correction=5,
        cache_protected_len=0,
        c2kv_tool_kv_accounting={"active_tool_kv_tokens": 2,
                                 "active_tool_repair_tokens": 2},
    )
    old_ids = [1, 2, -1, -2, 5, 6, 7, 8]
    req = SimpleNamespace(
        c2kv_kv_memory_hint={"persistent_tool_refresh": {
            "previous_segment": old_view, "new_segment": new_view,
            "source_protocol_token_sha256": "fixed-prefix"}},
        kv_memory_report={}, c2kv_pinned_keys=[], origin_input_ids=list(old_ids),
        origin_input_ids_unpadded=list(old_ids), c2kv_virtual_input_ids=list(old_ids),
        history_kv_reference_state=state,
    )
    cache._refresh_persistent_tool_prefix(slot, req)
    expected_row = (old_row[:2] + allocated.tolist() + old_row[4:] if page_size == 1
                    else allocated[:5 + new_width].tolist())
    assert cache.req_to_token_pool.req_to_token[0, :len(expected_row)].tolist() == expected_row
    expected_freed = old_row[2:4] if page_size == 1 else old_row
    assert frees == [expected_freed]
    new_tool_rows = expected_row[2:2 + new_width]
    assert torch.equal(target_k[new_tool_rows], source_k)
    assert torch.equal(target_v[new_tool_rows], source_v)
    assert torch.equal(target_k[expected_row[:2] + expected_row[2 + new_width:]], old_non_tool)
    assert slot.history_kv_reference_state is req.history_kv_reference_state is state
    assert slot.history_kv_resident_positions == [0, 1] + new_positions + [8, 10, 11]
    assert slot.kv_committed_len + slot.c2kv_position_correction == 12
    assert req.origin_input_ids == [1, 2] + [-50 - i for i in range(new_width)] + [5, 6, 7, 8]
    assert req.kv_memory_report["active_tool_repair_tokens"] == new_width
    assert req.kv_memory_report["persistent_tool_kv_refreshed"] is True
    cache._refresh_persistent_tool_prefix(slot, req)
    assert frees == [expected_freed]
    assert req.kv_memory_report['persistent_tool_kv_refresh_rebuilt_normal_row'] == (page_size > 1)
    assert req.kv_memory_report['persistent_tool_kv_refresh_allocated_tokens'] == len(allocated)


def test_canonical_source_and_history_measurement_views_are_independent():
    Chat = extract_class("entrypoints/openai/serving_chat.py", "OpenAIServingChat", {
        "_paper_source_request", "_paper_history_request"}, extra={"copy": copy})
    class Request(SimpleNamespace):
        def model_dump(self):
            return copy.deepcopy(self.__dict__)
        @classmethod
        def model_validate(cls, payload):
            return cls(**payload)
    source = [{"role": "system", "content": "raw tool source with its original wrapper"},
              {"role": "user", "content": "previous history"},
              {"role": "user", "content": "current query"}]
    req = Request(messages=[SimpleNamespace(content="system"),
        SimpleNamespace(content="carrier envelope", c2kv_region="tool", c2kv_key_hash="gist"),
        SimpleNamespace(content="previous history"), SimpleNamespace(content="current query")], tools=[],
        c2kv_kv_memory_hint={"paper_measurement": {"history_start_message_count": 2,
            "history_message_count": 3, "canonical_source_messages": source, "canonical_source_tools": []},
            "history_kv_eviction": {"target_tokens": 7}})
    generation_before = copy.deepcopy(req.__dict__)
    history = Chat._paper_history_request(req)
    canonical = Chat._paper_source_request(req, req.c2kv_kv_memory_hint["paper_measurement"])
    assert canonical.messages == source
    assert canonical.tools == []
    assert [item.content for item in history.messages] == ["system", "previous history", "current query"]
    assert history.c2kv_kv_memory_hint["paper_measurement"]["history_start_message_count"] == 1
    assert history.c2kv_kv_memory_hint["paper_measurement"]["history_message_count"] == 2
    assert req.__dict__ == generation_before
    assert history.c2kv_kv_memory_hint["history_kv_eviction"]["target_tokens"] == 7


def test_tool_accounting_does_not_change_history_denominator_or_residency(engine, monkeypatch):
    accounting = load("test_composition_accounting", "managers/c2kv_kv_accounting.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.c2kv_kv_accounting", accounting)
    req = SimpleNamespace(c2kv_active_region="tool", kv_memory_report={
        "full_equivalent_history_tokens": 100, "active_history_kv_tokens": 10})
    engine._add_c2kv_kv_memory_tokens(req, kind="gist", tokens=4, original_tokens=32)
    assert req.kv_memory_report["active_history_kv_tokens"] == 10
    assert req.kv_memory_report["full_equivalent_history_tokens"] == 100
    assert req.kv_memory_report["active_tool_kv_tokens"] == 4
    assert req.kv_memory_report["active_tool_gist_tokens"] == 4
    assert req.kv_memory_report["tool_encoder_source_tokens"] == 32
