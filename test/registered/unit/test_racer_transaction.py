"""CPU behavioral tests for held drafts, exact recovery, and evidence expiry."""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


transaction = load("sglang.srt.mem_cache.racer_transaction", "mem_cache/racer_transaction.py")


def methods(relative, cls, names):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch, "json": json, "logging": logging, "_is_streaming": lambda req: True, "Req": object, "SessionSlot": object, "Optional": Optional, "List": list, "Dict": dict, "Any": object, "ChatCompletionRequest": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[cls]()


def hint(decision="a", phase="draft", resolution=None):
    tx = {"decision_id": decision, "phase": phase}
    if resolution:
        tx["resolution"] = resolution
    return {"persistent_history_session": {"enabled": True, "transaction": tx}}


def test_discard_restores_resident_state_and_real_statistics_without_reprefill():
    reference = SimpleNamespace(layers={0: SimpleNamespace(key=torch.ones(1, 2, 1), value=torch.ones(1, 2, 1), positions=torch.tensor([[0, 2]]))})
    runtime = SimpleNamespace(policy=SimpleNamespace(commits=["real-tool"]), pre_queries=[torch.tensor([2.0])])
    req = SimpleNamespace(c2kv_kv_memory_hint=hint(), origin_input_ids=[10, 12, 90], c2kv_position_correction=2,
                          history_kv_resident_positions=[0, 2, 4], history_kv_reference_state=reference,
                          history_kv_runtime_state=runtime, history_kv_score_state={0: {2: 3.0}}, kv_memory_report={})
    transaction.checkpoint_generation(req)
    held = req.racer_held_generation
    assert held.reference_state is reference
    runtime.policy.commits.append("discarded-draft")
    runtime.pre_queries[0].add_(100)
    row = torch.tensor([[40, 41, 42, 43, 44, 45, 0, 0]])
    freed = []
    owner = methods("mem_cache/session_aware_cache.py", "SessionAwareCache", {"_resolve_racer_transaction"})
    owner.req_to_token_pool = SimpleNamespace(req_to_token=row)
    owner.page_size = 4
    owner.token_to_kv_pool_allocator = SimpleNamespace(free=lambda indices: freed.extend(indices.tolist()))
    slot = SimpleNamespace(racer_held_generation=held, req_pool_idx=0, kv_committed_len=6, kv_allocated_len=6,
                           history_kv_reference_state=object(), history_kv_runtime_state=runtime)
    incoming = SimpleNamespace(c2kv_kv_memory_hint=hint(phase="regenerate", resolution="discard"), kv_memory_report={})
    owner._resolve_racer_transaction(slot, incoming)
    assert slot.history_kv_reference_state is reference
    assert slot.history_kv_runtime_state.policy.commits == ["real-tool"]
    assert slot.history_kv_runtime_state.pre_queries[0].item() == 2.0
    assert slot.history_kv_resident_positions == [0, 2, 4]
    assert slot.c2kv_position_correction == 2
    assert row.tolist() == [[40, 41, 42, 0, 0, 0, 0, 0]]
    assert freed == [44]  # Shared prompt/tail page 10 remains owned.
    assert incoming.kv_memory_report["racer_previous_resolution"]["discarded_draft_executed"] is False
    owner._resolve_racer_transaction(slot, incoming)
    assert freed == [44]


def test_exact_commitkv_recovery_drops_generated_prefix_only_after_valid_discard():
    owner = methods("entrypoints/openai/serving_chat.py", "OpenAIServingChat", {"_prepare_persistent_history_delta"})
    owner._is_persistent_history_request = lambda _: True
    owner._translate_tool_session_coordinates = lambda *args: None
    owner._persistent_history_sessions = {"s": [10, 11, 90, 91, 100, 101]}
    owner._persistent_history_generation_bases = {"s": [10, 11, 90, 91]}
    owner._persistent_history_generation_prefixes = {"s": [90, 91]}
    owner._persistent_history_exact_output = {"s": True}
    owner._persistent_history_transactions = {"s": {"decision_id": "a", "tool_memory_segments": []}}
    payload = hint(phase="regenerate", resolution="discard")
    payload["persistent_history_session"]["recovery_append"] = {"enabled": True}
    payload["history_kv_eviction"] = {"method": "commitkv", "history_start": 0, "history_end": 2}
    req = SimpleNamespace(stream=False, session_params={"id": "s"}, c2kv_kv_memory_hint=payload)
    delta, _, canonical = owner._prepare_persistent_history_delta(req, [10, 11, 70, 71, 90, 91])
    assert delta == [70, 71, 90, 91]
    assert canonical == [10, 11, 70, 71, 90, 91]
    assert payload["persistent_session_drop_generation_prefix_tokens"] == 2
    assert payload["persistent_session_logical_prefix_tokens"] == 2
    assert req.session_params["drop_previous_output"] is True
    payload["persistent_history_session"]["transaction"]["resolution"] = "commit"
    with pytest.raises(ValueError, match="REGENERATION_MISMATCH"):
        owner._prepare_persistent_history_delta(req, canonical)


@pytest.mark.parametrize("replace", [False, True])
def test_multiround_recovery_reserves_only_resident_notes_and_never_emits_internal_events(replace):
    owner = methods("entrypoints/openai/serving_chat.py", "OpenAIServingChat", {"_prepare_persistent_history_delta"})
    owner._is_persistent_history_request = lambda _: True
    owner._translate_tool_session_coordinates = lambda *args: None
    owner._persistent_history_sessions = {"s": [10, 11, 70, 71, 90, 100, 101]}
    owner._persistent_history_generation_bases = {"s": [10, 11, 70, 71, 90]}
    owner._persistent_history_generation_prefixes = {"s": [90]}
    owner._persistent_history_exact_output = {"s": True}
    owner._persistent_history_transactions = {"s": {"decision_id": "a", "tool_memory_segments": [],
        "internal_source_spans": [[2, 4]], "active_ephemeral_source_spans": [[2, 4]]}}
    payload = hint(phase="regenerate", resolution="discard")
    payload["persistent_history_session"]["recovery_append"] = {"enabled": True, "replace_previous_evidence": replace}
    payload["history_kv_eviction"] = {"method": "commitkv", "history_start": 0, "history_end": 2}
    payload["history_kv_event_token_spans"] = [{"message_index": 0, "start": 0, "end": 2}, {"message_index": 1, "start": 2, "end": 4}]
    req = SimpleNamespace(stream=False, session_params={"id": "s"}, c2kv_kv_memory_hint=payload)
    delta, _, _ = owner._prepare_persistent_history_delta(req, [10, 11, 70, 71, 80, 81, 90])
    assert delta == [80, 81, 90]
    assert payload["racer_active_ephemeral_source_spans"] == ([] if replace else [[2, 4]])
    assert payload["racer_internal_source_spans"] == [[2, 4]]
    assert payload["history_kv_event_token_spans"] == [{"message_index": 0, "start": 0, "end": 2}]


def test_next_decision_expires_native_evidence_without_reviving_history(monkeypatch):
    eviction = load("racer_test_eviction", "mem_cache/history_kv_eviction.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.history_kv_eviction", eviction)
    owner = methods("mem_cache/session_aware_cache.py", "SessionAwareCache", {"_expire_racer_evidence"})
    row = torch.tensor([[40, 41, 42, 43, 44, 45, 0, 0]])
    keys = torch.zeros(64, 1, 1)
    values = torch.zeros_like(keys)
    keys[40:46, 0, 0] = torch.tensor([0, 3, 4, 5, 6, 7])
    values.copy_(keys + 10)
    cache = SimpleNamespace(start_layer=0, layer_num=1, _get_key_buffer=lambda _: keys, _get_value_buffer=lambda _: values)
    owner.req_to_token_pool = SimpleNamespace(req_to_token=row)
    owner.token_to_kv_pool_allocator = SimpleNamespace(page_size=4, get_kvcache=lambda: cache, free=lambda indices: None, available_size=lambda: 64)
    slot = SimpleNamespace(racer_ephemeral_spans=[(4, 6)], history_kv_reference_state=None,
                           history_kv_resident_positions=[0, 3, 4, 5, 6, 7], req_pool_idx=0,
                           kv_committed_len=6, kv_allocated_len=6, c2kv_position_correction=2,
                           history_kv_score_state={0: {3: 1.0, 4: 2.0}})
    incoming = SimpleNamespace(origin_input_ids=[10, 13, 70, 71, 90, 100, 101], kv_memory_report={})
    owner._expire_racer_evidence(slot, incoming)
    assert slot.history_kv_resident_positions == [0, 3, 6, 7]
    assert slot.kv_committed_len + slot.c2kv_position_correction == 8
    assert incoming.origin_input_ids == [10, 13, 90, 100, 101]
    assert keys[row[0, :4], 0, 0].tolist() == [0, 3, 6, 7]
    assert slot.history_kv_score_state == {0: {3: 1.0}}
    assert incoming.kv_memory_report["racer_evidence_expiry"]["expired_native_tokens"] == 2


@pytest.mark.parametrize("replacement", [False, True])
def test_corrected_next_action_restores_prompt_then_expires_evidence_in_same_frame(monkeypatch, replacement):
    eviction = load("racer_combined_eviction", "mem_cache/history_kv_eviction.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.history_kv_eviction", eviction)
    owner = methods("mem_cache/session_aware_cache.py", "SessionAwareCache", {"_resolve_racer_transaction", "_expire_racer_evidence"})
    prompt = SimpleNamespace(c2kv_kv_memory_hint=hint(), origin_input_ids=[10, 13, 70, 71, 90], c2kv_position_correction=2,
                             history_kv_resident_positions=[0, 3, 4, 5, 6], history_kv_runtime_state=None,
                             history_kv_reference_state=None, history_kv_score_state={}, kv_memory_report={})
    transaction.checkpoint_generation(prompt)
    row = torch.tensor([[40, 41, 42, 43, 44, 45, 0, 0]])
    keys = torch.zeros(64, 1, 1)
    keys[40:46, 0, 0] = torch.tensor([0, 3, 4, 5, 6, 9])
    values = keys.clone() + 100
    cache = SimpleNamespace(start_layer=0, layer_num=1, _get_key_buffer=lambda _: keys, _get_value_buffer=lambda _: values)
    freed = []
    owner.page_size = 4
    owner.req_to_token_pool = SimpleNamespace(req_to_token=row)
    owner.token_to_kv_pool_allocator = SimpleNamespace(page_size=4, get_kvcache=lambda: cache,
        free=lambda indices: freed.extend(indices.tolist()), available_size=lambda: 64)
    slot = SimpleNamespace(racer_held_generation=prompt.racer_held_generation,
        racer_ephemeral_spans=[(4, 6)], req_pool_idx=0, kv_committed_len=6, kv_allocated_len=6,
        c2kv_position_correction=4, history_kv_resident_positions=[0, 3, 4, 5, 6, 9],
        history_kv_reference_state=None, history_kv_runtime_state=None)
    # SessionController builds this from prompt IDs, never discarded output.
    incoming = SimpleNamespace(c2kv_kv_memory_hint=hint("b", resolution="discard"),
        origin_input_ids=[10, 13, 70, 71, 90, 200, 201], kv_memory_report={})
    if replacement:
        incoming.c2kv_kv_memory_hint = hint("a", phase="regenerate", resolution="discard")
        incoming.c2kv_kv_memory_hint["persistent_history_session"]["recovery_append"] = {"enabled": True, "replace_previous_evidence": True}
    owner._resolve_racer_transaction(slot, incoming)
    assert incoming.origin_input_ids == [10, 13, 90, 200, 201]
    assert slot.history_kv_resident_positions == [0, 3, 6]
    assert slot.kv_committed_len + slot.c2kv_position_correction == 7
    assert keys[row[0, :3], 0, 0].tolist() == [0, 3, 6]
    assert freed == [44]
    assert not slot.racer_ephemeral_spans
    assert incoming.kv_memory_report["racer_previous_resolution"]["restored_prompt_tokens"] == 5
    assert incoming.kv_memory_report["racer_evidence_expiry"]["trigger"] == ("recovery_replacement" if replacement else "next_decision")
    owner._resolve_racer_transaction(slot, incoming)
    assert freed == [44]


def test_accept_promotes_candidate_statistics_and_releases_checkpoint():
    owner = methods("mem_cache/session_aware_cache.py", "SessionAwareCache", {"_resolve_racer_transaction"})
    owner._expire_racer_evidence = lambda *args: None
    runtime = object()
    slot = SimpleNamespace(racer_held_generation=SimpleNamespace(decision_id="a"), history_kv_runtime_state=runtime)
    incoming = SimpleNamespace(c2kv_kv_memory_hint=hint("b", resolution="commit"), kv_memory_report={})
    owner._resolve_racer_transaction(slot, incoming)
    assert slot.racer_held_generation is None
    assert slot.history_kv_runtime_state is runtime
    assert incoming.kv_memory_report["racer_previous_resolution"]["promoted_algorithm_statistics"]


def test_source_replacement_removes_all_headwise_duplicates_and_reports_collateral():
    layer = SimpleNamespace(key=torch.arange(8).reshape(2, 4, 1), value=torch.arange(8).reshape(2, 4, 1) + 10,
                            positions=torch.tensor([[0, 4, 8, 10], [1, 3, 5, 9]]))
    original = SimpleNamespace(layers={0: layer})
    result, receipt = transaction.replace_reference_sources(original, [(4, 6)])
    assert result.layers[0].positions.tolist() == [[0, 10], [1, 9]]
    assert result.layers[0].key[:, :, 0].tolist() == [[0, 3], [4, 7]]
    assert receipt["source_token_slots"] == 2
    assert receipt["collateral_token_slots"] == 2
    assert receipt["layers"][0]["collateral_positions"] == [[8], [3]]
    assert layer.positions.tolist() == [[0, 4, 8, 10], [1, 3, 5, 9]]


def test_exact_server_spans_reserve_capacity_without_changing_commitkv_total_budget():
    payload = hint()
    payload["persistent_history_session"]["history_budget_tokens"] = 16
    payload["racer_active_ephemeral_source_spans"] = [[20, 24], [23, 27]]
    payload["history_kv_reference_config"] = {"method": "commitkv", "target_tokens": 14}
    payload["history_kv_eviction"] = {"target_tokens": 14}
    transaction.enforce_request_budget(payload)
    assert payload["history_kv_eviction"]["target_tokens"] == 9
    assert payload["history_kv_reference_config"]["target_tokens"] == 16
    assert payload["history_kv_reference_config"]["racer_effective_target_tokens"] == 9
    assert payload["racer_budget"]["native_evidence_tokens"] == 7
    assert payload["racer_budget"]["resolved_targets"]["history_kv_reference_config"] == 9
    payload["racer_active_ephemeral_source_spans"] = [[20, 36]]
    with pytest.raises(ValueError, match="CAPACITY_EXHAUSTED"):
        transaction.enforce_request_budget(payload)


def test_commitkv_recovery_capacity_keeps_total_budget_and_pending_protection():
    core = load("racer_test_commitkv", "mem_cache/commitkv.py")
    policy = core.CommitKVRuntimeState(core.CommitKVConfig(measurement_layer_id=0))
    page = core.EventPage("real-action", 0, 0, 3)
    policy.pending = core.PendingCommit("real-tool", (page,), {}, (page.page_id,), 8)
    selected, receipt = policy.checkpoint(range(7, -1, -1), range(8), target_tokens=8,
                                           capacity_tokens=4, num_layers=1, num_kv_heads=1)
    assert selected[0].tolist() == [[0, 1, 2, 7]]
    assert policy.pending.total_budget == 8
    assert receipt["active_capacity_tokens"] == 4
    with pytest.raises(ValueError, match="pending tokens exceed"):
        policy.checkpoint(range(8), range(8), target_tokens=8, capacity_tokens=2, num_layers=1, num_kv_heads=1)
    runtime = SimpleNamespace(policy=policy, post_queries=[torch.ones(1, 1, 1)], post_positions=[torch.tensor([9])],
                              pending_commit_id="real-tool", receipts=[])
    interrupted = transaction.interrupt_replaced_commit_window(runtime, [1])
    assert interrupted["reason"] == "racer_source_replacement"
    assert interrupted["accepted_page_ids"] == []
    assert policy.pending is None
    assert runtime.pending_commit_id is None
    assert not runtime.post_queries


def test_shadow_features_retain_native_contract_and_configured_layer(monkeypatch):
    packed = load("racer_test_packed", "mem_cache/c2kv_native_packed.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_native_packed", packed)
    feature = load("racer_test_features", "mem_cache/racer_features.py")
    capability = {"num_hidden_layers": 4, "shadow_feature_layer": 3, "model_binding": {"weight_version": "frozen"}}
    assert feature.validate_shadow_request({"enabled": True, "prefill_layer": -1}, capability)
    with pytest.raises(ValueError, match="LAYER_MISMATCH"):
        feature.validate_shadow_request({"prefill_layer": 2}, capability)
    receipt = feature.shadow_feature_receipt({"hidden_states": [[[1.0, 2.0], [3.0, 4.0]]]}, capability, 19)
    assert receipt["schema"] == "event-native-shadow-features-v1"
    assert receipt["prefill"]["hidden"] == [3.0, 4.0]
    assert receipt["prefill"]["position"] == {"kind": "prompt_last", "logical_position": 19}


@pytest.mark.parametrize("method", ["commitkv", "h2o", "snapkv_persistent", "streamingllm"])
def test_first_turn_without_completed_history_has_measured_zero_and_method_identity(monkeypatch, method):
    accounting = load("racer_test_accounting", "managers/c2kv_kv_accounting.py")
    lifecycle = load("racer_test_lifecycle", "mem_cache/history_kv_lifecycle.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.history_kv_lifecycle", lifecycle)
    payload = hint()
    payload.update(history_kv_method=method, persistent_session_logical_prefix_tokens=0,
                   persistent_session_canonical_prompt_tokens=5, persistent_session_delta_tokens=5,
                   active_history_kv_tokens=999)  # Deliberately untrustworthy client estimate.
    owner = methods("mem_cache/session_aware_cache.py", "SessionAwareCache", {"cache_finished_req", "_discard_persistent_decode_suffix", "_is_persistent_history_req"})
    row = torch.tensor([[40, 41, 42, 43, 44, 45, 46, 0]])
    owner.req_to_token_pool = SimpleNamespace(req_to_token=row)
    freed = []
    owner.page_size = 1
    owner.token_to_kv_pool_allocator = SimpleNamespace(free=lambda indices: freed.extend(indices.tolist()))
    saved = []
    owner.slots = {"s": SimpleNamespace(history_kv_resident_positions=[], racer_ephemeral_spans=[],
                                        save_from_req=lambda req, is_first: saved.append(req))}
    req = SimpleNamespace(session=SimpleNamespace(session_id="s"), origin_input_ids=list(range(5)),
        c2kv_kv_memory_hint=payload, history_kv_eviction=None,
        history_kv_reference_config={"method": method} if method == "commitkv" else None,
        history_kv_reference_state=None, history_kv_runtime_state=None,
        history_kv_resident_positions=None, history_kv_score_state={}, req_pool_idx=0,
        kv_committed_len=6, kv_allocated_len=7, c2kv_position_correction=0,
        reference_decode_logical_start=5, reference_decode_protected_len=5,
        output_ids=[100, 101], persistent_decode_cache_locs=[],
        kv_memory_report=accounting.initialize_c2kv_kv_memory_report(payload))
    transaction.checkpoint_generation(req)
    owner.cache_finished_req(req)
    report = req.kv_memory_report
    assert report["active_history_kv_tokens"] == 0
    assert type(report["active_history_kv_tokens"]) is int
    assert report["active_history_kv_tokens_source"] == "scheduler_runtime"
    assert report["client_reported_active_counters"]["active_history_kv_tokens"] == 999
    assert report["history_kv_method"] == report["history_kv_lifecycle"]["history_kv_method"] == method
    assert report["history_kv_lifecycle"]["persistent_session_enabled"] is True
    assert report["history_kv_lifecycle"]["full_history_reprefill_performed"] is False
    assert report["history_kv_lifecycle"]["retained_tokens_this_turn"] == 0
    assert report["racer_transaction"]["checkpoint_prompt_tokens"] == 5
    assert saved == [req]
