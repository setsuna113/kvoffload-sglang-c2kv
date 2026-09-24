"""CPU checks for the v4 recovery-copy lease at the next decision."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cache_methods():
    path = ROOT / "mem_cache/session_aware_cache.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SessionAwareCache")
    node.bases = []
    node.body = [
        n for n in node.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_resolve_racer_transaction", "_expire_racer_evidence"}
    ]
    namespace = {"torch": torch, "Req": object, "SessionSlot": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["SessionAwareCache"]()


def case(monkeypatch, *, schema="racer-native-protection-v2", enabled=True,
         phase="draft", replace=False, history_end=11, reference=False):
    eviction = load("racer_v2_expiry_eviction", "mem_cache/history_kv_eviction.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.history_kv_eviction", eviction)
    transaction = load("racer_v2_expiry_transaction", "mem_cache/racer_transaction.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.racer_transaction", transaction)
    owner = cache_methods()
    row = torch.tensor([[40, 41, 42, 43, 44, 45, 46, 47, 0, 0, 0, 0]])
    keys = torch.zeros(64, 1, 1)
    keys[40:48, 0, 0] = torch.tensor([0, 1, 4, 5, 6, 8, 9, 10])
    values = keys.clone() + 100
    kv_cache = SimpleNamespace(start_layer=0, layer_num=1,
                               _get_key_buffer=lambda _: keys,
                               _get_value_buffer=lambda _: values)
    owner.req_to_token_pool = SimpleNamespace(req_to_token=row)
    owner.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=4, get_kvcache=lambda: kv_cache,
        free=lambda indices: None, available_size=lambda: 64,
    )
    slot = SimpleNamespace(
        racer_held_generation=SimpleNamespace(decision_id="previous"),
        racer_ephemeral_spans=[(4, 10)], history_kv_reference_state=None,
        history_kv_resident_positions=[0, 1, 4, 5, 6, 8, 9, 10],
        history_kv_score_state={0: {4: 2.0, 6: 3.0, 8: 4.0}},
        req_pool_idx=0, kv_committed_len=8, kv_allocated_len=8,
        c2kv_position_correction=3,
    )
    hint = {
        "persistent_history_session": {
            "transaction": {"decision_id": "next", "phase": phase, "resolution": "commit"},
            "extra_protection": {"schema": schema, "enabled": enabled},
            "recovery_append": {"replace_previous_evidence": replace},
        },
        "racer_native_protection_units": [
            {"unit_id": "u1", "instances": [
                {"kind": "recovery", "spans": [[4, 6]]},
                {"kind": "recovery", "spans": [[8, 10]]},
            ]},
            {"unit_id": "u2", "instances": [
                {"kind": "recovery", "spans": [[6, 8]]},  # Position 7 is absent.
                {"kind": "recovery", "spans": [[8, 10]]},
            ]},
            {"unit_id": "u3", "instances": [
                {"kind": "recovery", "spans": [[6, 8]]},
                {"kind": "original", "spans": [[0, 1]]},
            ]},
        ],
        "racer_internal_source_spans": [[4, 10]],
        "racer_active_ephemeral_source_spans": [],
    }
    if history_end is not None:
        hint["history_kv_eviction"] = {
            "persistent_protected_prefix_tokens": 0,
            "persistent_canonical_history_end": history_end,
        }
    if reference:
        hint["history_kv_reference_config"] = {
            "method": "commitkv", "target_tokens": 8,
        }
    req = SimpleNamespace(
        c2kv_kv_memory_hint=hint, kv_memory_report={},
        origin_input_ids=[0, 1, 4, 5, 6, 8, 9, 10, 100, 101],
    )
    return owner, slot, req, row, keys


def test_next_draft_promotes_one_complete_resident_alias_per_unit(monkeypatch):
    owner, slot, req, row, keys = case(monkeypatch)
    owner._resolve_racer_transaction(slot, req)

    assert slot.history_kv_resident_positions == [0, 1, 4, 5, 8, 9, 10]
    assert keys[row[0, :7], 0, 0].tolist() == [0, 1, 4, 5, 8, 9, 10]
    assert req.origin_input_ids == [0, 1, 4, 5, 8, 9, 10, 100, 101]
    assert slot.kv_committed_len + slot.c2kv_position_correction == 11
    assert slot.history_kv_score_state == {0: {4: 2.0, 8: 4.0}}
    assert slot.racer_ephemeral_spans == []
    assert req.c2kv_kv_memory_hint["racer_promoted_recovery_source_spans"] == [[4, 6], [8, 10]]
    assert req.c2kv_kv_memory_hint["racer_internal_source_spans"] == [[6, 8]]
    assert req.c2kv_kv_memory_hint["racer_active_ephemeral_source_spans"] == []
    expiry = req.kv_memory_report["racer_evidence_expiry"]
    assert expiry["expired_native_tokens"] == 1
    assert expiry["promoted_recovery_tokens"] == 4
    assert expiry["promoted_recovery_unit_ids"] == ["u1", "u2"]
    assert expiry["retained_original_tokens"] == 3
    assert expiry["full_history_reprefill_performed"] is False
    owner._resolve_racer_transaction(slot, req)
    assert slot.history_kv_resident_positions == [0, 1, 4, 5, 8, 9, 10]


def test_promotion_stays_inside_native_history_boundary(monkeypatch):
    owner, slot, req, _, _ = case(monkeypatch, history_end=8)
    owner._resolve_racer_transaction(slot, req)
    assert slot.history_kv_resident_positions == [0, 1, 4, 5, 10]
    assert req.c2kv_kv_memory_hint["racer_promoted_recovery_source_spans"] == [[4, 6]]
    assert req.kv_memory_report["racer_evidence_expiry"]["expired_native_tokens"] == 3


def test_reference_backend_uses_same_bounded_history_pool(monkeypatch):
    owner, slot, req, _, _ = case(monkeypatch, reference=True)
    owner._resolve_racer_transaction(slot, req)
    assert slot.history_kv_resident_positions == [0, 1, 4, 5, 8, 9, 10]
    assert req.c2kv_kv_memory_hint["history_kv_reference_config"]["target_tokens"] == 8


def test_no_bounded_history_pool_expires_all_recovery_copies(monkeypatch):
    owner, slot, req, _, _ = case(monkeypatch, history_end=None)
    owner._resolve_racer_transaction(slot, req)
    assert slot.history_kv_resident_positions == [0, 1, 10]
    assert "racer_promoted_recovery_source_spans" not in req.c2kv_kv_memory_hint


@pytest.mark.parametrize("schema,enabled,phase,replace", [
    ("racer-native-protection-v1", True, "draft", False),
    ("racer-native-protection-v2", False, "draft", False),
    ("racer-native-protection-v2", True, "regenerate", True),
])
def test_historical_off_and_replacement_expire_all_evidence(
    monkeypatch, schema, enabled, phase, replace,
):
    owner, slot, req, _, _ = case(
        monkeypatch, schema=schema, enabled=enabled, phase=phase, replace=replace,
    )
    if phase == "regenerate":
        owner._expire_racer_evidence(slot, req)
    else:
        owner._resolve_racer_transaction(slot, req)
    assert slot.history_kv_resident_positions == [0, 1, 10]
    assert req.c2kv_kv_memory_hint["racer_internal_source_spans"] == [[4, 10]]
    assert "racer_promoted_recovery_source_spans" not in req.c2kv_kv_memory_hint
    assert "promoted_recovery_tokens" not in req.kv_memory_report["racer_evidence_expiry"]
    assert req.kv_memory_report["racer_evidence_expiry"]["expired_native_tokens"] == 5
    assert slot.racer_ephemeral_spans == []
