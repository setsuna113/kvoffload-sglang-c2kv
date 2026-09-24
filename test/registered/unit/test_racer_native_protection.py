"""CPU checks for optional native selection protection and safe fallback."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from types import MethodType
from typing import Optional

import torch


ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python" / "sglang" / "srt"
MODULE = SRT / "mem_cache" / "racer_native_protection.py"
spec = importlib.util.spec_from_file_location("sglang.srt.mem_cache.racer_native_protection", MODULE)
protection = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protection
spec.loader.exec_module(protection)


def _rows(*values):
    return [torch.tensor(row, dtype=torch.long) for row in values]


def test_budgeted_pin_preserves_native_rank_and_every_row_budget():
    candidates = [[10, 11, 12, 13, 14], [10, 11, 12, 13, 14]]
    native = _rows([1, 3, 4], [0, 2, 4])
    selected, status, changed = protection.protect_native_rows(
        candidates, native, [12],
        fixed_rows=[[14], [14]],
        drop_order=[[3, 1, 4], [0, 2, 4]],
    )
    assert status == "applied" and changed == 1
    assert [row.tolist() for row in selected] == [[1, 2, 4], [0, 2, 4]]
    assert all(row.numel() == 3 for row in selected)


def test_missing_source_in_one_head_falls_back_atomically():
    native = _rows([1, 3], [0, 2])
    selected, status, changed = protection.protect_native_rows(
        [[10, 11, 12, 13], [10, 11, 13, 14]], native, [12]
    )
    assert status == "source_not_resident" and changed == 0
    assert all(a is b for a, b in zip(selected, native))


def test_oversize_and_fixed_pending_fall_back_without_mutation():
    native = _rows([2, 3])
    selected, status, _ = protection.protect_native_rows(
        [[10, 11, 12, 13]], native, [10, 11, 12]
    )
    assert status == "mandatory_exceeds_budget" and selected[0] is native[0]
    selected, status, _ = protection.protect_native_rows(
        [[10, 11, 12, 13]], native, [10], fixed_positions=[12, 13]
    )
    assert status == "mandatory_exceeds_budget" and selected[0] is native[0]


def test_retired_full_and_already_kept_are_noops():
    native = _rows([2, 3])
    for required, forbidden, expected in (
        ([10], [10], "retired_source"),
        ([12], [], "already_retained"),
    ):
        selected, status, changed = protection.protect_native_rows(
            [[10, 11, 12, 13]], native, required,
            forbidden_positions=forbidden,
        )
        assert status == expected and changed == 0 and selected[0] is native[0]
    full = _rows([0, 1, 2])
    selected, status, changed = protection.protect_native_rows(
        [[10, 11, 12]], full, [11]
    )
    assert status == "full_history" and changed == 0 and selected[0] is full[0]


def test_scheduler_wrapper_off_is_exact_identity_and_on_records_receipt():
    scheduler_path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(scheduler_path.read_text(encoding="utf-8"))
    method = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
                  and cls.name == "Scheduler" for item in cls.body
                  if isinstance(item, ast.FunctionDef)
                  and item.name == "_protect_native_history_rows")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(scheduler_path), "exec"), namespace)
    wrapper = namespace[method.name]
    native = _rows([1, 2])
    off = SimpleNamespace(kv_memory_report={}, c2kv_kv_memory_hint={})
    assert wrapper(None, off, [[10, 11, 12]], native) is native
    receipt = {"schema": protection.SCHEMA, "status": "no_selection", "applied": False}
    on = SimpleNamespace(
        kv_memory_report={"racer_native_protection": receipt},
        c2kv_kv_memory_hint={
            "racer_native_protection_source_spans": [[10, 11]],
            "racer_native_protection_span_status": "resolved",
        },
        history_kv_eviction={"target_tokens": 2, "canonical_history_end": 13},
    )
    selected = wrapper(None, on, [[10, 11, 12]], native)
    assert selected[0].tolist() == [0, 2]
    assert receipt["applied"] is True and receipt["status"] == "applied"
    assert receipt["native_row_budgets"] == [2]


def test_reselection_keeps_pin_and_updates_final_receipt():
    scheduler_path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(scheduler_path.read_text(encoding="utf-8"))
    method = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
                  and cls.name == "Scheduler" for item in cls.body
                  if isinstance(item, ast.FunctionDef)
                  and item.name == "_protect_native_history_rows")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(scheduler_path), "exec"), namespace)
    wrapper = namespace[method.name]
    receipt = {"status": "no_selection", "applied": False}
    req = SimpleNamespace(
        kv_memory_report={"racer_native_protection": receipt},
        c2kv_kv_memory_hint={
            "racer_native_protection_source_spans": [[10, 11]],
            "racer_native_protection_span_status": "resolved"},
        history_kv_eviction={"target_tokens": 2, "canonical_history_end": 20},
    )
    first = wrapper(None, req, [[10, 11, 12]], _rows([1, 2]))
    second = wrapper(None, req, [[10, 12, 13]], _rows([1, 2]))
    assert [row.tolist() for row in first] == [[0, 2]]
    assert [row.tolist() for row in second] == [[0, 2]]
    assert receipt["selection_count"] == 2 and receipt["applied_any"] is True
    assert receipt["applied"] is True


def test_reselection_early_noops_clear_final_applied_state():
    scheduler_path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(scheduler_path.read_text(encoding="utf-8"))
    method = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
                  and cls.name == "Scheduler" for item in cls.body
                  if isinstance(item, ast.FunctionDef)
                  and item.name == "_protect_native_history_rows")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(scheduler_path), "exec"), namespace)
    wrapper = namespace[method.name]
    receipt = {"status": "no_selection", "applied": False}
    hint = {"racer_native_protection_source_spans": [[10, 11]],
            "racer_native_protection_span_status": "resolved"}
    config = {"target_tokens": 2, "canonical_history_end": 13}
    req = SimpleNamespace(kv_memory_report={"racer_native_protection": receipt},
                          c2kv_kv_memory_hint=hint, history_kv_eviction=config)
    assert wrapper(None, req, [[10, 11, 12]], _rows([1, 2]))[0].tolist() == [0, 2]
    assert receipt["applied"] is True and receipt["changed_rows"] == 1
    hint["racer_native_protection_span_status"] = "source_span_unavailable"
    assert wrapper(None, req, [[10, 11, 12]], _rows([1, 2]))[0].tolist() == [1, 2]
    assert receipt["status"] == "source_span_unavailable"
    assert receipt["applied"] is False and receipt["changed_rows"] == 0
    hint["racer_native_protection_span_status"] = "resolved"
    config["canonical_history_end"] = 10
    wrapper(None, req, [[10, 11, 12]], _rows([1, 2]))
    assert receipt["status"] == "source_outside_history"
    assert receipt["applied"] is False and receipt["changed_rows"] == 0
    assert receipt["applied_any"] is True and receipt["selection_count"] == 3


def test_source_in_common_prefix_is_already_retained_without_budget_charge():
    scheduler_path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(scheduler_path.read_text(encoding="utf-8"))
    method = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
                  and cls.name == "Scheduler" for item in cls.body
                  if isinstance(item, ast.FunctionDef)
                  and item.name == "_protect_native_history_rows")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(scheduler_path), "exec"), namespace)
    receipt = {"status": "no_selection", "applied": False}
    req = SimpleNamespace(
        kv_memory_report={"racer_native_protection": receipt},
        c2kv_kv_memory_hint={"racer_native_protection_source_spans": [[10, 11]],
                             "racer_native_protection_span_status": "resolved"},
        history_kv_resident_positions=[10, 11, 12],
        history_kv_eviction={"history_start": 1, "target_tokens": 2,
                             "canonical_history_end": 13},
    )
    native = _rows([0, 1])
    selected = namespace[method.name](None, req, [[11, 12]], native)
    assert selected is native and receipt["status"] == "already_retained_common"
    assert receipt["common_protected_source_tokens"] == 1


def test_pyramid_pool_rank_is_not_raw_score_rank():
    path = SRT / "mem_cache" / "history_kv_reference.py"
    module_spec = importlib.util.spec_from_file_location("history_kv_reference_test", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    scores = torch.tensor([[0., 10., 1., 7., 0., 0.]])
    selected, metadata = module.select_pyramidkv_headwise(
        [scores], target_tokens=3, recent_window=1, kernel_size=3,
        include_optional_rank=True,
    )
    optional = metadata["_racer_native_optional_rank"][0][0]
    assert optional == [1, 2]
    assert scores[0, optional[0]] > scores[0, optional[1]]
    assert set(selected[0][0].tolist()) == {1, 2, 5}


def test_agentkv_query_rank_is_not_oldest_first():
    path = SRT / "mem_cache" / "agentkv.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)
                 and item.name in {"_identity_indices", "select_agentkv_layer_indices"}]
    namespace = {"torch": torch, "math": math,
                 "AGENTKV_SINK_TOKENS": 16, "AGENTKV_OBSERVATION_WINDOW": 8,
                 "AGENTKV_ANCHOR_BUDGET": 32}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    key = torch.zeros((30, 1, 1))
    key[16, 0, 0] = 10.
    key[19, 0, 0] = 9.
    ranks = []
    selected = namespace["select_agentkv_layer_indices"](
        key, torch.ones((1, 1, 1)), target_tokens=26,
        optional_rank_output=ranks,
    )
    assert ranks == [[19, 16]]
    assert {16, 19}.issubset(selected[0].tolist())


def test_h2o_rank_uses_cumulative_resident_score_once():
    path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
                  and cls.name == "Scheduler" for item in cls.body
                  if isinstance(item, ast.FunctionDef)
                  and item.name == "_select_history_kv_eviction_indices")
    namespace = {"torch": torch, "Optional": Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    scheduler = SimpleNamespace()
    scheduler._select_history_kv_eviction_indices = MethodType(namespace[method.name], scheduler)
    config = {"method": "h2o", "history_start": 0, "history_end": 6,
              "target_tokens": 3, "history_kv_h2o_recent_fraction": 1 / 3,
              "persistent_session": True}
    req = SimpleNamespace(
        history_kv_eviction=config,
        history_kv_selection_scores={"layers": [torch.tensor([1., 10., 0., 0., 0., 0.])]},
        history_kv_resident_positions=list(range(6)),
        history_kv_score_state={0: {0: 100.}},
        kv_memory_report={"racer_native_protection": {"status": "no_selection"}},
    )
    assert scheduler._select_history_kv_eviction_indices(req, config) == [0, 1, 5]
    assert config["_racer_native_optional_rank"] == [1, 0]
    assert req.history_kv_score_state[0][0] == 101.


def test_protected_union_does_not_hide_the_actual_recent_token():
    path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
                  and cls.name == "Scheduler" for item in cls.body
                  if isinstance(item, ast.FunctionDef)
                  and item.name == "_select_history_kv_eviction_indices")
    namespace = {"torch": torch, "Optional": Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    scheduler = SimpleNamespace()
    scheduler._select_history_kv_eviction_indices = MethodType(namespace[method.name], scheduler)
    config = {"method": "h2o", "history_start": 0, "history_end": 8,
              "target_tokens": 3, "history_kv_h2o_recent_fraction": 1 / 3,
              "protected_history_indices": [7]}
    req = SimpleNamespace(
        history_kv_eviction=config,
        history_kv_selection_scores={"layers": [torch.arange(8., dtype=torch.float32)]},
        history_kv_resident_positions=list(range(8)),
        kv_memory_report={"racer_native_protection": {"status": "no_selection"}},
    )
    selected = scheduler._select_history_kv_eviction_indices(req, config)
    assert selected == [4, 5, 6, 7]
    assert set(selected) - set(config["_racer_native_optional_rank"]) == {6, 7}
    assert config["h2o_recent_kept"] == 1


def test_snapkv_rank_uses_pooling_not_raw_score():
    path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
                  and cls.name == "Scheduler" for item in cls.body
                  if isinstance(item, ast.FunctionDef)
                  and item.name == "_select_history_kv_eviction_indices")
    selection_path = SRT / "mem_cache" / "history_kv_selection.py"
    module_spec = importlib.util.spec_from_file_location("history_kv_selection_test", selection_path)
    selection = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(selection)
    namespace = {"torch": torch, "Optional": Optional,
                 "pool_snapkv_scores_by_position": selection.pool_snapkv_scores_by_position}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    scheduler = SimpleNamespace()
    scheduler._select_history_kv_eviction_indices = MethodType(namespace[method.name], scheduler)
    config = {"method": "snapkv_persistent", "history_start": 0, "history_end": 6,
              "target_tokens": 3, "history_kv_recent_window": 1,
              "history_kv_kernel_size": 3, "history_kv_pooling": "avgpool"}
    req = SimpleNamespace(
        history_kv_eviction=config,
        history_kv_selection_scores={"layers": [torch.tensor([0., 10., 1., 7., 0., 0.])]},
        history_kv_resident_positions=list(range(6)),
        kv_memory_report={"racer_native_protection": {"status": "no_selection"}},
    )
    assert scheduler._select_history_kv_eviction_indices(req, config) == [1, 2, 5]
    assert config["_racer_native_optional_rank"] == [1, 2]


def test_pyramid_builder_pins_source_in_actual_headwise_kv(monkeypatch):
    reference_path = SRT / "mem_cache" / "history_kv_reference.py"
    module_spec = importlib.util.spec_from_file_location(
        "sglang.srt.mem_cache.history_kv_reference", reference_path)
    reference = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_spec.name, reference)
    module_spec.loader.exec_module(reference)
    monkeypatch.setitem(sys.modules, "sglang.srt.observability", SimpleNamespace(
        paper_telemetry=SimpleNamespace(sample=lambda *args, **kwargs: None)))
    scheduler_path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(scheduler_path.read_text(encoding="utf-8"))
    methods = {item.name: item for cls in tree.body if isinstance(cls, ast.ClassDef)
               and cls.name == "Scheduler" for item in cls.body
               if isinstance(item, ast.FunctionDef) and item.name in {
                   "_build_pyramidkv_reference_state", "_protect_native_history_rows"}}
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=list(methods.values()), type_ignores=[]),
                 str(scheduler_path), "exec"), namespace)
    keys = torch.arange(40., dtype=torch.float32).view(20, 2, 1)
    kv_cache = SimpleNamespace(get_kv_buffer=lambda layer: (keys, keys + 100))
    scheduler = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(20, dtype=torch.long).view(1, -1)),
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: kv_cache),
    )
    for name, method in methods.items():
        setattr(scheduler, name, MethodType(namespace[name], scheduler))
    config = {"history_start": 0, "history_end": 8, "target_tokens": 4,
              "history_kv_recent_window": 1, "history_kv_kernel_size": 1,
              "canonical_history_end": 8}
    receipt = {"status": "no_selection", "applied": False}
    req = SimpleNamespace(
        req_pool_idx=0, history_kv_reference_state=None,
        history_kv_resident_positions=list(range(20)),
        history_kv_eviction=config,
        c2kv_kv_memory_hint={
            "racer_native_protection_source_spans": [[0, 1]],
            "racer_native_protection_span_status": "resolved"},
        kv_memory_report={"racer_native_protection": receipt},
    )
    scores = {"layer_ids": [0],
              "headwise_layers": [torch.arange(8., dtype=torch.float32).expand(2, -1)]}
    state = scheduler._build_pyramidkv_reference_state(req, config, scores)
    assert state.layers[0].positions.tolist() == [[0, 6, 7], [0, 6, 7]]
    assert receipt["applied"] is True and receipt["native_row_budgets"] == [3, 3]


def test_tool_carrier_remap_includes_optional_protection():
    path = SRT / "mem_cache" / "c2kv_composition.py"
    module_spec = importlib.util.spec_from_file_location("c2kv_composition_test", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    hint = {"persistent_history_session": {"extra_protection": {
        "source_message_indices": [1, 3], "event_ids": ["a", "b"]}}}
    module.remap_message_metadata(hint, [0], 4)
    assert hint["persistent_history_session"]["extra_protection"] == {
        "source_message_indices": [0, 2], "event_ids": ["a", "b"]}
