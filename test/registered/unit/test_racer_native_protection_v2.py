"""CPU contracts for v2 semantic-unit admission and coverage."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python" / "sglang" / "srt"
PATH = SRT / "mem_cache" / "racer_native_protection.py"
SPEC = importlib.util.spec_from_file_location(
    "sglang.srt.mem_cache.racer_native_protection", PATH
)
protection = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protection
SPEC.loader.exec_module(protection)


def _unit(name, event, *aliases, complete=False, status="resolved"):
    return {"unit_id": name, "event_id": event, "complete_event": complete,
            "status": status,
            "instances": [{"kind": kind, "spans": [[start, end]]}
                          for kind, start, end in aliases]}


def _event(name, *aliases):
    return {"event_id": name,
            "instances": [{"kind": kind, "spans": [[start, end]],
                           "complete_event": complete}
                          for kind, start, end, complete in aliases]}


def _rows(*rows):
    return [torch.tensor(row, dtype=torch.long) for row in rows]


def test_missing_and_oversized_units_do_not_block_later_units_or_other_rows():
    units = [
        _unit("missing", "a", ("original", 99, 100)),
        _unit("large", "b", ("original", 10, 13)),
        _unit("small", "c", ("original", 10, 11)),
    ]
    result = protection.protect_native_units(
        [[10, 11, 12, 13], [11, 12, 13, 14]],
        _rows([2, 3], [2, 3]), units, [], phase="draft",
        fixed_rows=[[13], [14]], drop_order=[[2, 3], [2, 3]],
    )
    assert [row.tolist() for row in result["selected"]] == [[0, 3], [2, 3]]
    assert result["changed_rows"] == 1
    assert [unit["admitted_rows"] for unit in result["units"]] == [0, 0, 1]
    assert all(row.numel() == 2 for row in result["selected"])


def test_alternative_alias_uses_lowest_marginal_cost_and_common_suffix():
    result = protection.protect_native_units(
        [[10, 11, 12, 13]], _rows([1, 2]),
        [_unit("u", "e", ("original", 10, 11), ("recovery", 11, 12))],
        [_event("e", ("original", 10, 11, True),
                ("recovery", 11, 12, True))],
        phase="draft", common_positions=[99], drop_order=[[1, 2]],
    )
    assert result["selected"][0].tolist() == [1, 2]
    assert result["units"][0]["row_aliases"][0]["kind"] == "recovery"
    assert result["event_coverage"][0]["status"] == "full"

    common = protection.protect_native_units(
        [[11, 12]], _rows([1]),
        [_unit("u", "e", ("original", 10, 12))],
        [_event("e", ("original", 10, 12, True))],
        phase="draft", common_positions=[10], drop_order=[[1]],
    )
    assert common["units"][0]["status"] == "admitted"
    assert common["event_coverage"][0]["status"] == "full"


def test_per_head_alias_admission_and_event_coverage_never_promotes_fragments():
    units = [_unit("fragment", "e", ("original", 10, 11))]
    events = [_event("e", ("original", 10, 12, True))]
    result = protection.protect_native_units(
        [[10, 11, 12], [11, 12, 13]], _rows([2], [0]),
        units, events, phase="draft", drop_order=[[2], [0]],
    )
    assert result["units"][0]["status"] == "partial"
    assert result["units"][0]["admitted_rows"] == 1
    assert result["event_coverage"][0]["status"] == "partial"
    assert result["event_coverage"][0]["full_rows"] == 0


def test_regeneration_reports_coverage_without_competing_pins():
    result = protection.protect_native_units(
        [[10, 11, 12]], _rows([1, 2]),
        [_unit("u", "e", ("original", 10, 11))],
        [_event("e", ("original", 10, 11, True))],
        phase="regenerate", drop_order=[[1, 2]],
    )
    assert result["selected"][0].tolist() == [1, 2]
    assert result["changed_rows"] == 0
    assert result["units"][0]["admitted_rows"] == 0
    assert result["event_coverage"][0]["status"] == "absent"


def test_streaming_latest_budget_can_swap_oldest_for_complete_unit():
    result = protection.protect_native_units(
        [[10, 11, 12, 13]], _rows([2, 3]),
        [_unit("u", "e", ("original", 10, 11))], [],
        phase="draft", drop_order=[[2, 3]],
    )
    assert result["selected"][0].tolist() == [0, 3]


def test_commitkv_alias_expands_across_pages_and_charges_full_budget():
    pages = [SimpleNamespace(token_indices=range(0, 2)),
             SimpleNamespace(token_indices=range(2, 4))]
    units = protection.expand_commitkv_page_units(
        [_unit("cross", "e", ("original", 1, 3)),
         _unit("later", "e2", ("original", 4, 5)),
         _unit("recovery", "e3", ("recovery", 1, 2))], pages,
    )
    assert set(protection._instance_positions(units[0]["instances"][0])) == set(range(4))
    assert set(protection._instance_positions(units[2]["instances"][0])) == {0, 1}
    result = protection.protect_native_units(
        [list(range(6))], _rows([4, 5]), units[:2], [],
        phase="draft", drop_order=[[4, 5]],
    )
    assert result["units"][0]["admitted_rows"] == 0
    assert result["units"][1]["admitted_rows"] == 1
    assert result["selected"][0].tolist() == [4, 5]

    absent = protection.protect_native_units(
        [[0, 1, 2, 4, 5]], _rows([3, 4]), units[:1], [],
        phase="draft", drop_order=[[3, 4]],
    )
    assert absent["units"][0]["status"] == "absent"


def test_commitkv_recovery_page_budget_is_rebuilt_after_promotion_marker_expires():
    units = [_unit("fragment", "e", ("recovery", 11, 12)),
             _unit("whole", "e", ("recovery", 10, 14))]
    page_spans = protection.commitkv_recovery_page_spans(units, page_size=2)
    assert page_spans == [(10, 12), (12, 14)]
    pages = [SimpleNamespace(token_indices=range(start, end))
             for start, end in page_spans]
    for _ in ("promotion_draft", "next_draft_without_promotion_marker"):
        expanded = protection.expand_commitkv_page_units(units, pages)
        assert set(protection._instance_positions(expanded[0]["instances"][0])) == {10, 11}
        result = protection.protect_native_units(
            [[10, 11, 12, 13]], _rows([1]), expanded, [],
            phase="draft", drop_order=[[1]],
        )
        assert result["units"][0]["admitted_rows"] == 0
        assert result["selected"][0].tolist() == [1]


def test_scheduler_v2_reselection_and_receipt_track_final_rows():
    path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    methods = {item.name: item for cls in tree.body if isinstance(cls, ast.ClassDef)
               and cls.name == "Scheduler" for item in cls.body
               if isinstance(item, ast.FunctionDef) and item.name in {
                   "_protect_native_history_rows", "_protect_native_history_units"}}
    namespace = {}
    exec(compile(ast.Module(body=list(methods.values()), type_ignores=[]),
                 str(path), "exec"), namespace)
    scheduler = SimpleNamespace()
    for name in methods:
        setattr(scheduler, name, MethodType(namespace[name], scheduler))
    receipt = {"schema": protection.SCHEMA_V2, "phase": "draft",
               "status": "no_selection"}
    req = SimpleNamespace(
        kv_memory_report={"racer_native_protection": receipt},
        c2kv_kv_memory_hint={
            "racer_native_protection_units": [_unit("u", "e", ("original", 10, 11))],
            "racer_native_protection_events": [_event("e", ("original", 10, 11, True))],
        },
        history_kv_resident_positions=list(range(10, 15)),
        history_kv_eviction={"history_start": 0, "history_end": 3, "target_tokens": 2},
    )
    first = scheduler._protect_native_history_rows(
        req, [[10, 11, 12]], _rows([1, 2]), drop_order=[[1, 2]])
    assert first[0].tolist() == [0, 2]
    assert receipt["applied"] and receipt["changed_rows"] == 1
    second = scheduler._protect_native_history_rows(
        req, [[11, 12, 13]], _rows([1, 2]), drop_order=[[1, 2]])
    assert second[0].tolist() == [1, 2]
    assert not receipt["applied"] and receipt["changed_rows"] == 0
    assert receipt["units"][0]["status"] == "absent"
    assert receipt["selection_count"] == 2
