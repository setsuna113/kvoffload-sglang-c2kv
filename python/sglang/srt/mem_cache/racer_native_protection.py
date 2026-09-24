"""Optional, budget-preserving protection of resident native history KV."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


SCHEMA = "racer-native-protection-v1"
SCHEMA_V2 = "racer-native-protection-v2"


def protect_native_rows(
    candidate_positions: Sequence[Sequence[int] | Any],
    native_indices: Sequence[Any],
    required_positions: Sequence[int],
    *,
    forbidden_positions: Sequence[int] = (),
    fixed_positions: Sequence[int] = (),
    fixed_rows: Sequence[Sequence[int]] | None = None,
    drop_order: Sequence[Sequence[int]] | None = None,
) -> tuple[list[Any], str, int]:
    """Pin all requested positions in every row without changing row capacity.

    Rows are layer/head selections, or a single shared physical selection. A
    failed row leaves every native row untouched. ``drop_order`` lists native
    candidate indices from least to most preferred; absent ranking uses older
    positions first, matching the latest-first serving convention.
    """
    original = list(native_indices)
    required = set(int(position) for position in required_positions)
    forbidden = set(int(position) for position in forbidden_positions)
    fixed = set(int(position) for position in fixed_positions)
    if not required:
        return original, "empty_request", 0
    if required & forbidden:
        return original, "retired_source", 0
    if len(candidate_positions) != len(original):
        raise ValueError("native protection row count mismatch")
    if drop_order is not None and len(drop_order) != len(original):
        raise ValueError("native protection ranking row count mismatch")
    if fixed_rows is not None and len(fixed_rows) != len(original):
        raise ValueError("native protection fixed row count mismatch")

    prepared = []
    changed = 0
    full_history = True
    for row, (positions, selected) in enumerate(zip(candidate_positions, original)):
        positions = [int(item) for item in positions]
        if len(positions) != len(set(positions)):
            return original, "duplicate_candidate_position", 0
        if selected.ndim != 1:
            raise ValueError("native protection indices must be one-dimensional")
        chosen = [int(item) for item in selected.tolist()]
        if any(index < 0 or index >= len(positions) for index in chosen):
            raise ValueError("native protection index outside candidate row")
        capacity = len(chosen)
        full_history &= capacity == len(positions)
        if not required.issubset(positions):
            return original, "source_not_resident", 0
        if len(required) > capacity:
            return original, "mandatory_exceeds_budget", 0
        mandatory = {index for index, position in enumerate(positions)
                     if position in required}
        missing = mandatory.difference(chosen)
        if not missing:
            prepared.append(selected)
            continue
        order = (list(drop_order[row]) if drop_order is not None
                 else sorted(chosen, key=lambda index: positions[index]))
        row_fixed = fixed | (set(int(position) for position in fixed_rows[row])
                             if fixed_rows is not None else set())
        removable = [index for index in order if index in chosen and index not in mandatory
                     and positions[index] not in row_fixed]
        if len(removable) < len(missing):
            return original, "mandatory_exceeds_budget", 0
        kept = set(chosen).difference(removable[:len(missing)]) | missing
        if len(kept) != capacity:
            raise RuntimeError("native protection changed row capacity")
        prepared.append(selected.new_tensor(sorted(kept)))
        changed += 1
    if full_history:
        return original, "full_history", 0
    return prepared, "applied" if changed else "already_retained", changed


def requested_positions(hint: dict) -> list[int]:
    spans = hint.get("racer_native_protection_source_spans") or []
    return sorted({position for start, end in spans for position in range(start, end)})


def _instance_positions(instance: dict) -> frozenset[int]:
    positions = set()
    for span in instance.get("spans") or []:
        if len(span) != 2:
            return frozenset()
        start, end = span
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            return frozenset()
        positions.update(range(start, end))
    return frozenset(positions)


def expand_commitkv_page_units(units: Sequence[dict], pages: Sequence[Any]) -> list[dict]:
    """Charge every intersected CommitKV page in full for each alias."""
    expanded = []
    page_positions = [frozenset(int(item) for item in page.token_indices)
                      for page in pages]
    for unit in units:
        copied = dict(unit)
        copied_instances = []
        for instance in unit.get("instances") or []:
            copied_instance = dict(instance)
            if instance.get("kind") in {"original", "recovery"}:
                positions = set(_instance_positions(instance))
                for page in page_positions:
                    if positions & page:
                        positions.update(page)
                copied_instance["spans"] = [[position, position + 1]
                                            for position in sorted(positions)]
            copied_instances.append(copied_instance)
        copied["instances"] = copied_instances
        expanded.append(copied)
    return expanded


def commitkv_recovery_page_spans(units: Sequence[dict], page_size: int) -> list[tuple[int, int]]:
    """Rebuild protection-only recovery pages from resolved aliases each turn."""
    if page_size < 1:
        raise ValueError("CommitKV page size must be positive")
    spans = []
    for unit in units:
        if unit.get("status", "resolved") != "resolved":
            continue
        for instance in unit.get("instances") or []:
            if instance.get("kind") != "recovery":
                continue
            for span in instance.get("spans") or []:
                if len(span) != 2:
                    continue
                start, end = span
                if type(start) is int and type(end) is int and 0 <= start < end:
                    spans.append((start, end))
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [(start, min(start + page_size, end))
            for begin, end in merged
            for start in range(begin, end, page_size)]


def protect_native_units(
    candidate_positions: Sequence[Sequence[int] | Any],
    native_indices: Sequence[Any],
    units: Sequence[dict],
    events: Sequence[dict],
    *,
    phase: str,
    common_positions: Sequence[int] = (),
    forbidden_positions: Sequence[int] = (),
    fixed_positions: Sequence[int] = (),
    fixed_rows: Sequence[Sequence[int]] | None = None,
    drop_order: Sequence[Sequence[int]] | None = None,
) -> dict:
    """Admit semantic units independently in each native layer/head row.

    Input unit order is the priority order. Each accepted unit contributes one
    wholly resident alias per row. An unavailable or oversized unit cannot
    block a later unit, and every row keeps exactly its native selected count.
    Regeneration is coverage-only and never changes native selections.
    """
    if phase not in {"draft", "regenerate"}:
        raise ValueError("native protection phase must be draft or regenerate")
    if len(candidate_positions) != len(native_indices):
        raise ValueError("native protection row count mismatch")
    if fixed_rows is not None and len(fixed_rows) != len(native_indices):
        raise ValueError("native protection fixed row count mismatch")
    if drop_order is not None and len(drop_order) != len(native_indices):
        raise ValueError("native protection ranking row count mismatch")

    common = set(map(int, common_positions))
    forbidden = set(map(int, forbidden_positions))
    fixed = set(map(int, fixed_positions))
    normalized = []
    for unit in units:
        instances = [(_instance_positions(item), str(item.get("kind") or ""))
                     for item in unit.get("instances") or []]
        normalized.append((unit, [(positions, kind) for positions, kind in instances
                                  if positions and kind in {"original", "recovery"}]))

    output = []
    resident_rows = []
    admitted_rows = []
    unit_row_status = [[] for _ in units]
    unit_row_alias = [[] for _ in units]
    changed_rows = 0
    for row_id, (positions, selected) in enumerate(zip(candidate_positions, native_indices)):
        positions = [int(item) for item in positions]
        if len(positions) != len(set(positions)):
            raise ValueError("native protection duplicate candidate position")
        chosen = {int(index) for index in selected.tolist()}
        if selected.ndim != 1 or any(index < 0 or index >= len(positions) for index in chosen):
            raise ValueError("native protection selected index outside candidate row")
        capacity = int(selected.numel())
        if len(chosen) != capacity:
            raise ValueError("native protection duplicate selected index")
        by_position = {position: index for index, position in enumerate(positions)}
        pinned = set(fixed)
        if fixed_rows is not None:
            pinned.update(int(position) for position in fixed_rows[row_id])
        accepted_positions = set()
        ranking = (list(drop_order[row_id]) if drop_order is not None
                   else sorted(chosen, key=lambda index: positions[index]))

        for unit_id, (unit, instances) in enumerate(normalized):
            if unit.get("status", "resolved") != "resolved":
                unit_row_status[unit_id].append("source_span_unavailable")
                unit_row_alias[unit_id].append(None)
                continue
            resident = common | {positions[index] for index in chosen}
            available = []
            retired = False
            for instance_index, (alias, kind) in enumerate(instances):
                if alias & forbidden:
                    retired = True
                    continue
                if not alias.issubset(common | set(by_position)):
                    continue
                missing = {by_position[position] for position in alias - resident}
                available.append((len(missing), instance_index, missing, alias, kind))
            available.sort(key=lambda item: (item[0], item[1]))
            accepted = None
            for cost, instance_index, missing, alias, kind in available:
                if phase == "regenerate" and cost:
                    continue
                removable = [index for index in ranking if index in chosen
                             and positions[index] not in pinned
                             and positions[index] not in alias]
                if len(removable) < cost:
                    continue
                accepted = (cost, instance_index, missing, alias, kind, removable)
                break
            if accepted is None:
                status = ("retired_source" if retired and not available else
                          "source_not_resident" if not available else
                          "not_retained" if phase == "regenerate" else
                          "unit_exceeds_budget")
                unit_row_status[unit_id].append(status)
                unit_row_alias[unit_id].append(None)
                continue
            cost, instance_index, missing, alias, kind, removable = accepted
            if cost:
                chosen.difference_update(removable[:cost])
                chosen.update(missing)
            pinned.update(alias)
            if phase == "draft":
                accepted_positions.update(alias)
            unit_row_status[unit_id].append("applied" if cost else "already_retained")
            unit_row_alias[unit_id].append({"instance_index": instance_index,
                                           "kind": kind, "positions": sorted(alias)})

        if len(chosen) != capacity:
            raise RuntimeError("native protection changed row capacity")
        before = {int(index) for index in selected.tolist()}
        if chosen != before:
            output.append(selected.new_tensor(sorted(chosen)))
            changed_rows += 1
        else:
            output.append(selected)
        resident_rows.append(common | {positions[index] for index in chosen})
        admitted_rows.append(accepted_positions)

    total_rows = len(native_indices)
    unit_results = []
    for index, (unit, _) in enumerate(normalized):
        statuses = unit_row_status[index]
        retained = sum(status in {"applied", "already_retained"} for status in statuses)
        admitted = retained if phase == "draft" else 0
        unit_results.append({
            "unit_id": unit.get("unit_id"), "event_id": unit.get("event_id"),
            "complete_event": bool(unit.get("complete_event")),
            "admitted_rows": admitted, "retained_rows": retained,
            "total_rows": total_rows,
            "status": "admitted" if admitted == total_rows and total_rows else
                      "partial" if admitted else "retained" if retained else "absent",
            "coverage_status": "full" if retained == total_rows and total_rows else
                               "partial" if retained else "absent",
            "row_statuses": statuses,
            "row_aliases": unit_row_alias[index],
        })
    event_coverage = _event_coverage(events, normalized, resident_rows)
    admitted_all_rows = set.intersection(*admitted_rows) if admitted_rows else set()
    return {
        "selected": output, "changed_rows": changed_rows,
        "units": unit_results, "event_coverage": event_coverage,
        "admitted_positions_all_rows": sorted(admitted_all_rows),
    }


def _event_coverage(events: Sequence[dict], normalized_units: list,
                    resident_rows: list[set[int]]) -> list[dict]:
    coverage = []
    for event in events:
        event_id = event.get("event_id")
        instances = event.get("instances") or []
        complete = [_instance_positions(item) for item in instances
                    if item.get("complete_event") is True]
        related = set().union(*(_instance_positions(item) for item in instances))
        full_rows = 0
        partial_rows = 0
        for resident in resident_rows:
            if any(alias and alias.issubset(resident) for alias in complete):
                full_rows += 1
            elif related & resident:
                partial_rows += 1
        total_rows = len(resident_rows)
        coverage.append({
            "event_id": event_id, "full_rows": full_rows,
            "partial_rows": partial_rows, "total_rows": total_rows,
            "status": "full" if full_rows == total_rows and total_rows else
                      "partial" if full_rows or partial_rows else "absent",
        })
    return coverage
