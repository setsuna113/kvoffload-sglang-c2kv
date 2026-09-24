"""Optional, budget-preserving protection of resident native history KV."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


SCHEMA = "racer-native-protection-v1"


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
