"""Held-generation checkpoints for persistent RACER sessions.

The ordinary prompt row is protected throughout decode. Reference selection
replaces immutable KV tensors, while method runtime/query windows are mutable.
Keep the former by reference and copy only the latter. No full-history KV is
constructed by this checkpoint.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

import torch


def transaction_config(hint: dict) -> dict | None:
    value = (hint.get("persistent_history_session") or {}).get("transaction")
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("decision_id"), str) or not value["decision_id"]:
        raise ValueError("RACER_TRANSACTION_DECISION_ID_REQUIRED")
    if value.get("phase") not in {"draft", "regenerate"}:
        raise ValueError("RACER_TRANSACTION_PHASE_INVALID")
    if value.get("resolution") not in {None, "commit", "discard"}:
        raise ValueError("RACER_TRANSACTION_RESOLUTION_INVALID")
    return value


def enforce_request_budget(hint: dict) -> None:
    """Clamp history allocation after the server resolves exact evidence spans."""
    persistent = hint.get("persistent_history_session") or {}
    if not persistent.get("transaction"):
        return
    budget = persistent.get("history_budget_tokens")
    if type(budget) is not int or budget <= 0:
        raise ValueError("RACER_HISTORY_BUDGET_REQUIRED")
    spans = sorted(hint.get("racer_active_ephemeral_source_spans") or [])
    evidence, horizon = 0, -1
    for start, end in spans:
        if start < 0 or end < start:
            raise ValueError("RACER_EVIDENCE_SOURCE_SPAN_INVALID")
        evidence += max(0, end - max(start, horizon))
        horizon = max(horizon, end)
    available = budget - evidence
    if available <= 0:
        raise ValueError("RACER_RECOVERY_CAPACITY_EXHAUSTED")
    requested = {}
    for name in ("history_kv_eviction", "history_kv_reference_config"):
        config = hint.get(name)
        if isinstance(config, dict):
            target = int(config.get("target_tokens") or available)
            requested[name] = target
            if name == "history_kv_reference_config" and str(config.get("method")) == "commitkv":
                config["target_tokens"] = budget
                config["racer_effective_target_tokens"] = min(target, available)
                config["budget_policy_kind"] = "tokens"
                config["budget_policy_value"] = budget
            else:
                config["target_tokens"] = min(target, available)
    hint["racer_budget"] = {
        "history_budget_tokens": budget, "native_evidence_tokens": evidence,
        "available_history_tokens": available, "requested_targets": requested,
        "resolved_targets": {name: hint[name].get("racer_effective_target_tokens", hint[name]["target_tokens"]) for name in requested},
        "source": "server_resolved_source_spans", "cap_increased": False,
    }


def tensor_storage_bytes(value: Any) -> int:
    """Count actual unique tensor storage, including method query buffers."""
    seen, storage = set(), {}

    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            data = item.untyped_storage()
            storage[(str(item.device), data.data_ptr())] = data.nbytes()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "__dict__"):
            visit(vars(item))

    visit(value)
    return sum(storage.values())


@dataclass
class HeldGeneration:
    decision_id: str
    prompt_len: int
    position_correction: int
    resident_positions: list[int]
    reference_state: Any
    runtime_state: Any
    score_state: Any
    input_ids: list[int]
    tensor_bytes: int
    reference_bytes: int
    runtime_bytes: int


def checkpoint_generation(req) -> None:
    hint = getattr(req, "c2kv_kv_memory_hint", None) or {}
    transaction = transaction_config(hint)
    if transaction is None:
        return
    reference = getattr(req, "history_kv_reference_state", None)
    runtime = copy.deepcopy(getattr(req, "history_kv_runtime_state", None))
    ids = list(req.origin_input_ids)
    positions = getattr(req, "history_kv_resident_positions", None)
    if positions is None:
        positions = list(range(len(ids)))
    if len(positions) != len(ids):
        raise RuntimeError("RACER_TRANSACTION_PROMPT_LEDGER_MISMATCH")
    checkpoint = HeldGeneration(
        decision_id=transaction["decision_id"],
        prompt_len=len(ids),
        position_correction=int(getattr(req, "c2kv_position_correction", 0) or 0),
        resident_positions=list(positions),
        reference_state=reference,
        runtime_state=runtime,
        score_state=copy.deepcopy(getattr(req, "history_kv_score_state", None)),
        input_ids=ids,
        tensor_bytes=tensor_storage_bytes((reference, runtime)),
        reference_bytes=tensor_storage_bytes(reference),
        runtime_bytes=tensor_storage_bytes(runtime),
    )
    req.racer_held_generation = checkpoint
    report = getattr(req, "kv_memory_report", None)
    if isinstance(report, dict):
        report["racer_budget"] = dict(hint.get("racer_budget") or {})
        report["racer_transaction"] = {
            **transaction,
            "status": "held",
            "checkpoint_prompt_tokens": checkpoint.prompt_len,
            "checkpoint_reference_bytes": checkpoint.reference_bytes,
            "checkpoint_runtime_bytes": checkpoint.runtime_bytes,
            "checkpoint_tensor_bytes": checkpoint.tensor_bytes,
            "checkpoint_normal_kv_copy_tokens": 0,
            "checkpoint_scope": "resident_prompt_only",
            "temporary_residency_included_in_peak": os.environ.get("C2KV_PAPER_TELEMETRY", "").strip().lower() in {"1", "true", "yes", "on"},
            "full_history_reprefill_performed": False,
        }


def replace_reference_sources(state, spans):
    """Drop source copies without padding or changing attention multiplicity.

    Reference layers have rectangular head layouts. A column touched by any
    replaced source is removed for all heads; collateral removals are explicit.
    This intervention applies only to RACER recovery, never the baseline.
    """
    if state is None:
        return None, {"source_token_slots": 0, "collateral_token_slots": 0, "layers": []}
    result = copy.copy(state)
    result.layers = {}
    receipt = {"source_token_slots": 0, "collateral_token_slots": 0, "layers": []}
    for layer_id, layer in state.layers.items():
        source = torch.zeros_like(layer.positions, dtype=torch.bool)
        for start, end in spans:
            source |= (layer.positions >= start) & (layer.positions < end)
        drop = source.any(dim=0)
        copied = copy.copy(layer)
        copied.key = layer.key[:, ~drop, :].clone()
        copied.value = layer.value[:, ~drop, :].clone()
        copied.positions = layer.positions[:, ~drop].clone()
        result.layers[layer_id] = copied
        collateral = drop.unsqueeze(0).expand_as(source) & ~source
        receipt["source_token_slots"] += int(source.sum().item())
        receipt["collateral_token_slots"] += int(collateral.sum().item())
        receipt["layers"].append({
            "layer_id": int(layer_id),
            "source_positions": [layer.positions[head][source[head]].tolist() for head in range(source.shape[0])],
            "collateral_positions": [layer.positions[head][collateral[head]].tolist() for head in range(source.shape[0])],
        })
    return result, receipt


def interrupt_replaced_commit_window(runtime, removed_positions):
    """A recovery intervention cannot supply a causal post-action window."""
    policy = getattr(runtime, "policy", None)
    pending = getattr(policy, "pending", None)
    if pending is None:
        return None
    removed = set(removed_positions)
    protected = set(pending.protected_page_ids)
    if not any(page.page_id in protected and removed.intersection(page.token_indices) for page in pending.pages):
        return None
    observed = sum(item.shape[0] for item in runtime.post_queries)
    receipt = policy.record_incomplete_post(pending.commit_id, observed_query_count=observed)
    receipt["reason"] = "racer_source_replacement"
    runtime.receipts.append(receipt)
    runtime.pending_commit_id = None
    runtime.post_queries.clear()
    runtime.post_positions.clear()
    return receipt
