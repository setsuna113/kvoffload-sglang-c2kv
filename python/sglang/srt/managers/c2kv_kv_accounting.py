"""Pure helpers for C2KV history-KV accounting.

Active counters describe blocks the scheduler actually installed for the
current request.  Client hints may carry an estimate from older clients, but
that estimate must not seed the runtime counters or the subsequent injection
will count the same block twice.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


ACTIVE_RESIDENT_COUNTERS = (
    "active_history_kv_tokens",
    "active_c2kv_gist_tokens",
    "active_raw_repair_tokens",
    "active_full_raw_tokens",
)


def initialize_c2kv_kv_memory_report(hint: Any) -> Dict[str, Any]:
    """Create a runtime report without treating client estimates as resident."""
    incoming = dict(hint) if isinstance(hint, Mapping) else {}
    report = dict(incoming)
    client_counters: Dict[str, int] = {}
    for key in ACTIVE_RESIDENT_COUNTERS:
        try:
            value = max(0, int(incoming.get(key) or 0))
        except (TypeError, ValueError):
            value = 0
        if value:
            client_counters[key] = value
        report[key] = 0

    for key in ("full_equivalent_history_tokens", "active_recomputed_raw_tokens"):
        try:
            report[key] = max(0, int(report.get(key) or 0))
        except (TypeError, ValueError):
            report[key] = 0

    if report["full_equivalent_history_tokens"] > 0:
        report["full_equivalent_history_tokens_source"] = "request_hint"

    if client_counters:
        report["client_reported_active_counters"] = client_counters
    report["active_history_kv_tokens_source"] = "scheduler_runtime"
    report["source"] = "sglang_c2kv_runtime_injection"
    return report


def add_c2kv_kv_memory_tokens(
    report: Dict[str, Any],
    *,
    kind: str,
    tokens: int,
    original_tokens: int = 0,
) -> None:
    """Accumulate one block after the scheduler has installed it."""
    tokens = max(0, int(tokens or 0))
    original_tokens = max(0, int(original_tokens or 0))
    report["active_history_kv_tokens"] = (
        int(report.get("active_history_kv_tokens") or 0) + tokens
    )
    if original_tokens and not report.get("full_equivalent_history_tokens"):
        report["full_equivalent_history_tokens"] = (
            int(report.get("full_equivalent_history_tokens") or 0)
            + original_tokens
        )
        report["full_equivalent_history_tokens_source"] = (
            "scheduler_first_injected_original_legacy"
        )
    counter = {
        "gist": "active_c2kv_gist_tokens",
        "repair": "active_raw_repair_tokens",
        "recomputed": "active_recomputed_raw_tokens",
        "full": "active_full_raw_tokens",
    }.get(kind)
    if counter is not None:
        report[counter] = int(report.get(counter) or 0) + tokens
    report["active_history_kv_tokens_source"] = "scheduler_runtime"
