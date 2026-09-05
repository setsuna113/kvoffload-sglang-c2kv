"""Pure serving-policy helpers shared by the C2KV request and graph paths."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Optional, Tuple


C2KV_REPAIR_PLACEMENTS = (
    "in_place",
    "append_keep_ledger",
    "append_tail",
)

C2KV_IN_PLACE_REPAIR_MODES = frozenset(
    {
        "d_corr_recompute",
        "d_corr_recompute_w2",
        "d_corr_replace_w1",
        "d_corr_replace_w2",
        "d_corr_replace_w4",
        "d_corr_replace_all",
        "append_masked_w2",
        "raw_all_replace",
        "raw_all_replace_direct",
    }
)

C2KV_IN_PLACE_REPAIR_MODE_PREFIXES = ("history_kv_", "cacheblend")


def validate_gist_param(gist_param: str) -> str:
    """Validate the currently supported lowercase projection-head schema."""

    value = str(gist_param or "")
    if not value or set(value.lower()) - set("qkv"):
        raise ValueError(
            "--c2kv-gist-param must be a non-empty combination of q, k, and v"
        )
    if value != value.lower():
        raise ValueError(
            "C2KV_GIST_PARAM_CASE_UNSUPPORTED: mixed-case --c2kv-gist-param "
            f"{value!r} encodes partial query projection in the reference "
            "implementation, which this server does not silently approximate; "
            "use lowercase 'qkv' for the reference/base-query checkpoint"
        )
    return value


def compute_gist_cache_key(
    token_ids: Iterable[int],
    compression_ratio: int,
    extractor_config: Optional[dict] = None,
) -> str:
    """Hash every request-varying input that changes extracted gist KV."""

    payload = {
        "schema": "c2kv-gist-v2",
        "token_ids": [int(token_id) for token_id in token_ids],
        "compression_ratio": int(compression_ratio),
        "extractor_config": dict(extractor_config or {}),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_rope_position_range(
    min_position: int, max_position: int, table_size: int
) -> None:
    """Reject position aliasing instead of clamping to a RoPE table edge."""

    if table_size <= 0:
        raise ValueError("C2KV_ROPE_POSITION_OUT_OF_RANGE: RoPE table is empty")
    if min_position < 0 or max_position >= table_size:
        raise ValueError(
            "C2KV_ROPE_POSITION_OUT_OF_RANGE: requested absolute positions "
            f"[{min_position}, {max_position}] outside RoPE table [0, {table_size})"
        )


def resolve_query_projection(
    server_default: str,
    request_value: Optional[bool],
    segment_values: Iterable[Optional[bool]],
) -> Tuple[bool, str]:
    """Resolve one request-wide projection mode and its provenance.

    The model consumes one request-level mode. An explicit request value wins;
    otherwise all explicit message values must agree. Unset messages do not
    dilute an explicit message choice.
    """

    if server_default not in ("base", "gist"):
        raise ValueError(
            "C2KV_QUERY_PROJECTION_INVALID: server default must be 'base' or "
            f"'gist', got {server_default!r}"
        )
    if request_value is not None:
        return bool(request_value), "request"

    explicit = {bool(value) for value in segment_values if value is not None}
    if len(explicit) > 1:
        raise ValueError(
            "C2KV_QUERY_PROJECTION_CONFLICT: annotated messages disagree on "
            "c2kv_use_gist_projection; set one request-level value or make "
            "all explicit message values agree"
        )
    if explicit:
        return explicit.pop(), "message"
    return server_default == "gist", "flag"


def resolve_repair_placement(repair_mode: str, placement: Optional[str]) -> str:
    """Resolve the explicit placement or the legacy repair-mode default."""

    if placement is None or placement == "":
        repair_mode = str(repair_mode or "")
        legacy_in_place = (
            repair_mode in C2KV_IN_PLACE_REPAIR_MODES
            or repair_mode.startswith(C2KV_IN_PLACE_REPAIR_MODE_PREFIXES)
        )
        return "in_place" if legacy_in_place else "append_keep_ledger"
    if placement not in C2KV_REPAIR_PLACEMENTS:
        raise ValueError(
            f"Unknown c2kv_repair_placement {placement!r}; expected one of "
            f"{C2KV_REPAIR_PLACEMENTS}"
        )
    return placement


def is_c2kv_graph_compatible(forward_batch) -> bool:
    """Return whether graph replay can preserve this batch's projection mode.

    Full, CPU, and piecewise graph captures do not own a dynamic C2KV
    projection-mask buffer. A non-None mask therefore has to run eagerly.
    """

    return getattr(forward_batch, "c2kv_use_gist_projection", None) is None
