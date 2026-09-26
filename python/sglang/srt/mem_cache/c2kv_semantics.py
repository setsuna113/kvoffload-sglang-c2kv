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


def c2kv_tool_gist_identity(source: str) -> str:
    """Stable identity of a tool gist set, computed from its on-disk metadata.

    Both the model process (after loading the tensors) and the HTTP process
    (for the native capability report) derive the same string, so cache keys
    and native chunk handles bind to one checkpoint without an RPC.
    """
    import os

    with open(os.path.join(source, "config.json"), "r", encoding="utf-8") as handle:
        config = json.load(handle)
    metadata = {
        key: value
        for key, value in config.items()
        if key.startswith("history_memory_") or key.startswith("gist_")
    }
    trainer_state = {}
    state_path = os.path.join(source, "trainer_state.json")
    if os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        trainer_state = {
            key: state.get(key) for key in ("global_step", "parameter_version")
        }
    manifest = {}
    manifest_path = os.path.join(source, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            package = json.load(handle)
        manifest = {"files": package.get("files"), "schema": package.get("schema")}
    payload = {
        "schema": "c2kv-tool-gist-identity-v1",
        "source": os.path.basename(os.path.normpath(source)),
        "config": metadata,
        "trainer_state": trainer_state,
        "manifest": manifest,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
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


def validate_c2kv_prefill_graph_512_setup(server_args, model, device) -> None:
    """Reject unsupported C2KV prefill captures before compiling any graph."""

    expected = {
        "enable_c2kv": True,
        "disable_piecewise_cuda_graph": False,
        "piecewise_cuda_graph_tokens": [512],
        "piecewise_cuda_graph_compiler": "eager",
        "chunked_prefill_size": 512,
        "attention_backend": "flashinfer",
        "page_size": 1,
        "c2kv_query_proj": "base",
        "c2kv_gist_type": "dynamic-interleave",
        "c2kv_gist_param": "qkv",
        "c2kv_shadow_feature_layer": -2,
        "enable_return_hidden_states": True,
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "speculative_algorithm": None,
    }
    errors = [
        f"{name}={getattr(server_args, name, None)!r}, expected {value!r}"
        for name, value in expected.items()
        if getattr(server_args, name, None) != value
    ]
    if device != "cuda":
        errors.append(f"device={device!r}, expected 'cuda'")
    if type(model).__name__ != "Qwen3ForCausalLM":
        errors.append(f"model={type(model).__name__}, expected Qwen3ForCausalLM")
    if getattr(model, "full_length_pic", False):
        errors.append("full_length_pic=True, expected False")
    if errors:
        raise ValueError("C2KV_PREFILL_GRAPH_512_UNSUPPORTED: " + "; ".join(errors))


def is_c2kv_prefill_graph_512_eligible(forward_batch) -> bool:
    """Pure-Python gate for the one native prompt-last C2KV prefill shape."""

    if getattr(getattr(forward_batch, "forward_mode", None), "name", None) != "EXTEND":
        return False
    if getattr(forward_batch, "batch_size", None) != 1:
        return False
    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None or len(input_ids) != 512:
        return False
    if getattr(forward_batch, "extend_num_tokens", None) != 512:
        return False
    if getattr(getattr(forward_batch, "capture_hidden_mode", None), "name", None) != "LAST":
        return False
    if not is_c2kv_graph_compatible(forward_batch):
        return False
    if getattr(forward_batch, "input_embeds", None) is not None:
        return False
    if getattr(forward_batch, "spec_info", None) is not None:
        return False
    if getattr(forward_batch, "is_prefill_only", False):
        return False

    # replay_prepare does not copy these request-specific Python state objects.
    if getattr(forward_batch, "c2kv_history_kv_eviction_configs", None) is not None:
        return False
    if getattr(forward_batch, "c2kv_history_kv_selection_scores", None) is not None:
        return False
    for name in (
        "history_kv_reference_states",
        "history_kv_reference_configs",
        "history_kv_runtime_states",
    ):
        values = getattr(forward_batch, name, None)
        if values is not None and (
            not isinstance(values, (list, tuple))
            or any(item is not None for item in values)
        ):
            return False

    # Output-token logprobs use the next-token logits after forward; prompt
    # logprobs need the distinct logits_processor input-logprob path.
    if getattr(forward_batch, "return_logprob", False):
        starts = getattr(forward_batch, "extend_logprob_start_lens_cpu", None)
        lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if not isinstance(starts, (list, tuple)) or not isinstance(lengths, (list, tuple)):
            return False
        if len(starts) != 1 or len(lengths) != 1:
            return False
        if type(starts[0]) is not int or type(lengths[0]) is not int:
            return False
        if starts[0] < lengths[0]:
            return False
        token_ids_logprobs = getattr(forward_batch, "token_ids_logprobs", None)
        if token_ids_logprobs is not None and (
            not isinstance(token_ids_logprobs, (list, tuple))
            or len(token_ids_logprobs) != 1
            or token_ids_logprobs[0] is not None
        ):
            return False
        top_nums = getattr(forward_batch, "top_logprobs_nums", None)
        if top_nums is not None and (
            not isinstance(top_nums, (list, tuple))
            or len(top_nums) != 1
            or type(top_nums[0]) is not int
            or top_nums[0] != 0
        ):
            return False
    return True
