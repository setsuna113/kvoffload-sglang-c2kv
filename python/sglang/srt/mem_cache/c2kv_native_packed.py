"""Pure contracts for event-native packed C2KV generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


NATIVE_PACKED_REQUEST_SCHEMA = "c2kv-native-packed-generation-v1"
NATIVE_PACKED_RESPONSE_SCHEMA = "c2kv-native-packed-generation-response-v1"
NATIVE_PACKED_CAPABILITY_SCHEMA = "c2kv-native-packed-capability-v1"
NATIVE_CHUNK_HANDLE_SCHEMA = "c2kv-native-chunk-handle-v1"
PACKING_VERSION = "history-event-v1"
RAW_LAYOUT_PROFILE = "event-native-evidence-v1"


@dataclass(frozen=True)
class NativePackedPlan:
    logical_input_ids: tuple[int, ...]
    segment_boundaries: tuple[tuple[int, int], ...]
    unique_chunks: tuple[Mapping[str, Any], ...]
    selected_handles: tuple[str, ...]
    compression_handles: tuple[str, ...]
    costs: Mapping[str, int]


def _token_ids(value: Iterable[Any], *, field: str, allow_empty: bool) -> tuple[int, ...]:
    ids = tuple(value)
    if not allow_empty and not ids:
        raise ValueError(f"{field} must contain at least one token")
    if any(type(token_id) is not int or token_id < 0 for token_id in ids):
        raise ValueError(f"{field} must contain nonnegative integer token ids")
    return ids


def canonical_model_binding(
    *,
    model_path: str,
    tokenizer_path: str | None,
    weight_version: str | None,
    dtype: str,
    kv_cache_dtype: str,
    gist_parameter_dtype: str,
    gist_compute_dtype: str,
    gist_type: str,
    gist_param: str,
    gist_extra_embed_num: int,
    gist_residual_type: str,
    gist_overlap: int,
    pic_enabled: bool,
    pic_param: str,
    query_projection: str,
) -> dict[str, Any]:
    """Return the exact serving fields that bind cached gist tensors."""

    return {
        "model_path": str(model_path),
        "tokenizer_path": None if tokenizer_path is None else str(tokenizer_path),
        "weight_version": None if weight_version is None else str(weight_version),
        "dtype": str(dtype),
        "kv_cache_dtype": str(kv_cache_dtype),
        "gist_parameter_dtype": str(gist_parameter_dtype),
        "gist_compute_dtype": str(gist_compute_dtype),
        "gist_type": str(gist_type),
        "gist_param": str(gist_param),
        "gist_extra_embed_num": int(gist_extra_embed_num),
        "gist_residual_type": str(gist_residual_type),
        "gist_overlap": int(gist_overlap),
        "pic_enabled": bool(pic_enabled),
        "pic_param": str(pic_param),
        "query_projection": str(query_projection),
    }


def canonical_chunk_payload(
    chunk: Mapping[str, Any],
    *,
    model_binding: Mapping[str, Any],
    packing_version: str,
    encoding_scope: str,
    compression_ratio: int,
) -> dict[str, Any]:
    token_ids = _token_ids(
        chunk.get("token_ids") or (), field="chunk.token_ids", allow_empty=False
    )
    source_indices = tuple(chunk.get("source_indices") or ())
    if any(type(index) is not int or index < 0 for index in source_indices):
        raise ValueError("chunk.source_indices must contain nonnegative integers")
    required = ("chunk_id", "event_id", "part_index", "source_token_start", "source_token_end")
    missing = [name for name in required if chunk.get(name) is None]
    if missing:
        raise ValueError(f"chunk is missing required fields: {', '.join(missing)}")
    part_index = chunk["part_index"]
    source_token_start = chunk["source_token_start"]
    source_token_end = chunk["source_token_end"]
    if type(part_index) is not int or part_index < 0:
        raise ValueError("chunk.part_index must be a nonnegative integer")
    if (
        type(source_token_start) is not int
        or type(source_token_end) is not int
        or source_token_start < 0
        or source_token_end <= source_token_start
        or source_token_end - source_token_start != len(token_ids)
    ):
        raise ValueError(
            "chunk source token bounds must be nonnegative and match token_ids length"
        )
    return {
        "schema": NATIVE_CHUNK_HANDLE_SCHEMA,
        "model_binding": dict(model_binding),
        "packing_version": str(packing_version),
        "encoding_scope": str(encoding_scope),
        "compression_ratio": int(compression_ratio),
        "chunk": {
            "chunk_id": str(chunk["chunk_id"]),
            "event_id": str(chunk["event_id"]),
            "part_index": part_index,
            "source_indices": list(source_indices),
            "source_token_start": source_token_start,
            "source_token_end": source_token_end,
            "token_ids": list(token_ids),
        },
    }


def canonical_chunk_handle(
    chunk: Mapping[str, Any],
    *,
    model_binding: Mapping[str, Any],
    packing_version: str,
    encoding_scope: str,
    compression_ratio: int,
) -> str:
    payload = canonical_chunk_payload(
        chunk,
        model_binding=model_binding,
        packing_version=packing_version,
        encoding_scope=encoding_scope,
        compression_ratio=compression_ratio,
    )
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_chunk(
    chunk: Mapping[str, Any],
    *,
    model_binding: Mapping[str, Any],
    packing_version: str,
    encoding_scope: str,
    compression_ratio: int,
) -> dict[str, Any]:
    payload = canonical_chunk_payload(
        chunk,
        model_binding=model_binding,
        packing_version=packing_version,
        encoding_scope=encoding_scope,
        compression_ratio=compression_ratio,
    )
    normalized = dict(payload["chunk"])
    handle = canonical_chunk_handle(
        chunk,
        model_binding=model_binding,
        packing_version=packing_version,
        encoding_scope=encoding_scope,
        compression_ratio=compression_ratio,
    )
    supplied_handle = chunk.get("handle")
    if supplied_handle is not None and supplied_handle != handle:
        raise ValueError(
            f"C2KV_NATIVE_HANDLE_MISMATCH: chunk {normalized['chunk_id']!r} "
            f"supplied {supplied_handle!r}, expected {handle!r}"
        )
    normalized["handle"] = handle
    if chunk.get("source_position_start") is not None:
        normalized["source_position_start"] = chunk["source_position_start"]
    if chunk.get("gist_position_ids") is not None:
        normalized["gist_position_ids"] = list(chunk["gist_position_ids"])
    return normalized


def plan_native_packed_request(
    *,
    system_input_ids: Sequence[int],
    workspace_input_ids: Sequence[int],
    encoder_chunks: Sequence[Mapping[str, Any]],
    compression_chunks: Sequence[Mapping[str, Any]],
    model_binding: Mapping[str, Any],
    packing_version: str,
    raw_layout_profile: str,
    encoding_scope: str,
    compression_ratio: int,
) -> NativePackedPlan:
    """Validate the native token frame and build existing C2KV segment spans."""

    if packing_version != PACKING_VERSION:
        raise ValueError(
            f"Unsupported packing_version {packing_version!r}; expected {PACKING_VERSION!r}"
        )
    if raw_layout_profile != RAW_LAYOUT_PROFILE:
        raise ValueError(
            "Unsupported raw_layout_profile "
            f"{raw_layout_profile!r}; expected {RAW_LAYOUT_PROFILE!r}"
        )
    if not encoding_scope:
        raise ValueError("encoding_scope must be a nonempty string")
    if type(compression_ratio) is not int or compression_ratio <= 0:
        raise ValueError("compression_ratio must be a positive integer")

    system_ids = _token_ids(
        system_input_ids, field="system_input_ids", allow_empty=True
    )
    workspace_ids = _token_ids(
        workspace_input_ids, field="workspace_input_ids", allow_empty=False
    )
    selected = [
        _normalize_chunk(
            chunk,
            model_binding=model_binding,
            packing_version=packing_version,
            encoding_scope=encoding_scope,
            compression_ratio=compression_ratio,
        )
        for chunk in encoder_chunks
    ]
    extras = [
        _normalize_chunk(
            chunk,
            model_binding=model_binding,
            packing_version=packing_version,
            encoding_scope=encoding_scope,
            compression_ratio=compression_ratio,
        )
        for chunk in compression_chunks
    ]

    cursor = len(system_ids)
    logical_ids = list(system_ids)
    segment_boundaries: list[tuple[int, int]] = []
    gist_tokens = 0
    for chunk in selected:
        token_ids = tuple(chunk["token_ids"])
        source_position_start = chunk.get("source_position_start")
        if source_position_start != cursor:
            raise ValueError(
                "C2KV_NATIVE_SOURCE_POSITION_MISMATCH: "
                f"chunk {chunk['chunk_id']!r} starts at {source_position_start!r}, "
                f"expected {cursor}"
            )
        expected_positions = [
            cursor + min(start + compression_ratio, len(token_ids)) - 1
            for start in range(0, len(token_ids), compression_ratio)
        ]
        if chunk.get("gist_position_ids") != expected_positions:
            raise ValueError(
                "C2KV_NATIVE_GIST_POSITION_MISMATCH: "
                f"chunk {chunk['chunk_id']!r} has {chunk.get('gist_position_ids')!r}, "
                f"expected {expected_positions!r}"
            )
        logical_ids.extend(token_ids)
        segment_boundaries.append((cursor, cursor + len(token_ids)))
        cursor += len(token_ids)
        gist_tokens += len(expected_positions)
    logical_ids.extend(workspace_ids)

    unique: dict[str, Mapping[str, Any]] = {}
    # Match EventNativeGenerator's always-compress phase and keep selected
    # chunks most-recent in the bounded pool before packed generation.
    for chunk in [*extras, *selected]:
        previous = unique.get(chunk["handle"])
        if previous is not None and previous["token_ids"] != chunk["token_ids"]:
            raise ValueError("A native chunk handle maps to conflicting token ids")
        unique.setdefault(chunk["handle"], chunk)

    presented_encoder_tokens = sum(len(chunk["token_ids"]) for chunk in selected)
    resident_tokens = len(system_ids) + len(workspace_ids) + gist_tokens
    return NativePackedPlan(
        logical_input_ids=tuple(logical_ids),
        segment_boundaries=tuple(segment_boundaries),
        unique_chunks=tuple(unique.values()),
        selected_handles=tuple(chunk["handle"] for chunk in selected),
        compression_handles=tuple(chunk["handle"] for chunk in extras),
        costs={
            "system_tokens": len(system_ids),
            "raw_tokens": len(workspace_ids),
            "presented_encoder_tokens": presented_encoder_tokens,
            "eligible_encoder_tokens": sum(
                len(chunk["token_ids"]) for chunk in unique.values()
            ),
            "gist_tokens": gist_tokens,
            "gist_prefix_kv_tokens": gist_tokens,
            "raw_workspace_kv_tokens": len(workspace_ids),
            "resident_kv_tokens": resident_tokens,
        },
    )


def float16_roundtrip(values: Sequence[Any]) -> list[float]:
    """IEEE-754 binary16 round-trip without importing the accelerator runtime."""

    import struct

    return [struct.unpack("e", struct.pack("e", float(value)))[0] for value in values]


__all__ = [
    "NATIVE_PACKED_CAPABILITY_SCHEMA",
    "NATIVE_PACKED_REQUEST_SCHEMA",
    "NATIVE_PACKED_RESPONSE_SCHEMA",
    "PACKING_VERSION",
    "RAW_LAYOUT_PROFILE",
    "NativePackedPlan",
    "canonical_chunk_handle",
    "canonical_model_binding",
    "float16_roundtrip",
    "plan_native_packed_request",
]
