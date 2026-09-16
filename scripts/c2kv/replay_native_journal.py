"""Replay an event-native generation journal with auditable comparisons.

This tool validates the complete request/response journal before sending any
requests.  It replays only engine generation and does not claim BFCL task or
scorer coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import types
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


SCHEMA = "c2kv-engine-generation-replay-v2"
SCOPE = "engine_generation_replay"
ENDPOINT = "/v1/c2kv/native_generate"


class ReplayHttpError(RuntimeError):
    """HTTP failure that retains the response body for the audit artifact."""

    def __init__(self, message: str, *, actual: Any = None, detail: Any = None):
        super().__init__(message)
        self.actual = actual
        self.detail = detail


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _error(code: str, message: str, **fields: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **fields}


def _rid(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _default_out(journal: Path) -> Path:
    return journal.with_name(f"{journal.stem}.engine_generation_replay.json")


def load_journal(journal_path: Path) -> dict[str, Any]:
    """Load and strictly pair journal records without dropping invalid rows."""

    source = journal_path.read_bytes()
    errors: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    seen_requests: set[str] = set()
    seen_responses: set[str] = set()
    pairs: list[dict[str, Any]] = []
    unpaired_records: list[dict[str, Any]] = []
    request_count = 0
    response_count = 0

    for line_number, raw_bytes in enumerate(source.splitlines(), start=1):
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(
                _error(
                    "journal_utf8_invalid",
                    str(exc),
                    line=line_number,
                    raw_hex=raw_bytes.hex(),
                )
            )
            continue
        if not raw.strip():
            errors.append(
                _error("journal_empty_line", "empty journal line", line=line_number, raw=raw)
            )
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(
                _error("journal_json_invalid", str(exc), line=line_number, raw=raw)
            )
            continue
        if not isinstance(row, Mapping):
            errors.append(
                _error(
                    "journal_row_not_object",
                    "journal row must be a JSON object",
                    line=line_number,
                    row=row,
                )
            )
            continue

        event = row.get("event")
        if event == "request":
            request_count += 1
            request = row.get("request")
            rid = _rid(request.get("rid")) if isinstance(request, Mapping) else None
            if rid is None:
                errors.append(
                    _error(
                        "request_missing_rid",
                        "request row lacks a nonempty request.rid",
                        line=line_number,
                        row=row,
                    )
                )
                unpaired_records.append({"line": line_number, "row": row})
                continue
            if rid in seen_requests:
                errors.append(
                    _error(
                        "duplicate_request",
                        "request rid occurs more than once",
                        line=line_number,
                        rid=rid,
                        row=row,
                    )
                )
                unpaired_records.append({"line": line_number, "row": row})
                continue
            seen_requests.add(rid)
            pending[rid] = {
                "rid": rid,
                "request": dict(request),
                "request_row": dict(row),
                "request_line": line_number,
            }
            continue

        if event == "response":
            response_count += 1
            expected = row.get("response")
            rid = _rid(expected.get("rid")) if isinstance(expected, Mapping) else None
            if rid is None:
                errors.append(
                    _error(
                        "response_missing_rid",
                        "response row lacks a nonempty response.rid",
                        line=line_number,
                        row=row,
                    )
                )
                unpaired_records.append({"line": line_number, "row": row})
                continue
            if rid in seen_responses:
                errors.append(
                    _error(
                        "duplicate_response",
                        "response rid occurs more than once",
                        line=line_number,
                        rid=rid,
                        row=row,
                    )
                )
                unpaired_records.append({"line": line_number, "row": row})
                continue
            seen_responses.add(rid)
            request_record = pending.pop(rid, None)
            if request_record is None:
                errors.append(
                    _error(
                        "orphan_response",
                        "response has no preceding unmatched request",
                        line=line_number,
                        rid=rid,
                        row=row,
                    )
                )
                unpaired_records.append({"line": line_number, "row": row})
                continue
            request_record.update(
                expected=dict(expected),
                response_row=dict(row),
                response_line=line_number,
            )
            pairs.append(request_record)
            continue

        errors.append(
            _error(
                "journal_event_invalid",
                "journal event must be request or response",
                line=line_number,
                row=row,
            )
        )
        unpaired_records.append({"line": line_number, "row": row})

    for rid, request_record in pending.items():
        errors.append(
            _error(
                "missing_response",
                "request has no response",
                line=request_record["request_line"],
                rid=rid,
                row=request_record["request_row"],
            )
        )
        unpaired_records.append(
            {"line": request_record["request_line"], "row": request_record["request_row"]}
        )

    pairs.sort(key=lambda item: item["request_line"])
    if not pairs:
        errors.append(_error("journal_has_no_pairs", "journal has no complete request/response pairs"))
    return {
        "source_bytes": source,
        "source_sha256": _sha256_bytes(source),
        "pairs": pairs,
        "errors": errors,
        "unpaired_records": unpaired_records,
        "request_count": request_count,
        "response_count": response_count,
    }


def post(base_url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        decoded = body.decode("utf-8", errors="replace")
        try:
            actual = json.loads(decoded)
        except json.JSONDecodeError:
            actual = decoded
        raise ReplayHttpError(
            f"HTTP {exc.code}: {exc.reason}",
            actual=actual,
            detail={"http_status": exc.code, "response_body": decoded},
        ) from exc
    decoded = body.decode("utf-8")
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise ReplayHttpError(
            "HTTP response is not valid JSON",
            actual=decoded,
            detail={"response_body": decoded},
        ) from exc
    if not isinstance(value, dict):
        raise ReplayHttpError("HTTP response JSON is not an object", actual=value)
    return value


def _required_response_fields(value: Any, label: str) -> tuple[list[int], str, str, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(value, Mapping):
        return [], "", "", [_error(f"{label}_not_object", f"{label} response is not an object")]

    ids = value.get("output_ids")
    if not isinstance(ids, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        errors.append(
            _error(f"{label}_output_ids_invalid", f"{label}.output_ids must be a list of integers")
        )
        ids = ids if isinstance(ids, list) else []
    text = value.get("text")
    if not isinstance(text, str):
        errors.append(_error(f"{label}_text_missing", f"{label}.text must be a string"))
        text = ""
    finish = value.get("finish_reason")
    if not isinstance(finish, str) or not finish:
        errors.append(
            _error(f"{label}_finish_reason_missing", f"{label}.finish_reason must be nonempty")
        )
        finish = ""
    return list(ids), text, finish, errors


def _identity_errors(
    request: Mapping[str, Any], expected: Mapping[str, Any], actual: Any
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    actual_mapping = actual if isinstance(actual, Mapping) else {}
    for field in ("rid", "session_id", "generation_id"):
        if field not in request:
            continue
        request_value = request[field]
        for label, response in (("expected", expected), ("actual", actual_mapping)):
            if field not in response:
                errors.append(
                    _error(
                        f"{label}_{field}_missing",
                        f"{label}.{field} is required because request.{field} is present",
                        request=request_value,
                    )
                )
            elif response[field] != request_value:
                errors.append(
                    _error(
                        f"{label}_{field}_mismatch",
                        f"{label}.{field} differs from request.{field}",
                        request=request_value,
                        response=response[field],
                    )
                )
    return errors


def _shadow_requested(request: Mapping[str, Any]) -> bool:
    value = request.get("shadow_features")
    return value is not None and not (
        isinstance(value, Mapping) and value.get("enabled") is False
    )


def _hidden_vector(features: Any, label: str) -> tuple[list[float] | None, dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(features, Mapping):
        return None, {}, [_error(f"{label}_shadow_missing", f"{label}.shadow_features is missing")]
    prefill = features.get("prefill")
    if not isinstance(prefill, Mapping):
        return None, {}, [_error(f"{label}_prefill_missing", f"{label} prefill feature is missing")]

    contract = {
        "schema": features.get("schema"),
        "layer": prefill.get("layer"),
        "position": prefill.get("position"),
        "readout": prefill.get("readout"),
        "status": prefill.get("status"),
    }
    if isinstance(contract["layer"], bool) or not isinstance(contract["layer"], int):
        errors.append(_error(f"{label}_prefill_layer_invalid", "prefill.layer must be an integer"))
    if (
        not isinstance(contract["position"], Mapping)
        or contract["position"].get("kind") != "prompt_last"
    ):
        errors.append(
            _error(
                f"{label}_prefill_position_invalid",
                "prefill.position.kind must be prompt_last",
            )
        )
    if contract["readout"] != "decoder_layer_output":
        errors.append(
            _error(
                f"{label}_prefill_readout_invalid",
                "prefill.readout must be decoder_layer_output",
            )
        )
    if contract["status"] != "captured":
        errors.append(_error(f"{label}_prefill_status_invalid", "prefill.status must be captured"))

    hidden = prefill.get("hidden")
    if not isinstance(hidden, list) or not hidden:
        errors.append(_error(f"{label}_prefill_hidden_missing", "prefill.hidden must be nonempty"))
        return None, contract, errors
    values: list[float] = []
    for index, item in enumerate(hidden):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            errors.append(
                _error(
                    f"{label}_prefill_hidden_invalid",
                    "prefill.hidden contains a nonnumeric value",
                    index=index,
                )
            )
            return None, contract, errors
        item = float(item)
        if not math.isfinite(item):
            errors.append(
                _error(
                    f"{label}_prefill_hidden_invalid",
                    "prefill.hidden contains a nonfinite value",
                    index=index,
                )
            )
            return None, contract, errors
        values.append(item)
    return values, contract, errors


def _cosine(left: list[float], right: list[float]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(item * item for item in left))
    right_norm = math.sqrt(math.fsum(item * item for item in right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def compare_shadow_features(
    request: Mapping[str, Any], expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    requested = request.get("shadow_features")
    if not _shadow_requested(request):
        return {"status": "not_requested", "passed": True, "errors": []}
    errors: list[dict[str, Any]] = []
    if not isinstance(requested, Mapping):
        errors.append(_error("shadow_request_invalid", "request.shadow_features must be an object"))
    elif isinstance(requested.get("prefill_layer"), bool) or not isinstance(
        requested.get("prefill_layer"), int
    ):
        errors.append(
            _error("shadow_request_layer_invalid", "shadow_features.prefill_layer must be an integer")
        )

    expected_hidden, expected_contract, expected_errors = _hidden_vector(
        expected.get("shadow_features"), "expected"
    )
    actual_hidden, actual_contract, actual_errors = _hidden_vector(
        actual.get("shadow_features"), "actual"
    )
    errors.extend(expected_errors)
    errors.extend(actual_errors)

    for field in ("schema", "layer", "position", "readout", "status"):
        if expected_contract.get(field) != actual_contract.get(field):
            errors.append(
                _error(
                    f"shadow_{field}_mismatch",
                    f"expected and actual shadow {field} differ",
                    expected=expected_contract.get(field),
                    actual=actual_contract.get(field),
                )
            )

    dimension = None
    max_abs = None
    cosine = None
    if expected_hidden is not None and actual_hidden is not None:
        dimension = len(expected_hidden)
        if len(expected_hidden) != len(actual_hidden):
            errors.append(
                _error(
                    "shadow_hidden_dimension_mismatch",
                    "expected and actual hidden dimensions differ",
                    expected=len(expected_hidden),
                    actual=len(actual_hidden),
                )
            )
        else:
            max_abs = max(abs(a - b) for a, b in zip(expected_hidden, actual_hidden, strict=True))
            cosine = _cosine(expected_hidden, actual_hidden)

    return {
        "status": "checked",
        "passed": not errors,
        "requested": dict(requested) if isinstance(requested, Mapping) else requested,
        "expected_contract": expected_contract,
        "actual_contract": actual_contract,
        "hidden_dimension": dimension,
        "hidden_max_abs_difference": max_abs,
        "hidden_cosine": cosine,
        "errors": errors,
    }


def _load_detector(
    controller_config: Path | None, runtime_root: Path | None
) -> tuple[dict[str, Any], Callable[[Any, Mapping[str, Any]], tuple[Any, Any]] | None, Mapping[str, Any] | None]:
    if controller_config is None:
        if runtime_root is not None:
            raise ValueError("--runtime-root requires --controller-config")
        return {"status": "detector_not_checked"}, None, None
    if runtime_root is None:
        raise ValueError("--controller-config requires --runtime-root")

    controller_bytes = controller_config.read_bytes()
    controller = json.loads(controller_bytes)
    if not isinstance(controller, Mapping):
        raise ValueError("controller config must be a JSON object")
    recovery = runtime_root / "benchmarks" / "memory_runtime" / "recovery"
    gate_path = recovery / "gate.py"
    if not gate_path.is_file():
        raise ValueError(f"runtime gate module does not exist: {gate_path}")

    package_name = f"_c2kv_replay_recovery_{_sha256_bytes(str(recovery).encode())[:12]}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(recovery)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.gate", gate_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load runtime gate module: {gate_path}")
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)
    reader = getattr(gate, "_read_prefill_score", None)
    if not callable(reader):
        raise ValueError("runtime gate lacks callable _read_prefill_score")

    try:
        head = controller["post_draft_recovery"]["prefill_head"]
    except (KeyError, TypeError) as exc:
        raise ValueError("controller config lacks post_draft_recovery.prefill_head") from exc
    if not isinstance(head, Mapping):
        raise ValueError("controller prefill_head must be an object")
    artifact_sha256 = head.get("artifact_sha256")
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_sha256)
    ):
        raise ValueError("controller prefill_head lacks a SHA-256 artifact binding")
    threshold = head.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("controller prefill_head threshold must be finite")
    metadata = {
        "status": "checked",
        "controller_config": str(controller_config.resolve()),
        "controller_config_sha256": _sha256_bytes(controller_bytes),
        "runtime_root": str(runtime_root.resolve()),
        "gate_source": str(gate_path.resolve()),
        "gate_source_sha256": _sha256_bytes(gate_path.read_bytes()),
        "head_artifact_sha256": artifact_sha256,
        "head_contract_sha256": _canonical_sha256(head),
        "threshold": float(threshold),
    }
    return metadata, reader, head


def compare_detector(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    detector_metadata: Mapping[str, Any],
    reader: Callable[[Any, Mapping[str, Any]], tuple[Any, Any]] | None,
    head: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if reader is None or head is None:
        if detector_metadata.get("status") == "detector_not_checked":
            return {"status": "detector_not_checked", "passed": True, "errors": []}
        error = _error("detector_setup_failed", "detector setup did not complete")
        return {
            "status": "detector_setup_failed",
            "passed": False,
            "errors": [error],
        }
    errors: list[dict[str, Any]] = []
    try:
        expected_score, expected_reason = reader(expected.get("shadow_features"), head)
    except Exception as exc:  # noqa: BLE001 - convert gate contract failure to evidence
        expected_score, expected_reason = None, f"reader_error:{exc!r}"
    try:
        actual_score, actual_reason = reader(actual.get("shadow_features"), head)
    except Exception as exc:  # noqa: BLE001 - convert gate contract failure to evidence
        actual_score, actual_reason = None, f"reader_error:{exc!r}"
    if expected_reason is not None:
        errors.append(
            _error("expected_detector_unavailable", "expected detector score unavailable", reason=expected_reason)
        )
    if actual_reason is not None:
        errors.append(
            _error("actual_detector_unavailable", "actual detector score unavailable", reason=actual_reason)
        )
    threshold = float(head["threshold"])
    expected_gate = expected_score >= threshold if expected_reason is None else None
    actual_gate = actual_score >= threshold if actual_reason is None else None
    gate_agrees = (
        expected_gate == actual_gate if expected_gate is not None and actual_gate is not None else None
    )
    if gate_agrees is False:
        errors.append(
            _error(
                "detector_gate_mismatch",
                "expected and actual detector decisions differ",
                expected=expected_gate,
                actual=actual_gate,
            )
        )
    score_difference = (
        abs(float(expected_score) - float(actual_score))
        if expected_score is not None and actual_score is not None
        else None
    )
    return {
        "status": "checked",
        "passed": not errors,
        "head_artifact_sha256": detector_metadata["head_artifact_sha256"],
        "threshold": threshold,
        "expected_score": expected_score,
        "actual_score": actual_score,
        "score_absolute_difference": score_difference,
        "expected_gate": expected_gate,
        "actual_gate": actual_gate,
        "gate_agrees": gate_agrees,
        "errors": errors,
    }


def replay(
    journal_path: Path,
    base_url: str,
    *,
    controller_config: Path | None = None,
    runtime_root: Path | None = None,
    post_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if post_fn is None:
        post_fn = post
    loaded = load_journal(journal_path)
    setup_errors: list[dict[str, Any]] = []
    try:
        detector_metadata, detector_reader, detector_head = _load_detector(
            controller_config, runtime_root
        )
    except Exception as exc:  # noqa: BLE001 - serialize setup failure into the artifact
        detector_metadata = {"status": "detector_setup_failed"}
        detector_reader = None
        detector_head = None
        setup_errors.append(_error("detector_setup_failed", repr(exc)))

    structural_failure = bool(loaded["errors"] or setup_errors)
    expected_total = 0
    expected_unavailable = 0
    for pair in loaded["pairs"]:
        ids = pair["expected"].get("output_ids")
        if isinstance(ids, list):
            expected_total += len(ids)
        else:
            expected_unavailable += 1

    results: list[dict[str, Any]] = []
    identical_ids = 0
    positional_matches = 0
    attempted = 0
    replayed = 0
    for index, pair in enumerate(loaded["pairs"], start=1):
        request = pair["request"]
        expected = pair["expected"]
        rid = pair["rid"]
        shadow_requested = _shadow_requested(request)
        actual: Any = None
        errors: list[dict[str, Any]] = []
        http_error_detail: Any = None

        expected_ids, expected_text, expected_finish, expected_errors = _required_response_fields(
            expected, "expected"
        )
        errors.extend(expected_errors)

        if structural_failure:
            errors.append(
                _error(
                    "replay_skipped_invalid_input",
                    "replay skipped because journal or detector setup validation failed",
                )
            )
        else:
            attempted += 1
            try:
                actual = post_fn(base_url, request)
                replayed += 1
            except ReplayHttpError as exc:
                actual = exc.actual
                http_error_detail = exc.detail
                errors.append(_error("http_error", str(exc), detail=exc.detail))
            except Exception as exc:  # noqa: BLE001 - report every request and continue
                errors.append(_error("http_error", repr(exc)))

        actual_ids: list[int] = []
        actual_text = ""
        actual_finish = ""
        if actual is not None:
            actual_ids, actual_text, actual_finish, actual_errors = _required_response_fields(
                actual, "actual"
            )
            errors.extend(actual_errors)
        errors.extend(_identity_errors(request, expected, actual))

        ids_equal = actual is not None and expected_ids == actual_ids
        text_equal = actual is not None and expected_text == actual_text
        finish_equal = actual is not None and expected_finish == actual_finish
        positional_match = sum(
            1 for expected_id, actual_id in zip(expected_ids, actual_ids) if expected_id == actual_id
        )
        positional_matches += positional_match
        if not ids_equal:
            errors.append(_error("output_ids_mismatch", "expected and actual output_ids differ"))
        if not text_equal:
            errors.append(_error("text_mismatch", "expected and actual text differ"))
        if not finish_equal:
            errors.append(_error("finish_reason_mismatch", "expected and actual finish_reason differ"))

        if actual is not None and isinstance(actual, Mapping):
            shadow = compare_shadow_features(request, expected, actual)
            detector = (
                compare_detector(
                    expected,
                    actual,
                    detector_metadata,
                    detector_reader,
                    detector_head,
                )
                if shadow_requested
                else {"status": "not_requested", "passed": True, "errors": []}
            )
        else:
            shadow = {
                "status": "unavailable",
                "passed": not shadow_requested,
                "errors": (
                    []
                    if not shadow_requested
                    else [_error("actual_shadow_missing", "actual response is unavailable")]
                ),
            }
            detector = (
                {"status": "not_requested", "passed": True, "errors": []}
                if not shadow_requested
                else {"status": "detector_not_checked", "passed": True, "errors": []}
                if detector_reader is None
                and detector_metadata.get("status") == "detector_not_checked"
                else {
                    "status": "unavailable",
                    "passed": False,
                    "head_artifact_sha256": detector_metadata.get("head_artifact_sha256"),
                    "errors": [_error("actual_detector_missing", "actual response is unavailable")],
                }
            )
        errors.extend(shadow["errors"])
        errors.extend(detector["errors"])
        passed = not errors
        if ids_equal:
            identical_ids += 1
        results.append(
            {
                "index": index,
                "rid": rid,
                "request_line": pair["request_line"],
                "response_line": pair["response_line"],
                "request": request,
                "expected": expected,
                "actual": actual,
                "journal_request_row": pair["request_row"],
                "journal_response_row": pair["response_row"],
                "http_error_detail": http_error_detail,
                "expected_len": len(expected_ids),
                "actual_len": len(actual_ids),
                "ids_equal": ids_equal,
                "text_equal": text_equal,
                "finish_reason_equal": finish_equal,
                "positional_match": positional_match,
                "shadow_features": shadow,
                "detector": detector,
                "passed": passed,
                "errors": errors,
            }
        )

    passed = bool(results) and not loaded["errors"] and not setup_errors and all(
        item["passed"] for item in results
    )
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "claim_scope": (
            "Per-request engine generation replay only; this artifact does not claim "
            "full BFCL task, tool execution, or scorer coverage."
        ),
        "passed": passed,
        "journal": str(journal_path.resolve()),
        "journal_sha256": loaded["source_sha256"],
        "base_url": base_url,
        "endpoint": ENDPOINT,
        "requests_total": loaded["request_count"],
        "responses_total": loaded["response_count"],
        "pairs": len(loaded["pairs"]),
        "requests_attempted": attempted,
        "requests_replayed": replayed,
        "requests_with_identical_ids": identical_ids,
        "token_positional_match": positional_matches,
        "token_expected_total": expected_total,
        "token_expected_unavailable_requests": expected_unavailable,
        "token_match_rate": positional_matches / expected_total if expected_total else None,
        "detector": detector_metadata,
        "journal_errors": loaded["errors"],
        "setup_errors": setup_errors,
        "unpaired_records": loaded["unpaired_records"],
        "results": results,
    }


def write_result(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:36100")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--controller-config", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or _default_out(args.journal)
    if out.resolve() == args.journal.resolve():
        print("FATAL: --out must not overwrite --journal", file=sys.stderr)
        return 2
    try:
        result = replay(
            args.journal,
            args.base_url,
            controller_config=args.controller_config,
            runtime_root=args.runtime_root,
        )
    except Exception as exc:  # noqa: BLE001 - preserve a machine-readable failure artifact
        source_sha256 = None
        try:
            source_sha256 = _sha256_bytes(args.journal.read_bytes())
        except OSError:
            pass
        result = {
            "schema": SCHEMA,
            "scope": SCOPE,
            "claim_scope": (
                "Per-request engine generation replay only; this artifact does not claim "
                "full BFCL task, tool execution, or scorer coverage."
            ),
            "passed": False,
            "journal": str(args.journal.resolve()),
            "journal_sha256": source_sha256,
            "base_url": args.base_url,
            "endpoint": ENDPOINT,
            "fatal_errors": [_error("replay_fatal", repr(exc))],
            "results": [],
        }
    write_result(out, result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "scope": result["scope"],
                "out": str(out.resolve()),
                "pairs": result.get("pairs", 0),
                "token_positional_match": result.get("token_positional_match", 0),
                "token_expected_total": result.get("token_expected_total", 0),
            },
            indent=2,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
