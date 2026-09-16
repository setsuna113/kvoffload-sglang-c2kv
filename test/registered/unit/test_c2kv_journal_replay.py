from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "c2kv" / "replay_native_journal.py"
SPEC = importlib.util.spec_from_file_location("c2kv_journal_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def shadow(hidden: list[float]) -> dict:
    return {
        "schema": "event-native-shadow-features-v1",
        "prefill": {
            "layer": 34,
            "position": {"kind": "prompt_last", "logical_position": 9},
            "readout": "decoder_layer_output",
            "status": "captured",
            "hidden": hidden,
        },
    }


def request(rid: str, *, request_shadow: bool = True) -> dict:
    value = {"rid": rid, "session_id": "bfcl/task-0", "input_ids": [1, 2]}
    if request_shadow:
        value["shadow_features"] = {"prefill_layer": -2}
    return value


def response(
    rid: str,
    *,
    session_id: str = "bfcl/task-0",
    ids: list[int] | None = None,
    text: str = "answer",
    finish: str = "stop",
    response_shadow: dict | None = None,
) -> dict:
    return {
        "rid": rid,
        "session_id": session_id,
        "output_ids": [10, 11] if ids is None else ids,
        "text": text,
        "finish_reason": finish,
        "shadow_features": shadow([1.0, 2.0]) if response_shadow is None else response_shadow,
    }


def rows_for(rid: str, **response_kwargs) -> list[dict]:
    return [
        {"event": "request", "request": request(rid)},
        {"event": "response", "response": response(rid, **response_kwargs)},
    ]


def write_rows(path: Path, rows: list[dict | str]) -> Path:
    rendered = [item if isinstance(item, str) else json.dumps(item) for item in rows]
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    return path


def actual_for(payload: dict, **overrides) -> dict:
    value = response(payload["rid"])
    value.update(overrides)
    return value


def error_codes(result: dict) -> set[str]:
    return {error["code"] for error in result["errors"]}


def test_http_failures_stay_in_full_token_denominator(tmp_path: Path):
    rows = []
    for index in range(21):
        rows.extend(rows_for(f"r{index}", ids=[index]))
    journal = write_rows(tmp_path / "journal.jsonl", rows)

    def fake_post(_base_url, payload):
        if payload["rid"] != "r0":
            raise RuntimeError("HTTP 500")
        return actual_for(payload, output_ids=[0], text="answer", finish_reason="stop")

    result = replay.replay(journal, "http://127.0.0.1:1", post_fn=fake_post)

    assert result["passed"] is False
    assert result["requests_attempted"] == 21
    assert result["requests_replayed"] == 1
    assert result["requests_with_identical_ids"] == 1
    assert result["token_positional_match"] == 1
    assert result["token_expected_total"] == 21
    assert result["token_match_rate"] == pytest.approx(1 / 21)
    assert sum(not item["passed"] for item in result["results"]) == 20


def test_finish_and_text_mismatches_fail_even_when_ids_match(tmp_path: Path):
    journal = write_rows(tmp_path / "journal.jsonl", rows_for("r1"))
    result = replay.replay(
        journal,
        "http://127.0.0.1:1",
        post_fn=lambda _url, payload: actual_for(
            payload, text="different", finish_reason="length"
        ),
    )

    row = result["results"][0]
    assert row["ids_equal"] is True
    assert row["passed"] is False
    assert {"text_mismatch", "finish_reason_mismatch"} <= error_codes(row)


@pytest.mark.parametrize(
    ("target", "mode", "codes"),
    [
        (
            "actual",
            "wrong",
            {
                "actual_rid_mismatch",
                "actual_session_id_mismatch",
                "actual_generation_id_mismatch",
            },
        ),
        (
            "actual",
            "missing",
            {
                "actual_rid_missing",
                "actual_session_id_missing",
                "actual_generation_id_missing",
            },
        ),
        (
            "expected",
            "wrong",
            {"expected_session_id_mismatch", "expected_generation_id_mismatch"},
        ),
        (
            "expected",
            "missing",
            {"expected_session_id_missing", "expected_generation_id_missing"},
        ),
    ],
)
def test_request_identity_fields_are_required_and_must_match(
    tmp_path: Path, target: str, mode: str, codes: set[str]
):
    rows = rows_for("identity")
    rows[0]["request"]["generation_id"] = "generation-0"
    rows[1]["response"]["generation_id"] = "generation-0"
    actual = response("identity")
    actual["generation_id"] = "generation-0"
    response_value = rows[1]["response"] if target == "expected" else actual
    fields = ("session_id", "generation_id") if target == "expected" else (
        "rid",
        "session_id",
        "generation_id",
    )
    for field in fields:
        if mode == "wrong":
            response_value[field] = f"wrong-{field}"
        else:
            response_value.pop(field)

    journal = write_rows(tmp_path / "journal.jsonl", rows)
    result = replay.replay(journal, "http://127.0.0.1:1", post_fn=lambda *_: actual)

    row = result["results"][0]
    assert row["ids_equal"] is True
    assert row["passed"] is False
    assert codes <= error_codes(row)


def test_complete_raw_json_and_source_identity_are_persisted(tmp_path: Path):
    journal = write_rows(tmp_path / "journal.jsonl", rows_for("raw"))
    actual = response("raw")
    result = replay.replay(
        journal, "http://engine:30000", post_fn=lambda _url, _payload: actual
    )
    out = tmp_path / "result.json"
    replay.write_result(out, result)
    saved = json.loads(out.read_text(encoding="utf-8"))

    assert saved["scope"] == "engine_generation_replay"
    assert saved["base_url"] == "http://engine:30000"
    assert saved["journal_sha256"] == replay._sha256_bytes(journal.read_bytes())
    assert saved["results"][0]["request"] == request("raw")
    assert saved["results"][0]["expected"] == response("raw")
    assert saved["results"][0]["actual"] == actual
    assert "full BFCL" in saved["claim_scope"]


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        ([], "journal_has_no_pairs"),
        (["{"], "journal_json_invalid"),
        ([{"event": "response", "response": response("orphan")}], "orphan_response"),
        (
            [
                {"event": "request", "request": request("dup")},
                {"event": "request", "request": request("dup")},
                {"event": "response", "response": response("dup")},
            ],
            "duplicate_request",
        ),
        (
            [
                {"event": "request", "request": request("dup-response")},
                {"event": "response", "response": response("dup-response")},
                {"event": "response", "response": response("dup-response")},
            ],
            "duplicate_response",
        ),
        ([{"event": "request", "request": request("missing")}], "missing_response"),
    ],
)
def test_invalid_journals_fail_without_http(tmp_path: Path, rows, code):
    journal = write_rows(tmp_path / "journal.jsonl", rows)
    calls = []
    result = replay.replay(
        journal,
        "http://127.0.0.1:1",
        post_fn=lambda *_args: calls.append(True),
    )

    assert result["passed"] is False
    assert code in {error["code"] for error in result["journal_errors"]}
    assert calls == []


def test_missing_requested_shadow_features_fails(tmp_path: Path):
    journal = write_rows(tmp_path / "journal.jsonl", rows_for("shadow"))
    actual = response("shadow")
    actual.pop("shadow_features")
    result = replay.replay(journal, "http://127.0.0.1:1", post_fn=lambda *_: actual)

    row = result["results"][0]
    assert row["passed"] is False
    assert "actual_shadow_missing" in error_codes(row)


def test_disabled_shadow_features_are_not_requested(tmp_path: Path):
    rows = rows_for("shadow-disabled")
    rows[0]["request"]["shadow_features"] = {"enabled": False}
    actual = response("shadow-disabled")
    actual.pop("shadow_features")
    journal = write_rows(tmp_path / "journal.jsonl", rows)
    result = replay.replay(journal, "http://127.0.0.1:1", post_fn=lambda *_: actual)

    row = result["results"][0]
    assert row["passed"] is True
    assert row["shadow_features"] == {
        "status": "not_requested",
        "passed": True,
        "errors": [],
    }
    assert row["detector"] == {"status": "not_requested", "passed": True, "errors": []}


def test_missing_actual_response_field_fails(tmp_path: Path):
    journal = write_rows(tmp_path / "journal.jsonl", rows_for("missing-field"))
    actual = response("missing-field")
    actual.pop("finish_reason")
    result = replay.replay(journal, "http://127.0.0.1:1", post_fn=lambda *_: actual)

    row = result["results"][0]
    assert row["passed"] is False
    assert "actual_finish_reason_missing" in error_codes(row)
    assert "finish_reason_mismatch" in error_codes(row)


def test_hidden_numeric_difference_is_reported_without_requiring_bit_identity(tmp_path: Path):
    journal = write_rows(tmp_path / "journal.jsonl", rows_for("features"))
    actual = response("features", response_shadow=shadow([1.25, 1.5]))
    result = replay.replay(journal, "http://127.0.0.1:1", post_fn=lambda *_: actual)

    comparison = result["results"][0]["shadow_features"]
    assert result["passed"] is True
    assert comparison["hidden_dimension"] == 2
    assert comparison["hidden_max_abs_difference"] == pytest.approx(0.5)
    assert isinstance(comparison["hidden_cosine"], float)


def test_detector_gate_mismatch_fails_and_records_scores_and_head_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    journal = write_rows(tmp_path / "journal.jsonl", rows_for("gate"))
    controller = tmp_path / "controller.json"
    controller.write_text(
        json.dumps(
            {
                "post_draft_recovery": {
                    "prefill_head": {
                        "artifact_sha256": "a" * 64,
                        "threshold": 0.5,
                        "weights": [1.0, 1.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    gate_dir = runtime / "benchmarks" / "memory_runtime" / "recovery"
    gate_dir.mkdir(parents=True)
    (gate_dir / "gate.py").write_text(
        "def _read_prefill_score(value, head):\n"
        "    return float(value['prefill']['hidden'][0]), None\n",
        encoding="utf-8",
    )
    actual = response("gate", response_shadow=shadow([0.25, 2.0]))
    expected_rows = rows_for("gate", response_shadow=shadow([0.75, 2.0]))
    journal = write_rows(journal, expected_rows)

    result = replay.replay(
        journal,
        "http://127.0.0.1:1",
        controller_config=controller,
        runtime_root=runtime,
        post_fn=lambda *_: actual,
    )

    detector = result["results"][0]["detector"]
    assert result["passed"] is False
    assert detector["expected_score"] == 0.75
    assert detector["actual_score"] == 0.25
    assert detector["gate_agrees"] is False
    assert detector["head_artifact_sha256"] == "a" * 64
    assert "detector_gate_mismatch" in error_codes(result["results"][0])


def test_detector_is_explicitly_not_checked_without_controller(tmp_path: Path):
    journal = write_rows(tmp_path / "journal.jsonl", rows_for("no-head"))
    result = replay.replay(
        journal, "http://127.0.0.1:1", post_fn=lambda *_: response("no-head")
    )

    assert result["detector"]["status"] == "detector_not_checked"
    assert result["results"][0]["detector"]["status"] == "detector_not_checked"


def test_main_writes_failure_artifact_and_returns_nonzero(tmp_path: Path, monkeypatch):
    journal = write_rows(tmp_path / "journal.jsonl", rows_for("cli"))
    out = tmp_path / "result.json"
    monkeypatch.setattr(replay, "post", lambda *_: (_ for _ in ()).throw(RuntimeError("down")))

    code = replay.main(
        ["--journal", str(journal), "--base-url", "http://127.0.0.1:1", "--out", str(out)]
    )

    assert code == 1
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is False


def test_main_refuses_to_overwrite_input(tmp_path: Path):
    journal = write_rows(tmp_path / "journal.jsonl", rows_for("same"))
    before = journal.read_bytes()

    code = replay.main(["--journal", str(journal), "--out", str(journal)])

    assert code == 2
    assert journal.read_bytes() == before
