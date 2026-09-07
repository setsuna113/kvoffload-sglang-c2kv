"""CPU-only contracts for C2KV request policy and graph eligibility."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.normpath(
    os.path.join(
        _HERE,
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "srt",
        "mem_cache",
        "c2kv_semantics.py",
    )
)
_SPEC = importlib.util.spec_from_file_location("c2kv_semantics_under_test", _PATH)
semantics = importlib.util.module_from_spec(_SPEC)
sys.modules["c2kv_semantics_under_test"] = semantics
_SPEC.loader.exec_module(semantics)


@pytest.mark.parametrize("default, expected", [("base", False), ("gist", True)])
def test_projection_falls_back_to_server_flag(default, expected):
    assert semantics.resolve_query_projection(default, None, [None, None]) == (
        expected,
        "flag",
    )


def test_gist_cache_identity_includes_ratio_and_extractor_config():
    ids = [11, 12, 13]
    base = semantics.compute_gist_cache_key(ids, 4, {"gist_type": "dynamic"})
    assert base == semantics.compute_gist_cache_key(
        ids, 4, {"gist_type": "dynamic"}
    )
    assert base != semantics.compute_gist_cache_key(
        ids, 8, {"gist_type": "dynamic"}
    )
    assert base != semantics.compute_gist_cache_key(
        ids, 4, {"gist_type": "static"}
    )


def test_rope_position_range_rejects_aliasing():
    semantics.validate_rope_position_range(0, 31, 32)
    with pytest.raises(ValueError, match="C2KV_ROPE_POSITION_OUT_OF_RANGE"):
        semantics.validate_rope_position_range(-1, 2, 32)
    with pytest.raises(ValueError, match="C2KV_ROPE_POSITION_OUT_OF_RANGE"):
        semantics.validate_rope_position_range(4, 32, 32)


def test_injection_uses_range_validation_instead_of_position_clamping():
    path = os.path.normpath(
        os.path.join(
            _HERE,
            "..",
            "..",
            "..",
            "python",
            "sglang",
            "srt",
            "mem_cache",
            "c2kv_injection.py",
        )
    )
    with open(path, encoding="utf-8") as source_file:
        source = source_file.read()
    assert source.count("_validate_rope_positions(") >= 3
    assert ".clamp(" not in source


def test_mixed_case_gist_param_is_rejected_instead_of_lowercased():
    assert semantics.validate_gist_param("qkv") == "qkv"
    with pytest.raises(ValueError, match="C2KV_GIST_PARAM_CASE_UNSUPPORTED"):
        semantics.validate_gist_param("QkV")


def test_request_projection_overrides_messages_and_reports_request_source():
    assert semantics.resolve_query_projection("base", True, [False, None]) == (
        True,
        "request",
    )
    assert semantics.resolve_query_projection("gist", False, [True]) == (
        False,
        "request",
    )


def test_single_explicit_message_value_controls_the_whole_request():
    # In particular, False must not be turned back into gist merely because a
    # second annotated message is unset and the server default is gist.
    assert semantics.resolve_query_projection("gist", None, [False, None]) == (
        False,
        "message",
    )
    assert semantics.resolve_query_projection("base", None, [None, True]) == (
        True,
        "message",
    )


def test_conflicting_message_projection_values_are_rejected():
    with pytest.raises(ValueError, match="C2KV_QUERY_PROJECTION_CONFLICT"):
        semantics.resolve_query_projection("base", None, [True, False])


@pytest.mark.parametrize(
    "repair_mode, expected",
    [
        ("d_corr_recompute", "in_place"),
        ("history_kv_h2o", "in_place"),
        ("cacheblend", "in_place"),
        ("d_corr", "append_keep_ledger"),
    ],
)
def test_legacy_repair_placement_is_stable(repair_mode, expected):
    assert semantics.resolve_repair_placement(repair_mode, None) == expected


def test_explicit_repair_placement_wins_and_unknown_value_is_rejected():
    assert (
        semantics.resolve_repair_placement("d_corr_recompute", "append_tail")
        == "append_tail"
    )
    with pytest.raises(ValueError, match="Unknown c2kv_repair_placement"):
        semantics.resolve_repair_placement("d_corr", "tail")


def test_any_projection_mask_forces_eager_execution():
    assert semantics.is_c2kv_graph_compatible(SimpleNamespace())
    assert semantics.is_c2kv_graph_compatible(
        SimpleNamespace(c2kv_use_gist_projection=None)
    )
    assert not semantics.is_c2kv_graph_compatible(
        SimpleNamespace(c2kv_use_gist_projection=[True])
    )


@pytest.mark.parametrize("runner", ["cpu_graph_runner.py", "piecewise_cuda_graph_runner.py"])
def test_graph_runners_without_projection_buffers_apply_the_compatibility_gate(runner):
    path = os.path.normpath(
        os.path.join(
            _HERE,
            "..",
            "..",
            "..",
            "python",
            "sglang",
            "srt",
            "model_executor",
            runner,
        )
    )
    with open(path, encoding="utf-8") as source_file:
        source = source_file.read()
    assert "if not is_c2kv_graph_compatible(forward_batch):" in source


def test_full_cuda_graph_runner_owns_the_dynamic_projection_mask():
    path = os.path.normpath(
        os.path.join(
            _HERE,
            "..",
            "..",
            "..",
            "python",
            "sglang",
            "srt",
            "model_executor",
            "cuda_graph_runner.py",
        )
    )
    with open(path, encoding="utf-8") as source_file:
        source = source_file.read()
    assert "def update_c2kv_gist_projection_mask(" in source
    assert "if not is_c2kv_graph_compatible(forward_batch):" not in source
