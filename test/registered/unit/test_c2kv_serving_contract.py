"""CPU-only contracts for C2KV request policy and graph eligibility."""

from __future__ import annotations

import ast
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


def _cuda_graph_runner_source():
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
        return source_file.read()


def _cuda_graph_method(name):
    graph_runner = next(
        node
        for node in ast.parse(_cuda_graph_runner_source()).body
        if isinstance(node, ast.ClassDef) and node.name == "CudaGraphRunner"
    )
    method = next(
        node
        for node in graph_runner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {
        "is_c2kv_graph_compatible": semantics.is_c2kv_graph_compatible,
        "CaptureHiddenMode": SimpleNamespace(NULL=0),
    }
    exec("from __future__ import annotations\n" + ast.unparse(method), namespace)  # noqa: S102
    return namespace[name]


def _graph_runner_stub(base_query_graph):
    return SimpleNamespace(
        c2kv_base_query_graph=base_query_graph,
        require_mlp_tp_gather=False,
        num_tokens_per_bs=1,
        enable_pdmux=False,
        disable_padding=True,
        graphs={1: object()},
        require_mlp_sync=False,
        is_encoder_decoder=False,
        capture_hidden_mode=0,
        enable_two_batch_overlap=False,
        model_runner=SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_ngram=lambda: False)
        ),
    )


def _graph_forward_batch(mask):
    return SimpleNamespace(
        c2kv_use_gist_projection=mask,
        batch_size=1,
        capture_hidden_mode=0,
        spec_info=None,
    )


def test_full_cuda_graph_runner_selects_base_only_capture_when_opted_in():
    source = _cuda_graph_runner_source()
    assert "def update_c2kv_gist_projection_mask(" in source
    graph_runner = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "CudaGraphRunner"
    )
    init = next(
        node
        for node in graph_runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    flag = next(
        node.value
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "c2kv_base_query_graph"
            for target in node.targets
        )
    )
    create = next(
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "DecodeInputBuffers"
        and node.func.attr == "create"
    )
    mask_allocation = next(
        keyword.value
        for keyword in create.keywords
        if keyword.arg == "enable_c2kv_query_projection"
    )

    for enabled, c2kv, pic, expected_base in (
        (False, True, False, False),
        (True, False, False, False),
        (True, True, True, False),
        (True, True, False, True),
    ):
        model_runner = SimpleNamespace(
            server_args=SimpleNamespace(enable_c2kv=c2kv),
            model=SimpleNamespace(full_length_pic=pic),
        )
        base_mode = eval(
            compile(ast.Expression(flag), "<c2kv-base-graph-flag>", "eval"),
            {
                "get_bool_env_var": lambda name, enabled=enabled: enabled
                if name == "C2KV_BASE_QUERY_GRAPH"
                else False,
                "model_runner": model_runner,
            },
        )
        assert base_mode is expected_base
        projection_buffer = eval(
            compile(ast.Expression(mask_allocation), "<c2kv-graph-buffer>", "eval"),
            {
                "model_runner": model_runner,
                "self": SimpleNamespace(c2kv_base_query_graph=base_mode),
            },
        )
        assert projection_buffer is (c2kv and not pic and not expected_base)


@pytest.mark.parametrize(
    "mask, expected",
    [(None, True), ([True], False), ([True, False], False), ([False, False], False)],
)
def test_base_query_graph_eligibility_rejects_every_mask(mask, expected):
    can_run = _cuda_graph_method("can_run")
    assert can_run(_graph_runner_stub(True), _graph_forward_batch(mask)) is expected
    assert can_run(_graph_runner_stub(False), _graph_forward_batch(mask)) is True


@pytest.mark.parametrize("mask", [[True], [True, False], [False, False]])
def test_direct_base_query_graph_replay_rejects_every_mask(mask):
    replay = _cuda_graph_method("replay")
    with pytest.raises(RuntimeError, match="C2KV base-query CUDA graph"):
        replay(_graph_runner_stub(True), _graph_forward_batch(mask))
