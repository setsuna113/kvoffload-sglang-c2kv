"""CPU-only checks for the C2KV foreground stream priority opt-in."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_MODEL_RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/model_executor/model_runner.py"
)


def _stream_init_code():
    tree = ast.parse(_MODEL_RUNNER_PATH.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    init = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    start = next(
        index
        for index, node in enumerate(init.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "foreground_priority_opt_in"
            for target in node.targets
        )
    )
    end = next(
        index
        for index in range(start + 1, len(init.body))
        if isinstance(init.body[index], ast.If)
        and isinstance(init.body[index].test, ast.Name)
        and init.body[index].test.id == "foreground_priority_opt_in"
    )
    return compile(
        ast.fix_missing_locations(
            ast.Module(body=init.body[start : end + 1], type_ignores=[])
        ),
        str(_MODEL_RUNNER_PATH),
        "exec",
    )


_STREAM_INIT_CODE = _stream_init_code()


def _run_stream_init(
    device, high_priority, gist_async, overlap_enabled, actual_priority=None
):
    calls = []
    logs = []
    environment = {}
    if high_priority is not None:
        environment["C2KV_FOREGROUND_HIGH_PRIORITY"] = high_priority
    if gist_async is not None:
        environment["C2KV_GIST_ASYNC"] = gist_async

    def create_stream(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            priority=(
                kwargs.get("priority", 0)
                if actual_priority is None
                else actual_priority
            )
        )

    runner = SimpleNamespace(device=device)

    def get_device_module(selected_device):
        assert selected_device == device
        return SimpleNamespace(Stream=create_stream)

    namespace = {
        "self": runner,
        "server_args": SimpleNamespace(disable_overlap_schedule=not overlap_enabled),
        "os": SimpleNamespace(environ=environment),
        "torch": SimpleNamespace(get_device_module=get_device_module),
        "log_info_on_rank0": lambda _logger, message: logs.append(message),
        "logger": object(),
    }
    exec(_STREAM_INIT_CODE, namespace)
    return calls, logs, runner.forward_stream


@pytest.mark.parametrize(
    "device,high_priority,gist_async,overlap_enabled,expected_logs",
    [
        ("cuda", None, "1", True, 0),
        ("cuda", "0", "1", True, 0),
        ("cuda", "1", None, True, 1),
        ("cuda", "1", "false", True, 1),
        ("cpu", "1", "1", True, 1),
        ("xpu", "1", "1", True, 1),
        ("cuda", "1", "1", False, 1),
    ],
)
def test_ineligible_configs_preserve_default_stream_call(
    device, high_priority, gist_async, overlap_enabled, expected_logs
):
    calls, logs, stream = _run_stream_init(
        device, high_priority, gist_async, overlap_enabled
    )

    assert calls == [{}]
    assert stream.priority == 0
    assert len(logs) == expected_logs
    if expected_logs:
        assert "selected=0, actual=0" in logs[0]


@pytest.mark.parametrize("actual_priority", [-1, 0])
def test_cuda_async_overlap_opt_in_requests_high_priority_and_logs_actual(
    actual_priority,
):
    calls, logs, stream = _run_stream_init(
        "cuda", "TRUE", "yes", True, actual_priority=actual_priority
    )

    assert calls == [{"priority": -1}]
    assert stream.priority == actual_priority
    assert len(logs) == 1
    assert f"selected=-1, actual={actual_priority}" in logs[0]
