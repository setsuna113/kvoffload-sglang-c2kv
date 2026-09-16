"""Exercise the CUDA launcher without loading a model or allocating a GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[3] / "scripts/c2kv/launch_cuda_native.sh"


@pytest.fixture
def launch_env(tmp_path):
    if not shutil.which("bash") or os.name == "nt":
        pytest.skip("The CUDA launcher targets Linux/WSL bash")
    checkpoint = tmp_path / "checkpoint-real"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(checkpoint, target_is_directory=True)
    capture = tmp_path / "argv.json"
    python_stub = tmp_path / "python-stub"
    python_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    python_stub.chmod(0o755)
    env = {**os.environ, "CKPT": str(alias), "PYTHON_BIN": str(python_stub),
           "CAPTURE": str(capture), "PAGE_SIZE": "1"}
    return env, capture, alias


def test_checkpoint_alias_and_cuda_contract_are_preserved(launch_env):
    env, capture, alias = launch_env
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())
    assert argv[argv.index("--model-path") + 1] == str(alias)
    for flag, value in (("--device", "cuda"), ("--attention-backend", "torch_native"),
                        ("--page-size", "1"), ("--max-running-requests", "1")):
        assert argv[argv.index(flag) + 1] == value
    assert "--disable-overlap-schedule" in argv


@pytest.mark.parametrize("failure", ["missing_identity", "missing_config", "paged_cuda"])
def test_invalid_launch_fails_before_python(launch_env, failure):
    env, capture, _ = launch_env
    if failure == "missing_identity":
        env.pop("CKPT")
    elif failure == "missing_config":
        env["CKPT"] += "-missing"
    else:
        env["PAGE_SIZE"] = "128"
    result = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not capture.exists()
