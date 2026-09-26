"""CPU regression for C2KV position correction transport and arithmetic."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/model_executor/forward_batch_info.py"
)


@pytest.fixture(scope="module")
def correction_code():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    forward_batch = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ForwardBatch"
    )
    init_new = next(
        node for node in forward_batch.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_new"
    )
    branches = [
        node for node in init_new.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Attribute)
        and node.test.left.attr == "c2kv_position_corrections"
    ]
    assert len(branches) == 1
    return compile(
        ast.Module(body=branches, type_ignores=[]), str(SOURCE), "exec"
    )


class _CpuTensor:
    def __init__(self, value, transport):
        self.value = value
        self.transport = transport

    def to(self, device, *, non_blocking=False):
        self.transport.transfers.append((self.value, device, non_blocking))
        return self.value.to(device, non_blocking=non_blocking)


class _TorchTransport:
    int32 = torch.int32
    int64 = torch.int64

    def __init__(self):
        self.creations = []
        self.transfers = []

    def tensor(self, values, *, dtype, device=None):
        self.creations.append((dtype, device))
        value = torch.tensor(values, dtype=dtype, device=device)
        return _CpuTensor(value, self) if device is None else value

    def repeat_interleave(self, *args, **kwargs):
        return torch.repeat_interleave(*args, **kwargs)


class _Mode:
    def __init__(self, value):
        self.value = value

    def is_decode(self):
        return self.value == "decode"

    def is_target_verify(self):
        return self.value == "target_verify"


def _run_correction(code, mode, positions, corrections, extend_seq_lens=None):
    transport = _TorchTransport()
    ret = SimpleNamespace(forward_mode=_Mode(mode), positions=positions)
    batch = SimpleNamespace(
        c2kv_position_corrections=corrections,
        extend_seq_lens=extend_seq_lens,
    )
    exec(code, {"torch": transport}, {
        "batch": batch, "ret": ret, "device": torch.device("cpu")
    })
    return ret.positions, transport


def _assert_correction_transfer(transport):
    assert transport.creations[0] == (torch.int64, None)
    assert len(transport.transfers) == 1
    source, device, non_blocking = transport.transfers[0]
    assert source.device.type == "cpu"
    assert source.dtype == torch.int64
    assert device == torch.device("cpu")
    assert non_blocking is True


@pytest.mark.parametrize("mode", ["decode", "target_verify"])
@pytest.mark.parametrize(
    "corrections",
    [[0], [-3], [0, -3, 2, 7, -1, 0, 5, -8]],
)
def test_decode_correction_values_and_nonblocking_transfer(
    correction_code, mode, corrections
):
    base = torch.arange(10, 10 + len(corrections), dtype=torch.int32)
    result, transport = _run_correction(
        correction_code, mode, base, corrections
    )

    assert result.dtype == torch.int64
    assert result.tolist() == [10 + i + value for i, value in enumerate(corrections)]
    _assert_correction_transfer(transport)


@pytest.mark.parametrize(
    "corrections,extend_seq_lens",
    [([-2], [3]), ([0, -3, 2, 7, -1, 0, 5, -8], [2, 1, 0, 3, 1, 2, 1, 1])],
)
def test_extend_correction_repeats_per_request(
    correction_code, corrections, extend_seq_lens
):
    base = torch.arange(sum(extend_seq_lens), dtype=torch.int32)
    result, transport = _run_correction(
        correction_code, "extend", base, corrections, extend_seq_lens
    )

    expanded = [
        correction
        for correction, count in zip(corrections, extend_seq_lens)
        for _ in range(count)
    ]
    assert result.dtype == torch.int64
    assert result.tolist() == [i + value for i, value in enumerate(expanded)]
    _assert_correction_transfer(transport)


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_missing_correction_keeps_positions_untouched(correction_code, mode):
    base = torch.tensor([4, 5], dtype=torch.int32)
    result, transport = _run_correction(
        correction_code, mode, base, None, [1, 1]
    )

    assert result is base
    assert transport.creations == []
    assert transport.transfers == []
