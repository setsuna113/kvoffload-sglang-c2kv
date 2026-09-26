"""CPU-only regression for strict C2KV sampler failure handling."""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest
import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "python"))
_SAMPLER_PATH = os.path.join(
    _PYTHON_ROOT, "sglang", "srt", "layers", "sampler.py"
)
_NATIVE_PATH = os.path.join(
    _PYTHON_ROOT, "sglang", "srt", "mem_cache", "c2kv_native_packed.py"
)


def _module(name, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    return module


def _load_sampler(monkeypatch):
    native_spec = importlib.util.spec_from_file_location(
        "sglang.srt.mem_cache.c2kv_native_packed", _NATIVE_PATH
    )
    native = importlib.util.module_from_spec(native_spec)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.mem_cache.c2kv_native_packed", native
    )
    native_spec.loader.exec_module(native)

    stubs = {
        "sglang.srt.distributed": _module("distributed", get_tp_group=lambda: None),
        "sglang.srt.layers.dp_attention": _module(
            "dp_attention",
            get_attention_tp_group=lambda: None,
            is_dp_attention_enabled=lambda: False,
        ),
        "sglang.srt.layers.logits_processor": _module(
            "logits_processor", LogitsProcessorOutput=type("LogitsProcessorOutput", (), {})
        ),
        "sglang.srt.layers.utils.hash": _module(
            "hash", murmur_hash32=lambda *args, **kwargs: 0
        ),
        "sglang.srt.layers.utils.logprob": _module(
            "logprob",
            get_token_ids_logprobs=lambda *args, **kwargs: None,
            get_top_logprobs=lambda *args, **kwargs: None,
        ),
        "sglang.srt.sampling.sampling_batch_info": _module(
            "sampling_batch_info", SamplingBatchInfo=type("SamplingBatchInfo", (), {})
        ),
        "sglang.srt.sampling.sampling_params": _module(
            "sampling_params", TOP_K_ALL=-1
        ),
        "sglang.srt.server_args": _module(
            "server_args", get_global_server_args=lambda: None
        ),
        "sglang.srt.utils.common": _module(
            "common",
            crash_on_warnings=lambda: False,
            get_bool_env_var=lambda _name: False,
            is_cuda=lambda: False,
            is_npu=lambda: False,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("c2kv_sampler_under_test", _SAMPLER_PATH)
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    return sampler


def test_strict_guard_accepts_finite_logits_and_masked_negative_infinity(monkeypatch):
    sampler = _load_sampler(monkeypatch)
    logits = torch.tensor([[1.0, -2.0, -float("inf")]], dtype=torch.float32)
    before = logits.clone()

    sampler._raise_on_c2kv_nonfinite_logits(logits)

    assert torch.equal(logits, before)


def test_strict_guard_fails_before_sampling_without_rewriting_logits(monkeypatch):
    sampler = _load_sampler(monkeypatch)
    logits = torch.tensor(
        [[float("nan"), float("inf"), -float("inf"), 0.0]], dtype=torch.float32
    )
    before = logits.clone()

    with pytest.raises(ValueError, match="C2KV_NATIVE_NONFINITE_LOGITS") as captured:
        sampler._raise_on_c2kv_nonfinite_logits(logits)

    assert "batch_rows=1" in str(captured.value)
    assert "nan=1" in str(captured.value)
    assert "posinf=1" in str(captured.value)
    assert "masked_neginf=1" in str(captured.value)
    assert torch.equal(torch.isnan(logits), torch.isnan(before))
    assert torch.equal(torch.isposinf(logits), torch.isposinf(before))
    assert torch.equal(torch.isneginf(logits), torch.isneginf(before))


def test_strict_guard_rejects_row_without_finite_candidate(monkeypatch):
    sampler = _load_sampler(monkeypatch)
    logits = torch.tensor([[-float("inf"), -float("inf")]], dtype=torch.float32)
    before = logits.clone()

    with pytest.raises(ValueError, match="C2KV_NATIVE_NONFINITE_LOGITS") as captured:
        sampler._raise_on_c2kv_nonfinite_logits(logits)

    assert "no_finite_candidate_rows=1" in str(captured.value)
    assert torch.equal(logits, before)
