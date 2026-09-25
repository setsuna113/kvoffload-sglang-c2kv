"""C2KV gist FlexAttention on GPUs whose default tile config exceeds shared memory.

box9 RTX 4090 (sm_89), ToolSandbox find_current_location_insufficient_information:
the compiled extraction for q/k/v [1, 32|8, 32, 128] chose the A100 default
(BLOCK_M=128, BLOCK_N=64, 3 stages, 8 warps) and failed with "No valid triton
configs. OutOfMemoryError: out of resource: triton_tem_fused_0 Required: 106496
Hardware limit: 101376".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python" / "sglang" / "srt"


def _load(relative):
    name = "sglang.srt." + relative[:-3].replace("/", ".")
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SRT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gist_utils = _load("mem_cache/gist_utils.py")
LIVE_ERROR = ("No valid triton configs. OutOfMemoryError: out of resource: triton_tem_fused_0 "
              "Required: 106496 Hardware limit: 101376 Reducing block sizes or `num_stages` may help.")


class Recorder:
    def __init__(self, name, fail_for=(), error=None):
        self.name, self.fail_for, self.error, self.calls = name, set(fail_for), error, []

    def __call__(self, query, key, value, **kwargs):
        self.calls.append(tuple(query.shape))
        if tuple(query.shape) in self.fail_for:
            raise self.error
        return self.name


def _qkv(length):
    return (torch.zeros(1, 32, length, 128, dtype=torch.bfloat16),
            torch.zeros(1, 8, length, 128, dtype=torch.bfloat16),
            torch.zeros(1, 8, length, 128, dtype=torch.bfloat16))


def test_shapes_that_compile_never_touch_the_fallback():
    primary = Recorder("primary")
    built = []
    attention = gist_utils.FlexAttentionSharedMemoryFallback(
        primary, lambda: built.append(1) or Recorder("fallback"))
    for length in (512, 1037, 32):
        assert attention(*_qkv(length), block_mask=None, scale=0.1, enable_gqa=True) == "primary"
    assert built == [] and not attention.fallback_shapes


def test_only_the_failing_shape_uses_the_small_tile_compile():
    primary = Recorder("primary", fail_for={(1, 32, 32, 128)}, error=RuntimeError(LIVE_ERROR))
    fallback = Recorder("fallback")
    factory_calls = []
    attention = gist_utils.FlexAttentionSharedMemoryFallback(
        primary, lambda: factory_calls.append(1) or fallback)
    assert attention(*_qkv(32), scale=0.1) == "fallback"
    assert attention(*_qkv(32), scale=0.1) == "fallback"
    assert attention(*_qkv(512), scale=0.1) == "primary"
    # The failing shape is compiled once with the default config, never retried.
    assert primary.calls == [(1, 32, 32, 128), (1, 32, 512, 128)]
    assert fallback.calls == [(1, 32, 32, 128)] * 2 and factory_calls == [1]


@pytest.mark.parametrize("error", [
    RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
    RuntimeError("No valid triton configs. AssertionError: something else"),
    ValueError("shape mismatch"),
])
def test_other_failures_propagate(error):
    attention = gist_utils.FlexAttentionSharedMemoryFallback(
        Recorder("primary", fail_for={(1, 32, 32, 128)}, error=error),
        lambda: pytest.fail("only a shared-memory compile failure may fall back"))
    with pytest.raises(type(error)):
        attention(*_qkv(32))


def test_small_tiles_divide_the_block_mask_and_fit_99_kb():
    options = gist_utils.C2KV_SMALL_SMEM_KERNEL_OPTIONS
    assert options["FORCE_USE_FLEX_ATTENTION"] is True
    assert 128 % options["BLOCK_M"] == 0 and 128 % options["BLOCK_N"] == 0
    head_dim, element = 128, 2  # bf16
    # Q tile plus K and V tiles per pipeline stage, as Triton buffers them.
    smem = (options["BLOCK_M"] * head_dim
            + 2 * options["num_stages"] * options["BLOCK_N"] * head_dim) * element
    assert smem < 101376
    assert gist_utils.C2KV_KERNEL_OPTIONS == {"FORCE_USE_FLEX_ATTENTION": True}
