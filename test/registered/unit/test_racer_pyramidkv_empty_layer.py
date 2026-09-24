"""PyramidKV selection when a layer has no candidate left after RACER edits.

SZ BFCL Long b256 (racer_v2 PyramidKV c1_v2_verified, task 81): the turn-2
regeneration replaced a recovered source that PyramidKV kept in its recent
window, which emptied most headwise reference layers (32 of 23040 slots
remained, in 3 layers).  The next draft's ordinary history was entirely its
initial S0 sources, so the first captured layer had zero candidates and
selection raised "PyramidKV scores must have shape [Hkv, history]"; an empty
later layer already kept zero tokens.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import List

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python" / "sglang" / "srt"


def _load(relative):
    """Import one engine module; without the serving dependencies, load its file.

    The fallback registers the file under its package name, so the lazy
    in-function imports of production code resolve to the same module.
    """
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


try:
    importlib.import_module("sglang.srt.observability.paper_telemetry")
except ImportError:
    sys.modules.setdefault("sglang.srt.observability", SimpleNamespace(
        paper_telemetry=SimpleNamespace(sample=lambda *args, **kwargs: None)))
reference = _load("mem_cache/history_kv_reference.py")


def _build_method():
    path = SRT / "managers" / "scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "Scheduler"
                for item in cls.body
                if isinstance(item, ast.FunctionDef) and item.name == "_build_pyramidkv_reference_state")
    namespace = {"List": List, "math": math, "torch": torch, "Req": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_build_pyramidkv_reference_state"]


def _layer(positions, heads=2):
    positions = torch.tensor([list(positions)] * heads, dtype=torch.long)
    return reference.ReferenceLayerKV(key=positions.float().unsqueeze(-1),
                                      value=positions.float().unsqueeze(-1) + 1000,
                                      positions=positions)


def test_an_empty_first_layer_keeps_nothing_like_an_empty_later_layer():
    rich = torch.arange(12, dtype=torch.float32).view(2, 6)
    empty = rich[:, :0]
    with pytest.raises(ValueError, match=r"shape \[Hkv, history\]"):
        reference.select_pyramidkv_headwise([empty, empty], target_tokens=4)

    later, later_meta = reference.select_pyramidkv_headwise([rich, empty], target_tokens=4,
                                                            capacity_history_tokens=6)
    first, first_meta = reference.select_pyramidkv_headwise([empty, rich], target_tokens=4,
                                                            capacity_history_tokens=6)
    assert later[1].shape == first[0].shape == (2, 0)
    assert later_meta["per_layer_budget_tokens"][1] == first_meta["per_layer_budget_tokens"][0] == 0
    assert first[1].shape[1] == first_meta["per_layer_budget_tokens"][1] > 0


def test_draft_after_a_source_emptied_regeneration_keeps_the_surviving_layers():
    # Layers 1 and 2 kept one and two columns after the regeneration; layer 0
    # kept none.  The draft's three ordinary tokens are all initial S0 sources.
    existing = reference.ReferenceHistoryKVState(
        method="pyramidkv", layers={1: _layer([10]), 2: _layer([10, 11])}, expected_layer_ids=(1, 2))
    keys = torch.arange(40, dtype=torch.float32).view(20, 2, 1)
    pool = SimpleNamespace(get_kv_buffer=lambda layer: (keys, keys + 1000))
    scheduler = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(20, dtype=torch.long).view(1, -1)),
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool))
    scheduler._build_pyramidkv_reference_state = MethodType(_build_method(), scheduler)
    req = SimpleNamespace(req_pool_idx=0, history_kv_reference_state=existing,
                          history_kv_resident_positions=list(range(20)))
    config = {"history_start": 12, "history_end": 15, "target_tokens": 28,
              "racer_excluded_history_indices": [0, 1, 2]}
    scores = {"layer_ids": [0, 1, 2],
              "headwise_layers": [torch.ones(2, 3), torch.ones(2, 4), torch.ones(2, 5)]}
    state = scheduler._build_pyramidkv_reference_state(req, config, scores)
    assert state.layers[0].positions.shape == (2, 0)
    assert state.layers[1].positions.tolist() == [[10], [10]]
    assert state.layers[2].positions.tolist() == [[10, 11], [10, 11]]
    assert not any(bool(((layer.positions >= 12) & (layer.positions < 15)).any())
                   for layer in state.layers.values())
