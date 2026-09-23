"""A failed prefill must report its engine error before feature validation."""

import sys

from test_racer_transaction import load

import pytest


def test_shadow_receipt_preserves_engine_abort_reason(monkeypatch):
    packed = load("racer_failure_packed", "mem_cache/c2kv_native_packed.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_native_packed", packed)
    feature = load("racer_failure_features", "mem_cache/racer_features.py")
    meta = {"finish_reason": {"type": "abort", "message":
            "PHYSICAL_HISTORY_KV_EVICTION_EXCEPTION: COMMITKV_REFERENCE_HISTORY_EMPTY"},
            "hidden_states": []}
    with pytest.raises(ValueError, match="RACER_GENERATION_ABORTED:.*COMMITKV_REFERENCE_HISTORY_EMPTY"):
        feature.shadow_feature_receipt(meta, {}, 0)
    with pytest.raises(ValueError, match="RACER_SHADOW_PREFILL_HIDDEN_MISSING"):
        feature.shadow_feature_receipt({"hidden_states": []}, {}, 0)
