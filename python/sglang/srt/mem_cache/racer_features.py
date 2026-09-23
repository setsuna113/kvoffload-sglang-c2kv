"""The native detector feature contract shared by persistent chat backends."""

from sglang.srt.mem_cache.c2kv_native_packed import float16_roundtrip


def validate_shadow_request(request, capability):
    if not request or not request.get("enabled", True):
        return False
    layer = request.get("prefill_layer")
    if type(layer) is not int:
        raise ValueError("RACER_SHADOW_PREFILL_LAYER_REQUIRED")
    normalized = layer if layer >= 0 else capability["num_hidden_layers"] + layer
    if normalized != capability["shadow_feature_layer"]:
        raise ValueError("RACER_SHADOW_LAYER_MISMATCH")
    return True


def shadow_feature_receipt(meta_info, capability, logical_position):
    finish = meta_info.get("finish_reason") or {}
    if finish.get("type") == "abort":
        raise ValueError(f"RACER_GENERATION_ABORTED: {finish.get('message') or 'engine aborted generation'}")
    hidden = meta_info.get("hidden_states") or []
    if not hidden:
        raise ValueError("RACER_SHADOW_PREFILL_HIDDEN_MISSING")
    state = hidden[0]
    if state and isinstance(state[0], list):
        state = state[-1]
    if not isinstance(state, list) or not state:
        raise ValueError("RACER_SHADOW_PREFILL_HIDDEN_MALFORMED")
    return {
        "schema": "event-native-shadow-features-v1", "status": "captured",
        "bindings": {"model": capability["model_binding"], "layer_indexing": "zero_based_decoder_layer_output"},
        "prefill": {
            "status": "captured", "reason": None,
            "layer": capability["shadow_feature_layer"],
            "position": {"kind": "prompt_last", "logical_position": int(logical_position)},
            "readout": "decoder_layer_output", "stored_dtype": "float16",
            "hidden": float16_roundtrip(state),
        },
        "memgen": {"status": "disabled", "layer": None, "hidden": None},
        "capture_errors": [],
    }
