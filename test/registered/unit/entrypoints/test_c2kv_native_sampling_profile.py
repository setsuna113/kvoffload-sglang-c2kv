"""CPU contracts for the native packed generation sampling profiles."""

import ast
import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest


HTTP_SERVER = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/entrypoints/http_server.py"
)


def _load_functions(*names, **bindings):
    tree = ast.parse(HTTP_SERVER.read_text(encoding="utf-8"), filename=str(HTTP_SERVER))
    functions = []
    for name in names:
        node = next(
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        )
        node.decorator_list = []
        functions.append(node)
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "time": time,
        "C2KVNativePackedGenerateRequest": object,
        "Request": object,
        **bindings,
    }
    exec(compile(module, str(HTTP_SERVER), "exec"), namespace)
    return namespace


SAMPLING = _load_functions("_c2kv_native_sampling_params")[
    "_c2kv_native_sampling_params"
]


def _request(profile="greedy-v1", **sampling):
    return SimpleNamespace(
        sampling_profile=profile,
        sampling_params={"max_new_tokens": 1000, **sampling},
        shadow_features=None,
    )


def test_default_greedy_profile_preserves_old_temperature_gate():
    params = SAMPLING(_request())
    assert params == {"max_new_tokens": 1000, "temperature": 0.0}
    with pytest.raises(ValueError, match="requires greedy decoding"):
        SAMPLING(_request(temperature=0.001))


def test_acebench_profile_passes_exact_sampler_without_seed_or_shadow():
    original = {
        "max_new_tokens": 1000,
        "temperature": 0.001,
        "top_p": 1.0,
        "stop_token_ids": [151645],
    }
    request = SimpleNamespace(
        sampling_profile="acebench-agent-v1",
        sampling_params=original,
        shadow_features=None,
    )
    assert SAMPLING(request) == original
    assert request.sampling_params == original


@pytest.mark.parametrize(
    "sampling,shadow,error",
    [
        ({"temperature": 0, "top_p": 1}, None, "temperature=0.001"),
        ({"temperature": 0.001, "top_p": 0.9}, None, "top_p=1"),
        ({"temperature": 0.001}, None, "top_p=1"),
        ({"temperature": 0.001, "top_p": 1, "seed": 0}, None, "seed"),
        (
            {"temperature": 0.001, "top_p": 1, "sampling_seed": 0},
            None,
            "sampling_seed",
        ),
        ({"temperature": 0.001, "top_p": 1, "top_k": 5}, None, "top_k"),
    ],
)
def test_acebench_profile_rejects_other_sampler_or_detector_settings(
    sampling, shadow, error
):
    request = _request("acebench-agent-v1", **sampling)
    request.shadow_features = shadow
    with pytest.raises(ValueError, match=error):
        SAMPLING(request)


def test_capability_advertises_both_named_profiles():
    server_args = SimpleNamespace(
        dtype="float16", kv_cache_dtype="auto", enable_c2kv=True
    )
    model_config = SimpleNamespace(
        dtype="float16",
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=4,
        v_head_dim=4,
        hf_config=SimpleNamespace(pic_enabled=False),
    )
    manager = SimpleNamespace(
        server_args=server_args,
        model_config=model_config,
        model_path="model",
    )
    namespace = _load_functions(
        "_c2kv_native_capability",
        "_c2kv_tool_gist_capability",
        get_bool_env_var=lambda name: False,
        _global_state=SimpleNamespace(tokenizer_manager=manager),
        canonical_model_binding=lambda **kwargs: {
            **kwargs,
            "weight_version": "checkpoint",
        },
        _c2kv_dtype_nbytes=lambda dtype: 2,
        NATIVE_PACKED_CAPABILITY_SCHEMA="c2kv-native-packed-capability-v1",
        C2KV_NATIVE_PACKING_VERSION="history-event-v1",
        C2KV_NATIVE_RAW_LAYOUT_PROFILE="event-native-evidence-v1",
    )
    capability = namespace["_c2kv_native_capability"]()
    assert capability["sampling_profiles"] == ["greedy-v1", "acebench-agent-v1"]
    assert capability["serving_features"] == {
        "raw_prefix_cache": None,
        "background_extras": None,
        "bulk_cache_lookup": None,
        "bulk_first_miss": None,
        "cross_turn_prewarm": None,
        "async_compression": None,
    }
    namespace["get_bool_env_var"] = lambda name: True
    assert namespace["_c2kv_native_capability"]()["serving_features"] == {
        "raw_prefix_cache": "raw-prefix-v1",
        "background_extras": "selected-first-response-barrier-v1",
        "bulk_cache_lookup": "bulk-cache-lookup-v1",
        "bulk_first_miss": "bulk-first-miss-v1",
        "cross_turn_prewarm": "cross-turn-prewarm-v1",
        "async_compression": "nonblocking-history-v1",
    }
    namespace["get_bool_env_var"] = lambda name: name == "C2KV_NATIVE_ASYNC_COMPRESSION"
    features = namespace["_c2kv_native_capability"]()["serving_features"]
    assert features["cross_turn_prewarm"] is None
    assert features["async_compression"] == "nonblocking-history-v1"
    assert features["bulk_first_miss"] is None
    namespace["get_bool_env_var"] = lambda name: True
    server_args.disable_finished_insert = True
    assert namespace["_c2kv_native_capability"]()["serving_features"]["raw_prefix_cache"] is None
    server_args.disable_finished_insert = False
    server_args.speculative_algorithm = "EAGLE"
    assert namespace["_c2kv_native_capability"]()["serving_features"]["raw_prefix_cache"] is None
    server_args.tokenizer_worker_num = 2
    assert namespace["_c2kv_native_capability"]()["serving_features"]["cross_turn_prewarm"] is None
    assert namespace["_c2kv_native_capability"]()["serving_features"]["async_compression"] is None
    server_args.tokenizer_worker_num = 1
    server_args.dp_size = 2
    assert namespace["_c2kv_native_capability"]()["serving_features"]["cross_turn_prewarm"] is None
    assert namespace["_c2kv_native_capability"]()["serving_features"]["async_compression"] is None


@pytest.mark.parametrize("tool_layout", ["prefix_chunk", "anchored_segment", "raw_segment"])
def test_tool_native_full_denominator_needs_client_renderer_count(tool_layout):
    measure = _load_functions("_c2kv_native_whole_full_measurement")[
        "_c2kv_native_whole_full_measurement"
    ]
    tool_chunk = SimpleNamespace(projection_set="tool")
    request = SimpleNamespace(
        paper_whole_full_kv_tokens=None,
        encoder_chunks=[tool_chunk] if tool_layout == "prefix_chunk" else [],
        compression_chunks=[],
        tool_gist_segments=[object()] if tool_layout == "anchored_segment" else [],
        raw_tool_segments=[object()] if tool_layout == "raw_segment" else [],
    )
    plan = SimpleNamespace(logical_input_ids=[1, 2, 3])
    assert measure(request, plan) == (
        None, "unknown_missing_client_native_full_renderer"
    )
    request.paper_whole_full_kv_tokens = 13
    assert measure(request, plan) == (13, "client_native_full_renderer")


@pytest.mark.parametrize("shadow_enabled", [False, True])
@pytest.mark.parametrize("compact_response", [False, True])
def test_endpoint_forwards_acebench_sampling_to_generation_request(
    shadow_enabled, compact_response
):
    captured = {}
    paper_measurement = {
        "duration_ns": 123,
        "metrics": {"gist_generation_duration_ns": 0},
    }
    runtime_stats = {
        "paper_measurement": paper_measurement,
        "c2kv_raw_prefix_cache": {"status": "cached", "hit_tokens": 7},
        "kv_resident_tokens": 9,
    }

    async def generate_request(request, raw_request):
        captured["sampling_params"] = request.sampling_params
        captured["whole_full"] = request.c2kv_paper_whole_full_kv_tokens
        captured["whole_full_source"] = request.c2kv_paper_whole_full_source
        captured["return_hidden_states"] = request.return_hidden_states
        yield {
            "output_ids": [42],
            "text": "answer",
            "meta_info": {
                "output_token_logprobs": [(-0.5, 42)],
                "kv_runtime_stats": runtime_stats,
                **({"hidden_states": [[1.0, 2.0]]} if shadow_enabled else {}),
            },
        }

    manager = SimpleNamespace(generate_request=generate_request)
    plan = SimpleNamespace(
        logical_input_ids=[1, 2],
        selected_handles=[],
        segment_boundaries=[],
        compression_handles=[],
        unique_chunks=[],
        costs={
            "presented_encoder_tokens": 0,
            "gist_tokens": 0,
            "system_tokens": 1,
            "gist_prefix_kv_tokens": 0,
            "raw_workspace_kv_tokens": 1,
            "resident_kv_tokens": 2,
        },
    )
    namespace = _load_functions(
        "_c2kv_native_sampling_params",
        "_c2kv_native_whole_full_measurement",
        "_c2kv_native_background_extras_fallback_reason",
        "_c2kv_native_background_extras_eligible",
        "v1_c2kv_native_generate",
        get_bool_env_var=lambda name: (
            compact_response and name == "C2KV_NATIVE_COMPACT_RESPONSE"
        ),
        _global_state=SimpleNamespace(tokenizer_manager=manager),
        _c2kv_native_capability=lambda: {
            "enabled": True,
            "model_binding": {"pic_enabled": False},
            "shadow_feature_layer": 3,
            "num_hidden_layers": 4,
            "kv_bytes_per_token": 4,
        },
        plan_native_packed_request=lambda **kwargs: plan,
        GenerateReqInput=lambda **kwargs: SimpleNamespace(**kwargs),
        orjson_response=lambda value: value,
        NATIVE_PACKED_RESPONSE_SCHEMA="c2kv-native-packed-response-v1",
        logger=SimpleNamespace(error=lambda *args, **kwargs: None),
        _create_error_response=lambda error: {"error": str(error)},
        float16_roundtrip=lambda values: list(values),
    )
    sampling = {
        "max_new_tokens": 1000,
        "temperature": 0.001,
        "top_p": 1,
        "stop_token_ids": [151645],
    }
    request = SimpleNamespace(
        sampling_profile="acebench-agent-v1",
        sampling_params=sampling,
        shadow_features={"enabled": True, "prefill_layer": -1} if shadow_enabled else None,
        max_extraction_calls=0,
        max_tool_extraction_calls=None,
        encoder_chunks=[],
        compression_chunks=[],
        raw_tool_segments=[],
        tool_gist_segments=[],
        paper_whole_full_kv_tokens=17,
        system_input_ids=[1],
        workspace_input_ids=[2],
        packing_version="history-event-v1",
        raw_layout_profile="event-native-evidence-v1",
        encoding_scope="current",
        compression_ratio=8,
        rid="native-1",
        session_id="session-1",
        generation_id="generation-1",
    )
    response = asyncio.run(
        namespace["v1_c2kv_native_generate"](
            request, SimpleNamespace(headers={})
        )
    )
    assert captured["sampling_params"] == sampling
    assert captured["whole_full"] == 17
    assert captured["whole_full_source"] == "client_native_full_renderer"
    assert captured["return_hidden_states"] is shadow_enabled
    assert response["paper_measurement"] == paper_measurement
    assert response["sglang_runtime"] == (
        {
            "c2kv_raw_prefix_cache": runtime_stats["c2kv_raw_prefix_cache"],
            "kv_resident_tokens": 9,
        }
        if compact_response
        else runtime_stats
    )
    assert response["telemetry"]["generation"] == {
        "outer_request_id": response["outer_request_id"],
        "server_request_id": response["rid"],
        "phase": "c2kv_native:generation",
        **({} if compact_response else {"paper_measurement": paper_measurement}),
    }
    assert "execution_timing" in response["telemetry"]
    assert runtime_stats["paper_measurement"] is paper_measurement
    assert response["sampling_profile"] == "acebench-agent-v1"
    assert response["output_ids"] == [42]
    assert (response["shadow_features"] is not None) is shadow_enabled
    if shadow_enabled:
        assert response["shadow_features"]["prefill"]["hidden"] == [1.0, 2.0]
