"""Explicit opt-in real CUDA smoke; never collected as an automatic GPU test.

Run against an already authorized server with --disable-overlap-schedule:
  python test/srt/test_racer_cuda_smoke.py --base-url URL --model MODEL --output FILE
Each method uses one session and three bounded generations. This script records
runtime receipts and source identity, not benchmark quality results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=120) as response:
        body = response.read()
        return json.loads(body) if body else None


def run_method(args, method):
    session = "racer-smoke-" + uuid.uuid4().hex
    archived = "Observation: the source record confirms code 7319. " + "Audit trail remains available. " * 48
    source = [
        {"role": "system", "content": "Use source records and answer briefly."},
        {"role": "assistant", "content": "lookup(record_id=7)"},
        {"role": "user", "content": archived},
        {"role": "user", "content": "Which code was confirmed? Explain with the record."},
    ]
    messages = list(source)
    outcomes = []
    opened = request(args.base_url, "/open_session", {"capacity_of_str_len": 0, "session_id": session, "streaming": True, "timeout": 180})
    if opened != session:
        raise RuntimeError("Server did not open the requested session")
    try:
        for index, phase in enumerate(("draft", "regenerate", "draft")):
            tx = {"decision_id": "decision-0" if index < 2 else "decision-1", "phase": phase}
            if index:
                tx["resolution"] = "discard"
            if index == 1:
                messages.extend([{"role": "assistant", "content": ""}, {"role": "user", "content": "Recovered source record:\n" + archived}])
            elif index == 2:
                # A deterministic correction is committed; the speculative
                # generation is discarded and cannot become an executed event.
                messages.extend([{"role": "assistant", "content": "The confirmed code is 7319."}, {"role": "user", "content": "Repeat only the code."}])
            persistent = {"enabled": True, "session_id": session, "transaction": tx, "history_budget_tokens": 1024}
            if index == 1:
                persistent["recovery_append"] = {"enabled": True, "source_message_indices": [2]}
            config = {"method": method, "target_tokens": 128, "history_start_message_count": 1,
                      "history_message_count": 3, "history_kv_recent_window": 16,
                      "history_kv_kernel_size": 3, "history_kv_pooling": "avgpool",
                      "history_kv_h2o_recent_fraction": 0.25, "persistent_session": True}
            hint = {"persistent_history_session": persistent, "history_kv_eviction": config,
                    "history_kv_method": method, "history_kv_backend": "reference_attention" if method in {"commitkv", "pyramidkv"} else "physical_eviction",
                    "history_kv_event_messages": [{"message_index": i, "role": message["role"], "phase": "act" if i == 1 else "tool" if i == 2 else "others"} for i, message in enumerate(messages)]}
            if method in {"commitkv", "pyramidkv"}:
                hint["history_kv_reference_config"] = {"method": method, "target_tokens": 128}
                if method == "commitkv":
                    hint["history_kv_reference_config"]["commitkv"] = {
                        "checkpoint_interval": 4, "window_size": 2, "measurement_layer_id": 0}
            if args.shadow_layer is not None:
                hint["shadow_features"] = {"enabled": True, "prefill_layer": args.shadow_layer}
            payload = {"model": args.model, "messages": messages, "temperature": 0, "max_tokens": 16, "min_tokens": 8,
                       "stream": False, "logprobs": True, "c2kv_use_gist_projection": False,
                       "session_params": {"id": session}, "c2kv_kv_memory_hint": hint,
                       "chat_template_kwargs": {"enable_thinking": False}}
            started = time.monotonic()
            result = request(args.base_url, "/v1/chat/completions", payload)
            metadata = result.get("metadata") or {}
            report = metadata.get("kv_memory_report") or {}
            generated = metadata.get("racer_generation") or {}
            accounting = generated.get("accounting") or {}
            assert type(accounting.get("resident_prompt_tokens")) is int, result
            assert accounting["history_and_evidence_tokens"] <= 1024, result
            assert generated["output_token_ids"] and len(generated["output_token_ids"]) == len(generated["output_token_logprobs"]), result
            assert report["history_kv_lifecycle"]["full_history_reprefill_performed"] is False, result
            assert result["choices"][0]["finish_reason"] in {"stop", "length", "tool_calls"}, result
            if index == 1:
                assert report["racer_previous_resolution"]["resolution"] == "discard", result
                assert report["racer_source_replacement"]["source_spans"], result
                assert report["racer_native_evidence_tokens"] > 0, result
            if index == 2:
                assert report["racer_evidence_expiry"]["expired_native_tokens"] > 0, result
                assert accounting["native_evidence_tokens"] == 0, result
            if args.shadow_layer is not None:
                assert generated["shadow_features"]["prefill"]["status"] == "captured", result
            outcomes.append({"phase": phase, "wall_seconds": time.monotonic() - started, "response": result})
    finally:
        request(args.base_url, "/close_session", {"session_id": session})
    return {"method": method, "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(), "generations": outcomes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shadow-layer", type=int)
    parser.add_argument("--methods", nargs="+", choices=["commitkv", "h2o", "snapkv_persistent", "pyramidkv", "streamingllm"], default=["commitkv", "h2o", "snapkv_persistent", "pyramidkv", "streamingllm"])
    args = parser.parse_args()
    model_info = request(args.base_url, "/get_model_info")
    capability = (model_info.get("c2kv_native_packed") or {}).get("racer_persistent") or {}
    if capability.get("schema") != "racer-persistent-transaction-v1" or capability.get("device") != "cuda" or capability.get("overlap_disabled") is not True:
        raise RuntimeError("A CUDA RACER engine with overlap disabled is required: " + json.dumps(capability))
    if args.shadow_layer is None:
        args.shadow_layer = (model_info.get("c2kv_native_packed") or {}).get("shadow_feature_layer")
    if type(args.shadow_layer) is not int:
        raise RuntimeError("CUDA smoke requires a server configured with --c2kv-shadow-feature-layer")
    result = {"schema": "racer-cuda-smoke-v1", "model_info": model_info, "results": []}
    try:
        for method in args.methods:
            result["results"].append(run_method(args, method))
    except Exception as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
