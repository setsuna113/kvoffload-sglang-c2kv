#!/usr/bin/env python3
"""Smoke test for the C2KV serving semantics documented in
c2kv/c2kv_serving_semantics.md. Runs against a live server; no model code.

Checks
  1. /v1/c2kv/extract on a few history docs, records original_seq_len
  2. a compressed chat request: response carries metadata.sglang_runtime
     with c2kv_query_proj and a c2kv_layout whose gist entries sum to the
     extract-side ledger (frame check)
  3. /v1/c2kv/repair_extract in the messages form: position_start equals
     rendered_prefix_len and equals the ledger position of the target doc
  4. the same chat with the repair KV under each placement; layout entries
     report the placement and the expected positions

Usage:
  python scripts/c2kv/smoke_c2kv_semantics.py --base-url http://127.0.0.1:35000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

SYSTEM = "You are a helpful assistant."
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather of a city on a date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["city", "date"],
            },
        },
    }
]
# turn-doc rendering used by the c2kv repo (train_data_multiturn._agent_history_turn_docs)
HISTORY_DOCS = [
    "Previous turn\n[User query]\nWhat is the weather in Shanghai tomorrow?\n"
    "[Assistant output]\nAction:\n<tool_call>{\"name\":\"get_weather\","
    "\"arguments\":{\"city\":\"Shanghai\",\"date\":\"tomorrow\"}}</tool_call>",
    "Previous turn\n[User query]\n{\"temp\": 26, \"cond\": \"sunny\"}\n"
    "[Assistant output]\nTomorrow in Shanghai: sunny, 26C.",
]
CURRENT = "And the day after tomorrow?"


def post(base_url: str, path: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check(cond: bool, msg: str) -> bool:
    print(("PASS " if cond else "FAIL ") + msg)
    return cond


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="c2kv-agent")
    ap.add_argument("--ratio", type=int, default=8)
    args = ap.parse_args()
    ok = True
    kw = {"enable_thinking": False}

    # 1. extract
    records = []
    for doc in HISTORY_DOCS:
        r = post(args.base_url, "/v1/c2kv/extract", {
            "text": doc, "compression_ratio": args.ratio, "role": "user",
            "chat_template_kwargs": kw,
        })
        ok &= check(r.get("success"), f"extract success gist_len={r.get('gist_len')} original_seq_len={r.get('original_seq_len')}")
        records.append(r)
    ledger = [int(r["original_seq_len"]) for r in records]

    # 2. compressed chat
    messages = [{"role": "system", "content": SYSTEM}]
    for doc, r in zip(HISTORY_DOCS, records):
        messages.append({"role": "user", "content": doc, "c2kv_key_hash": r["key_hash"], "c2kv_ratio": args.ratio})
    messages.append({"role": "user", "content": CURRENT})
    chat = {"model": args.model, "messages": messages, "tools": TOOLS,
            "temperature": 0, "max_tokens": 64, "chat_template_kwargs": kw}
    resp = post(args.base_url, "/v1/chat/completions", chat)
    runtime = (resp.get("metadata") or {}).get("sglang_runtime") or {}
    print("sglang_runtime:", json.dumps(runtime, indent=1)[:2000])
    ok &= check("c2kv_query_proj" in runtime, "response echoes c2kv_query_proj")
    layout = runtime.get("c2kv_layout") or []
    gists = [e for e in layout if e.get("kind") == "gist"]
    ok &= check(len(gists) == len(HISTORY_DOCS), f"{len(gists)} gist injections reported")
    if gists:
        ok &= check([int(g["original_seq_len"]) for g in gists] == ledger,
                    "gist original_seq_len ledger matches extract responses")
        # consecutive gists: cursor advances by original_seq_len
        for a, b in zip(gists, gists[1:]):
            ok &= check(int(b["position_cursor"]) == int(a["position_cursor"]) + int(a["original_seq_len"]),
                        "position_cursor advances by original_seq_len (frame consistent)")
    ok &= check(bool(runtime.get("c2kv_gist_seen")), "c2kv_gist_seen is True after gist injection")

    # 3. repair extract, messages form, target = doc 1 (index 2 in messages incl. system)
    plain = [{"role": "system", "content": SYSTEM}] + [
        {"role": "user", "content": doc} for doc in HISTORY_DOCS]
    target_index = 2
    rep = post(args.base_url, "/v1/c2kv/repair_extract", {
        "messages": plain, "target_index": target_index, "tools": TOOLS,
        "chat_template_kwargs": kw, "repair_mode": "d_corr",
        "source_doc_index": target_index - 1,
    })
    ok &= check(rep.get("success"), f"repair_extract(messages) success token_len={rep.get('token_len')} span=[{rep.get('span_start')},{rep.get('span_end')}) prefix={rep.get('rendered_prefix_len')}")
    if rep.get("success"):
        ok &= check(int(rep["position_start"]) == int(rep["rendered_prefix_len"]),
                    "position_start == rendered_prefix_len")
        ok &= check(not rep.get("already_rotated"), "model_prefill repair KV stored pre-RoPE")
        if gists:
            expected = int(gists[target_index - 1]["position_cursor"])
            ok &= check(int(rep["position_start"]) == expected,
                        f"repair position_start {rep['position_start']} == gist ledger position {expected} of the same doc")

    # 4. placements
    if rep.get("success"):
        for placement in ("append_keep_ledger", "append_tail", "in_place"):
            msgs = [{"role": "system", "content": SYSTEM}]
            for i, (doc, r) in enumerate(zip(HISTORY_DOCS, records)):
                m = {"role": "user", "content": doc, "c2kv_ratio": args.ratio}
                if placement == "in_place" and i == target_index - 1:
                    m["c2kv_repair_only_key_hashes"] = [rep["key_hash"]]
                    m["c2kv_repair_placement"] = placement
                else:
                    m["c2kv_key_hash"] = r["key_hash"]
                    if i == len(HISTORY_DOCS) - 1 and placement != "in_place":
                        m["c2kv_repair_key_hashes"] = [rep["key_hash"]]
                        m["c2kv_repair_placement"] = placement
                msgs.append(m)
            msgs.append({"role": "user", "content": CURRENT})
            resp = post(args.base_url, "/v1/chat/completions", {**chat, "messages": msgs})
            rt = (resp.get("metadata") or {}).get("sglang_runtime") or {}
            reps = [e for e in (rt.get("c2kv_layout") or []) if e.get("kind") == "repair"]
            ok &= check(len(reps) == 1 and reps[0].get("placement") == placement,
                        f"{placement}: repair injected with placement reported")
            if reps:
                e = reps[0]
                if placement == "append_tail":
                    ok &= check(int(e["position_start"]) == int(e["logical_before"]),
                                "append_tail: rotated to the logical tail position")
                else:
                    ok &= check(int(e["position_start"]) == int(rep["position_start"]),
                                f"{placement}: keeps original absolute position")
            text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
            print(f"  {placement}: {str(text)[:120]!r}")

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
