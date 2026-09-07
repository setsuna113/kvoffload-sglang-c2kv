#!/usr/bin/env python3
"""Smoke test for the C2KV serving semantics documented in
c2kv/c2kv_serving_semantics.md. Runs against a live server; no model code.

Checks
  1. /v1/c2kv/extract on a few history docs, records original_seq_len
  2. a compressed chat request: response carries metadata.sglang_runtime
     with the D6 projection provenance (c2kv_query_proj,
     c2kv_query_proj_effective, c2kv_query_proj_source), c2kv_tools_dump,
     and a c2kv_layout whose gist entries sum to the extract-side ledger
     (frame check)
  2b. an unannotated ("full" arm) chat request: the same keys are present,
     the projection reports base/none, and c2kv_layout is an empty list
     rather than an absent key
  3. /v1/c2kv/repair_extract in the messages form: position_start equals
     rendered_prefix_len and equals the ledger position of the target doc
  4. the same chat with the repair KV under each placement; layout entries
     report the placement and the expected positions
  5. an invalid c2kv_repair_placement is refused, and the refusal carries the
     machine-readable C2KV_REPAIR_PLACEMENT_INVALID code

Usage:
  python scripts/c2kv/smoke_c2kv_semantics.py --base-url http://127.0.0.1:35000
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import urllib.error
import urllib.request

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

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
    with OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_allow_error(
    base_url: str, path: str, payload: dict, timeout: int = 600
) -> tuple:
    """POST and return (status, parsed_body_or_raw_text).

    A rejected C2KV request can surface either way and the smoke test must not
    care which: an admission-time error goes through set_finish_with_abort with
    HTTPStatus.BAD_REQUEST and reaches the client as a non-200 body, while an
    injection-time failure is an HTTP 200 whose finish_reason.type is "abort".
    See c2kv/c2kv_serving_semantics.md section 3.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        status = exc.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


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
    # Projection provenance, graph eligibility, and the tool-dump frame must be
    # present on every request of a C2KV-enabled server.
    for key in (
        "c2kv_query_proj",
        "c2kv_query_proj_effective",
        "c2kv_query_proj_source",
        "c2kv_query_proj_decode_verified",
        "c2kv_query_proj_graph_eligible",
        "c2kv_tools_dump",
    ):
        ok &= check(key in runtime, f"compressed request reports {key}")
    ok &= check(
        runtime.get("c2kv_query_proj_effective") in ("base", "gist"),
        f"c2kv_query_proj_effective is base/gist "
        f"(got {runtime.get('c2kv_query_proj_effective')!r})",
    )
    # No message of this request carries c2kv_use_gist_projection, so the flag
    # decides and the effective mode must agree with it.
    ok &= check(
        runtime.get("c2kv_query_proj_source") == "flag",
        f"c2kv_query_proj_source is 'flag' with no message-level override "
        f"(got {runtime.get('c2kv_query_proj_source')!r})",
    )
    ok &= check(
        runtime.get("c2kv_query_proj_effective")
        == ("gist" if runtime.get("c2kv_query_proj") == "gist" else "base"),
        "c2kv_query_proj_effective follows c2kv_query_proj on a flag-decided "
        "request",
    )
    ok &= check(
        runtime.get("c2kv_query_proj_graph_eligible") is True,
        "NPU/CUDA full-graph replay accepts the effective projection mode",
    )
    ok &= check(
        runtime.get("c2kv_query_proj_decode_verified") is True,
        "decode projection is verified through the graph-owned mask",
    )
    ok &= check(
        isinstance(runtime.get("c2kv_layout"), list),
        f"c2kv_layout is a list (got {type(runtime.get('c2kv_layout')).__name__})",
    )
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

    # 2a. Request-level schema override. This field used to be passed through
    # serving_chat via getattr but was absent from ChatCompletionRequest, so
    # pydantic silently discarded it and this branch was unreachable.
    base_override = post(
        args.base_url,
        "/v1/chat/completions",
        {**chat, "c2kv_use_gist_projection": False},
    )
    base_override_rt = (
        (base_override.get("metadata") or {}).get("sglang_runtime") or {}
    )
    ok &= check(
        base_override_rt.get("c2kv_query_proj_effective") == "base"
        and base_override_rt.get("c2kv_query_proj_source") == "request",
        "request-level projection override resolves to base with request provenance",
    )
    ok &= check(
        base_override_rt.get("c2kv_query_proj_graph_eligible") is True
        and base_override_rt.get("c2kv_query_proj_decode_verified") is True,
        "request-level base override remains graph eligible and decode verified",
    )

    # Different request policies may share a scheduler batch. Both must keep
    # their own routing through prefill and decode.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(post, args.base_url, "/v1/chat/completions",
                                   {**chat, "c2kv_use_gist_projection": mode})
                   for mode in (False, True)]
        for mode, future in zip(("base", "gist"), futures):
            mixed_response = future.result()
            mixed_rt = ((mixed_response.get("metadata") or {}).get("sglang_runtime") or {})
            finish = (mixed_response.get("choices") or [{}])[0].get("finish_reason")
            ok &= check(finish in ("stop", "length", "tool_calls")
                        and mixed_rt.get("c2kv_query_proj_effective") == mode
                        and mixed_rt.get("c2kv_query_proj_source") == "request"
                        and mixed_rt.get("c2kv_query_proj_decode_verified") is True,
                        f"concurrent {mode} override completes with its own verified policy")

    conflicting = [dict(message) for message in messages]
    conflicting[1]["c2kv_use_gist_projection"] = False
    conflicting[2]["c2kv_use_gist_projection"] = True
    conflict_status, conflict_body = post_allow_error(
        args.base_url, "/v1/chat/completions", {**chat, "messages": conflicting})
    conflict_finish = ((conflict_body.get("choices") or [{}])[0].get("finish_reason")
                       if isinstance(conflict_body, dict) else None)
    ok &= check((conflict_status != 200 or conflict_finish == "abort")
                and "C2KV_QUERY_PROJECTION_CONFLICT" in json.dumps(conflict_body),
                "conflicting message projection policies are explicitly rejected")

    # 2b. the "full" arm: no C2KV annotation at all. The projection keys are
    # still there, the request never reached the resolver, and the ledger keys
    # are present-and-empty rather than absent (an absent key is ambiguous --
    # it also means "this build has no ledger").
    full_msgs = [{"role": "system", "content": SYSTEM}] + [
        {"role": "user", "content": doc} for doc in HISTORY_DOCS
    ] + [{"role": "user", "content": CURRENT}]
    full_resp = post(args.base_url, "/v1/chat/completions", {**chat, "messages": full_msgs})
    full_rt = (full_resp.get("metadata") or {}).get("sglang_runtime") or {}
    for key in (
        "c2kv_query_proj",
        "c2kv_query_proj_effective",
        "c2kv_query_proj_source",
        "c2kv_query_proj_decode_verified",
        "c2kv_query_proj_graph_eligible",
        "c2kv_tools_dump",
    ):
        ok &= check(key in full_rt, f"full-arm request reports {key}")
    ok &= check(
        full_rt.get("c2kv_query_proj_effective") == "base",
        f"full arm ran base projections "
        f"(got {full_rt.get('c2kv_query_proj_effective')!r})",
    )
    ok &= check(
        full_rt.get("c2kv_query_proj_source") == "none",
        f"full arm made no projection decision "
        f"(got {full_rt.get('c2kv_query_proj_source')!r})",
    )
    ok &= check(
        isinstance(full_rt.get("c2kv_layout"), list)
        and full_rt.get("c2kv_layout") == [],
        f"full arm reports c2kv_layout == [] "
        f"(got {full_rt.get('c2kv_layout')!r})",
    )
    ok &= check(
        full_rt.get("c2kv_gist_seen") is False
        and full_rt.get("c2kv_position_correction") == 0,
        "full arm reports c2kv_gist_seen=false, c2kv_position_correction=0",
    )
    ok &= check(
        "c2kv_injection_error" not in full_rt,
        "full arm reports no c2kv_injection_error",
    )

    # 3. repair extract, messages form, target = doc 1 (index 2 in messages incl. system)
    plain = [{"role": "system", "content": SYSTEM}] + [
        {"role": "user", "content": doc} for doc in HISTORY_DOCS]
    target_index = 2
    # NOTE: raw_kv_position_mode is deliberately NOT sent. The
    # messages/target_index form defaults to "pre_rope"
    # (http_server.v1_c2kv_repair_extract), which is the storage form the
    # append_tail placement in step 4 requires; sending it explicitly here
    # would stop this test from covering the default, i.e. the path every
    # client that omits the field takes (the bench proxy does).
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

    # 5. an unknown placement must be refused, and the refusal must name the
    # machine-readable code. It is rejected at admission today
    # (_build_c2kv_prefill_rounds -> set_finish_with_abort -> HTTP 400); the
    # injection-time backstop (_resolve_c2kv_repair_placement) surfaces the
    # same code as an HTTP 200 abort carrying
    # metadata.finish_message / metadata.sglang_runtime.c2kv_injection_error.
    # Accept either surface, require the code.
    if rep.get("success"):
        bad_msgs = [{"role": "system", "content": SYSTEM}]
        for i, (doc, r) in enumerate(zip(HISTORY_DOCS, records)):
            m = {"role": "user", "content": doc, "c2kv_ratio": args.ratio,
                 "c2kv_key_hash": r["key_hash"]}
            if i == len(HISTORY_DOCS) - 1:
                m["c2kv_repair_key_hashes"] = [rep["key_hash"]]
                m["c2kv_repair_placement"] = "bogus_placement"
            bad_msgs.append(m)
        bad_msgs.append({"role": "user", "content": CURRENT})
        status, body = post_allow_error(
            args.base_url, "/v1/chat/completions", {**chat, "messages": bad_msgs}
        )
        if isinstance(body, dict):
            meta = body.get("metadata") or {}
            rt = meta.get("sglang_runtime") or {}
            first_choice = (body.get("choices") or [{}])[0]
            finish = first_choice.get("finish_reason")
            served_text = (first_choice.get("message") or {}).get("content")
            # Whole body: the code may sit in the 400 error payload, in
            # metadata.finish_message, or in
            # metadata.sglang_runtime.c2kv_injection_error. The request itself
            # never contains the code, so this cannot self-satisfy.
            blob = json.dumps(body)
            surfaces = (
                f"finish_message={meta.get('finish_message')!r} "
                f"c2kv_injection_error={rt.get('c2kv_injection_error')!r}"
            )
        else:
            finish = None
            served_text = None
            blob = str(body)
            surfaces = ""
        ok &= check(
            status != 200 or finish == "abort",
            f"bogus placement is refused (status={status}, finish_reason={finish!r}, "
            f"text={str(served_text)[:60]!r})",
        )
        ok &= check(
            "C2KV_REPAIR_PLACEMENT_INVALID" in blob,
            "bogus placement refusal names C2KV_REPAIR_PLACEMENT_INVALID",
        )
        print(f"  bogus placement: status={status} {surfaces} {blob[:300]!r}")

    # 6. CacheBlend (c2kv_serving_semantics.md section 10): the history docs
    # as ONE multi-message span, one chunk per doc, 16 % recomputed; then a
    # chat request carrying the entry in place of the docs.
    cb_plain = [{"role": "system", "content": SYSTEM}] + [
        {"role": "user", "content": doc} for doc in HISTORY_DOCS]
    cb = post(args.base_url, "/v1/c2kv/repair_extract", {
        "messages": cb_plain, "target_index": 1,
        "target_end_index": len(HISTORY_DOCS), "tools": TOOLS,
        "chat_template_kwargs": kw, "repair_mode": "cacheblend",
        "source_doc_index": 0, "kv_reuse_method": "cacheblend",
        "cacheblend_recomp_ratio": 0.16,
    })
    ok &= check(cb.get("success"), f"cacheblend extract success token_len={cb.get('token_len')} span=[{cb.get('span_start')},{cb.get('span_end')}) prefix={cb.get('rendered_prefix_len')} err={cb.get('error')!r}")
    if cb.get("success"):
        acct = cb.get("cacheblend") or {}
        span_len = int(cb["span_end"]) - int(cb["span_start"])
        ok &= check(cb.get("kv_reuse_method") == "cacheblend",
                    "server echoes kv_reuse_method=cacheblend")
        ok &= check(int(cb.get("token_len") or 0) == span_len,
                    "cacheblend entry keeps the WHOLE span (compute saving, not memory)")
        ok &= check(int(acct.get("chunk_count") or 0) == len(HISTORY_DOCS),
                    f"one chunk per history doc ({acct.get('chunk_count')} == {len(HISTORY_DOCS)})")
        ok &= check(int(acct.get("recomputed_tokens") or 0) == max(1, int(span_len * 0.16)),
                    f"recomputed_tokens {acct.get('recomputed_tokens')} == int(span*0.16)")
        ok &= check(bool(cb.get("already_rotated")), "cacheblend entry stored post-RoPE (rotated)")
        ok &= check(int(cb["position_start"]) == int(cb["rendered_prefix_len"]),
                    "cacheblend position_start == rendered_prefix_len")
        cb_msgs = [{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": "[cacheblend history kv]",
                    "c2kv_repair_only_key_hashes": [cb["key_hash"]],
                    "c2kv_repair_placement": "in_place"},
                   {"role": "user", "content": CURRENT}]
        resp = post(args.base_url, "/v1/chat/completions", {**chat, "messages": cb_msgs})
        rt = (resp.get("metadata") or {}).get("sglang_runtime") or {}
        reps = [e for e in (rt.get("c2kv_layout") or []) if e.get("kind") == "repair"]
        ok &= check(len(reps) == 1 and reps[0].get("placement") == "in_place",
                    "cacheblend: entry injected in_place")
        if reps:
            ok &= check(int(reps[0]["position_start"]) == int(cb["position_start"]),
                        "cacheblend: keeps its absolute position")
        ok &= check("c2kv_injection_error" not in rt, "cacheblend: no c2kv_injection_error")
        text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content")
        print(f"  cacheblend: recomputed={acct.get('recomputed_tokens')}/{span_len} "
              f"dev_max={acct.get('deviation_max')} {str(text)[:120]!r}")

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
