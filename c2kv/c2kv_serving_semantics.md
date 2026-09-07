# C2KV serving contract

This document describes the current public contract of the C2KV SGLang fork.
It covers gist extraction and reuse, raw repair KV, history-KV baselines, and
CacheBlend. It is intended for clients that call the server directly or through
the `c2kv` benchmark harness.

## Server setup

Start SGLang with a C2KV-compatible checkpoint and `--enable-c2kv`:

```bash
python -m sglang.launch_server \
  --model-path /path/to/checkpoint \
  --served-model-name c2kv-agent \
  --enable-c2kv \
  --c2kv-query-proj base \
  --c2kv-gist-param qkv \
  --c2kv-tools-dump full
```

The C2KV-specific flags are:

| Flag | Default | Contract |
|---|---:|---|
| `--enable-c2kv` | off | Enables the C2KV pool, endpoints, and chat-message annotations. |
| `--c2kv-gist-type` | `dynamic-interleave` | Gist extraction layout expected by the checkpoint. |
| `--c2kv-gist-param` | `qkv` | Lowercase non-empty subset of `q`, `k`, and `v` with learned gist projections. Mixed case is rejected. |
| `--c2kv-pool-fraction` | `0.01` | Fraction of the model KV pool reserved for stored C2KV entries. |
| `--c2kv-max-tokens` | `65536` | Upper bound for the C2KV pool. |
| `--c2kv-query-proj` | `base` | Default projection for ordinary tokens after injected C2KV KV. `gist` is an explicit alternate mode. |
| `--c2kv-tools-dump` | `full` | Tool-schema serialization used by chat and extraction. `exclude_unset` is also accepted. All clients in one comparison must use the same value. |

Requests without C2KV annotations follow the ordinary SGLang path. C2KV chat
requests must contain one generation at a time; batched request shapes are
rejected because per-item C2KV fields cannot be preserved by the generic batch
container.

## Gist extraction and reuse

`POST /v1/c2kv/extract` stores gist KV for one text block:

```json
{
  "text": "the history block",
  "role": "user",
  "compression_ratio": 4,
  "tools": [],
  "chat_template_kwargs": {}
}
```

The response contains `key_hash`, `gist_len`, `original_seq_len`, `success`,
and `error`. `original_seq_len` is the number of tokens produced by the
server's rendering of this extraction request. It is also the logical span used
for later RoPE positions. A client must render raw and compressed history with
the same role, tools, `chat_template_kwargs`, and `--c2kv-tools-dump` regime.

Attach the returned key to the corresponding message in a normal
`POST /v1/chat/completions` request:

```json
{
  "model": "c2kv-agent",
  "messages": [
    {
      "role": "user",
      "content": "the history block",
      "c2kv_key_hash": "<key_hash>"
    },
    {"role": "user", "content": "the current request"}
  ]
}
```

The server replaces the annotated message's ordinary history KV with the
stored gist KV while advancing the logical position cursor by
`original_seq_len`. A missing or evicted key fails the C2KV injection instead
of silently serving the uncompressed prompt.

## Query projection

The server resolves one projection policy per request:

1. Request-level `c2kv_use_gist_projection` wins when explicitly present.
2. Otherwise, explicit values on annotated messages must agree.
3. Otherwise, `--c2kv-query-proj` supplies the default.

`null` means unset. Conflicting message values fail with
`C2KV_QUERY_PROJECTION_CONFLICT`.

`base` keeps ordinary query/current-turn/decode tokens on the base QKV
projection. `gist` swaps only the parts enabled by `--c2kv-gist-param` to the
learned gist projection. Gist extraction itself still uses the learned gist
projection for gist tokens, and raw repair extraction always uses base
projections.

CUDA and NPU full-graph replay carry the dynamic projection mask in graph-owned
buffers. CPU and piecewise graph runners reject a gist-projection batch from
their graph path and let it run eagerly.

## Raw repair KV

`POST /v1/c2kv/repair_extract` stores base-projection KV for a history span.
The preferred request form supplies the exact chat messages and lets the server
derive token boundaries from its own chat template:

```json
{
  "messages": [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "history block"}
  ],
  "target_index": 1,
  "tools": [],
  "repair_mode": "d_corr",
  "raw_kv_position_mode": "pre_rope"
}
```

The endpoint also accepts `input_ids` plus `[span_start, span_end)`, or `text`
plus an optional `role`. `repair_position_ids` overrides the selected span's
positions. `position_offset` applies to the `input_ids` and `text` forms.

The response contains the stored `key_hash`, token and position bounds,
`repair_mode`, `extract_source`, `already_rotated`, method-specific metadata,
and `success`/`error`.

Attach repair keys with either `c2kv_repair_key_hashes` or
`c2kv_repair_only_key_hashes` on the relevant chat message. The latter injects
the repairs without also injecting that message's gist. An explicit
`c2kv_repair_placement` selects placement:

| Placement | Meaning | Storage requirement |
|---|---|---|
| `in_place` | The raw span stands in for the compressed history and the query continues at the span's original logical end. | Post-RoPE or pre-RoPE. |
| `append_keep_ledger` | The raw span is appended physically while keeping its original positions; the existing gist remains and the query's logical position does not advance. | Post-RoPE or pre-RoPE. |
| `append_tail` | The raw span is re-rotated to fresh positions at the logical tail and advances the cursor. | Requires `raw_kv_position_mode="pre_rope"`. |

When placement is omitted, `history_kv_*`, `cacheblend*`, replace, recompute,
and masked-repair modes default to `in_place`; other legacy repair modes default
to `append_keep_ledger`.

The single-message `messages` form defaults to pre-RoPE storage unless
CacheBlend is requested. A multi-message span and CacheBlend always use rotated
storage at their original positions. `extract_source="serving_cache"` also
returns rotated KV. Rotated entries cannot be used with `append_tail`.

## History-KV baselines

Set exactly one of `history_kv_target_tokens` or
`history_kv_retention_ratio` on `/v1/c2kv/repair_extract`. The selected count is
clamped to `[1, span_length]`. Supported method names are:

| `history_kv_method` | Current implementation |
|---|---|
| `streamingllm` | Keeps up to four attention-sink tokens plus the most recent suffix, with exactly the requested total budget. |
| `h2o` | Accumulates causal prefill attention over all query tokens, sums GQA query groups into native KV heads, and keeps a heavy-hitter/recent split independently for every layer and KV head. |
| `snapkv_persistent` (`snapkv`) | Uses the configured observation window and pooling kernel, then keeps pooled-score winners plus recent tokens independently for every layer and KV head. |
| `snapkv_refresh` | Currently uses the same prefill-boundary headwise selector as `snapkv_persistent`; it does not implement online decode refresh. |
| `pyramidkv` (`pyramid`) | Shared-page-table approximation: constructs a layer-wise budget funnel and stores the union of selected tokens. The union may exceed the nominal target. |

Optional controls are `history_kv_recent_window` (default `64`),
`history_kv_kernel_size` (default `5`), `history_kv_pooling` (`avgpool` or
`maxpool`), and `history_kv_h2o_recent_fraction` (default `0.5`).

H2O and SnapKV select different source positions per layer and native KV head.
They therefore require `raw_kv_position_mode="rotated"`; one shared pre-RoPE
position vector cannot represent their source phases. Their response sets
`selected_relative_indices` to `null` and reports bounded per-layer/head
previews under `history_selection_metadata`. Check the following values before
labelling a run:

- `history_kv_method` echoes the normalized method name.
- `selected_token_count` equals the per-head target for H2O/SnapKV.
- `history_selection_metadata.per_head_selection` is `true`.
- `algorithm_version` is `h2o_prefill_gqa_v1` or
  `snapkv_gqa_headwise_v1`.

These are history-boundary adaptations of the cited algorithms. H2O does not
perform online decode-score updates, and PyramidKV is explicitly marked
`official_algorithm_implemented=false`.

## CacheBlend

CacheBlend uses the same repair endpoint with
`kv_reuse_method="cacheblend"`. It is mutually exclusive with
`history_kv_method`.

```json
{
  "messages": [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "history block one"},
    {"role": "assistant", "content": "history block two"}
  ],
  "target_index": 1,
  "target_end_index": 2,
  "kv_reuse_method": "cacheblend",
  "cacheblend_recomp_ratio": 0.16,
  "cacheblend_check_layer": 1,
  "cacheblend_metric": "v",
  "cacheblend_mask": "causal"
}
```

Each chunk is first prefetched independently. At `cacheblend_check_layer`, the
server ranks span tokens by K or V deviation from the in-context forward and
recomputes the requested fraction on later layers. `cacheblend_recomp_ratio=1`
is the dense-prefill equivalence endpoint; `0` keeps the rotated standalone
chunk cache after the check layer. CacheBlend keeps KV for the whole span, so
it reduces recomputation rather than KV residency.

Chunk boundaries resolve in this order:

1. Explicit `cacheblend_chunk_bounds`.
2. Explicit `cacheblend_chunk_tokens` fixed grid.
3. One chunk per message for a multi-message target span.
4. One chunk for the whole span.

CacheBlend stores post-RoPE KV at the span's original positions and must be
injected with `in_place`. The response must echo `kv_reuse_method="cacheblend"`
and includes `cacheblend.chunk_bounds`, `recomputed_tokens`,
`recomputed_relative_indices`, `effective_recomp_ratio`, deviation statistics,
and the effective configuration.

## Runtime verification

Non-streaming chat responses expose `metadata.sglang_runtime`. Clients should
check at least:

- `c2kv_tools_dump`, `c2kv_query_proj`, `c2kv_query_proj_effective`, and
  `c2kv_query_proj_source`;
- `c2kv_layout`, `c2kv_position_correction`, and `c2kv_gist_seen`;
- `c2kv_injection_error` when the request aborts;
- `kv_memory_report` for server-measured active history KV counters.

The layout is empty on a C2KV-enabled request that injected no stored entries.
Each gist or repair record includes its physical and logical placement so a
client can reject a run whose requested method was ignored.

Run the CPU contracts without starting a model:

```bash
python -m pytest -q \
  test/registered/unit/test_history_kv_selection.py \
  test/registered/unit/test_c2kv_kv_accounting.py \
  test/registered/unit/test_c2kv_cacheblend.py \
  test/registered/unit/test_c2kv_serving_contract.py \
  test/registered/unit/test_c2kv_pool.py \
  test/registered/unit/model_executor/test_c2kv_graph_projection.py
```

With a live server, run the end-to-end C2KV, repair-placement, projection, and
CacheBlend checks:

```bash
python scripts/c2kv/smoke_c2kv_semantics.py \
  --base-url http://127.0.0.1:30000 \
  --model c2kv-agent
```
