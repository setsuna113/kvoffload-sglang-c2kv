# C2KV serving semantics: paper vs. training checkpoint vs. this server

Read this before touching anything under `python/sglang/srt/**/c2kv*`,
`models/qwen3.py` (gist / repair paths) or `scheduler.py` (segment injection),
and before judging a number produced by this server. Three different things
are called "C2KV" in this project and they do **not** agree with each other:

| | what it is | where |
|---|---|---|
| **paper** | arXiv 2607.17715, "C²KV: Compressed and Composable KV Cache Reuse" | text only |
| **training checkpoint** | `qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088` and every other checkpoint produced by `python/train/*` of the `c2kv` repo | `c2kv` repo: `python/models/qwen3/modeling_qwen3.py`, `python/models/gist_utils.py`, `python/train/trainer.py`, `python/train/train_data_multiturn.py` |
| **this server** | the SGLang fork that serves those checkpoints | this repo |

The rule for every decision in this file: **the checkpoint is the ground
truth**. A checkpoint is defined by how it was trained. When the paper text and
the training code disagree, serving must follow the training code, otherwise
the served model is not the model that was trained. When you find such a
disagreement, do not "fix" the server towards the paper without an A/B
number; record it here instead.

Do not conclude from "the server matches the paper" that the checkpoint is
wrong, and do not conclude from "the server matches the training code" that
the paper is wrong. Both statements have been made in this project and both
were naive.

---

## 1. Projections: who uses `gist_{q,k,v}_proj`

Per layer the base model has `W_q, W_k, W_v`. C2KV adds a second set
`W_q^g, W_k^g, W_v^g` (`gist_q_proj` etc. in HF, fused `gist_qkv_proj` here),
initialised as a copy of the base weights (`gist_utils.py:906-927`) and the
only attention weights that are trained (`--only_train_gist True`).

| token class | paper (§3.2.2) | training code | this server |
|---|---|---|---|
| document tokens during extraction | base | base | base (`forward_with_gist`) |
| gist tokens during extraction | gist | gist | gist (`forward_with_gist`, `gist_qkv_proj`) |
| system prefix (before any gist) | base | base (`trainer.py:_build_system_kv`, prefilled without `use_gist`) | base |
| **query / current turn / decoded tokens (after gist KV)** | base ("applied only to the original document tokens") | **gist**: `modeling_qwen3.py:673` sets `use_gist=True` for the whole main forward whenever `context_input_ids` (gists) are present, and `:242-246` then routes every token's q/k/v through `gist_*_proj` for the parts listed in `gist_param` (`qkv` in every training script) | `--c2kv-query-proj gist` (default): gist for every token that comes after the first injected gist KV of its request; `--c2kv-query-proj base`: base (behaviour before 2026-09-02) |
| raw repair KV (`/v1/c2kv/repair_extract`) | n/a (paper §3.3.2 "original-token invariance": raw KV must equal the frozen base model's) | HF harness computes it with base projections | base (`generate_raw_repair_kv` passes `forward_batch=None`) |

So the checkpoint was trained with a query that reads gist K through
`W_q^g` and produces its own K/V through `W_k^g, W_v^g`. Serving with base
projections for the query (the pre-2026-09 behaviour, still available as
`--c2kv-query-proj base`) is a train/serve mismatch. Because `W^g` starts as a
copy of `W` and is trained with LR 5e-5 for O(10^3) steps, the two differ by
a small delta; the size of the effect on downstream numbers is **not
established** (2026-09-02: no A/B run yet). Any number produced by a server
must therefore record which mode produced it: the mode is echoed in every
chat response under `metadata.sglang_runtime.c2kv_query_proj`.

Implementation: `Req.c2kv_gist_seen` (set in
`Scheduler._inject_c2kv_gist_segment`) → `ModelWorkerBatch.c2kv_gist_seen` →
`ForwardBatch.c2kv_gist_proj_mask` (per token) →
`Qwen3Attention._c2kv_project_qkv`. Repair-only injections do not set
`c2kv_gist_seen`: a request whose history is entirely raw repair KV (no gist)
is served with base projections, like the Full arm.

Known limitation: the mask is built per forward from request state; it is not
part of CUDA/NPU graph capture. Serve C2KV with `--disable-cuda-graph` (the
c2kv position correction has the same limitation).

## 2. Position frames: what "original_seq_len" means

The paper (§3.2.4): stored gist KV are pre-RoPE; at reuse they get "new
positional embeddings according to their placement in the final input
sequence, where positional offsets are accumulated across concatenated
segments and each compressed KV segment is treated as occupying the same
effective span as its original uncompressed tokens".

This server implements exactly that with `Req.c2kv_position_correction`
(`scheduler.py`, gist injection): gist absolute position = `kv_start +
correction + local_gist_pos`, then `correction += original_seq_len -
gist_len`, so every later token sits at
`physical_index + correction` = its position in the uncompressed prompt.

**`original_seq_len` is the length of whatever text the client sent to
`/v1/c2kv/extract`, tokenized as one message with the given role.** The
server cannot know whether that text is the same rendering the client uses
for the raw version of the same content. Two clients in this project render
history differently:

- the `c2kv` repo's training data and its D-line harness render one *turn*
  (`Previous turn\n[User query]…\n[Assistant output]…`) per doc
  (`train_data_multiturn.py:_agent_history_turn_docs`), and the raw KV of a
  doc is computed from the same doc text, so gist frame == raw frame;
- `Tracy-ZYH/bfcl-c2kv` renders one *assistant+tool* unit per doc wrapped as
  `Completed history unit:\n<history_message role=…>` for the gist, but
  slices the raw repair KV from the native chat-template rendering of the
  original messages. Those two renderings differ by 14–27 tokens per unit
  (measured with the Qwen3 tokenizer), so its gists live in one position
  frame and its repair KV in another.

The server does not and cannot reconcile the two. What it does since
2026-09-02 is **report** every injection: `metadata.sglang_runtime.c2kv_layout`
lists, per gist, `position_cursor`, `original_seq_len`, `gist_len`, and per
repair entry `position_start`, `position_end`, `logical_before`,
`placement`. A client that keeps its own ledger must assert
`Σ original_seq_len == Σ raw lengths` from this list; a growing gap is the
frame bug above.

The tool prologue is part of the frame. `chat_template_tools_dump`
(`protocol.py`) serializes tools with `exclude_unset=True` so the rendered
prologue equals the client's tool JSON; before 2026-09-02 `model_dump()` also
emitted pydantic defaults (`"strict": false`, +4 tokens per tool), a constant
offset a client tokenizing the same JSON itself could not see.

## 3. Repair KV: source and placement

`/v1/c2kv/repair_extract` stores the raw (base-projection) K/V of a span of
tokens so it can be injected next to, or instead of, gist KV.

**Source.** Three forms, in decreasing fidelity:

1. `messages` + `target_index` (+ `tools`, `chat_template_kwargs`): the server
   renders `messages[:target_index+1]` like a chat request and captures the
   K/V of the target message inside that context. This is "the KV the base
   model would have for this message in this prompt" and is the form the
   `c2kv` repo's D-line results (`corr`, `keepG`, `erratum_tail`) are about.
   Response carries `span_start`/`span_end`/`rendered_prefix_len`.
2. `input_ids` + `span_start`/`span_end` (+ `position_offset`): the caller
   supplies the full context tokens itself (`bfcl-c2kv` does this).
3. `text` + `role`: the message alone, no context. Its K/V at layer > 0 differ
   from the in-context K/V (that is the "KV deviation" the paper argues
   against for training-free reuse). Use it only when you mean it.

`extract_source=serving_cache` reads K/V back from the radix cache of a
previous prefill instead of recomputing; it is post-RoPE and therefore
`already_rotated=True`, which excludes `append_tail` placement below.

**Storage.** The `model_prefill` path stores K **pre-RoPE** (captured after
`k_norm`, before `rotary_emb`, `models/qwen3.py:generate_raw_repair_kv`) plus
the absolute position ids it was computed at. Injection applies RoPE once.
Before 2026-09-02 K was stored post-RoPE and copied verbatim; numerically the
two differ only by the rounding of applying RoPE in `c2kv_injection.py`
versus inside the model forward (same as the gist path already does).

**Placement** (`c2kv_repair_placement` on the chat message that carries the
repair hashes; `Scheduler._inject_c2kv_repair_entry`). With
`kv_start` = physical prefix length and `L` = logical position before the
repair (`kv_start + correction`):

| placement | K rotated at | ledger after | meaning / origin |
|---|---|---|---|
| `in_place` | its original absolute span `[p, p+n)` | `correction = (p+n) - (kv_start+n)`, i.e. the query continues at `p+n` | the raw span **replaces** its gist (gist not injected). Upstream `d_corr_replace_*` / `d_corr_recompute*` / `raw_all_replace*`; D-line `replaceG` |
| `append_keep_ledger` | its original span `[p, p+n)` | `correction -= n`, i.e. the query's logical position is unchanged | the raw span is a **duplicate** of a gist that stays. Upstream `d_corr`, `d_corr_w*`, `d_corr_all`; D-line `corr`, `keepG` |
| `append_tail` | fresh positions `[L, L+n)` | unchanged (logical and physical both grew by `n`) | the raw span is a **new note at the end of history**. D-line `raw_erratum_tail` (its best arm). Requires a pre-RoPE entry |

If the message carries no `c2kv_repair_placement`, the legacy rule applies:
`in_place` for the modes listed in `Scheduler._C2KV_IN_PLACE_REPAIR_MODES`,
`append_keep_ledger` otherwise. That rule is keyed on `repair_mode` strings
chosen by `bfcl-c2kv`; new clients should send the placement explicitly.

Note what `append_keep_ledger` implies in a mismatched frame (section 2): the
duplicate raw KV is pushed by the frame gap away from the query while its gist
twin stays adjacent. `in_place` re-anchors the query to the raw span and hides
the gap for everything downstream. This is a candidate explanation for
`bfcl-c2kv`'s "Append W2 below plain C2KV" result; it is not established.

## 4. What is compressed (client side, but affects how to read numbers)

The server compresses whatever messages carry `c2kv_key_hash`. Which
messages a client compresses is a client decision, and the training regime is:

- training (`train_data_multiturn.py:_session_examples`): everything before
  the **last input message** (user query or tool result, tool→user mapped)
  is packed into turn docs and compressed; only that last input message (+
  answer) is raw;
- `c2kv` repo `benchmarks/proxy.py` (`_history_cutoff`): same rule;
- `bfcl-c2kv`: everything before the last *real* user query is compressed;
  the whole current turn (query + steps so far) stays raw.

So "c2kv" in a `bfcl-c2kv` table and "c2kv" in a `c2kv`-repo table compress
different amounts of the same conversation.

## 5. Flags and fields added on 2026-09-02

| where | name | default | meaning |
|---|---|---|---|
| server flag | `--c2kv-query-proj {base,gist}` | `gist` | section 1 |
| chat message field | `c2kv_repair_placement` | `None` (legacy rule) | section 3 |
| `/v1/c2kv/repair_extract` request | `messages`, `target_index`, `tools` | `None` | section 3, form 1 |
| `/v1/c2kv/repair_extract` response | `already_rotated`, `span_start`, `span_end`, `rendered_prefix_len` | | provenance / frame check |
| chat response | `metadata.sglang_runtime.c2kv_query_proj`, `.c2kv_layout`, `.c2kv_position_correction`, `.c2kv_gist_seen` | | provenance / frame check |
| error text | `C2KV_CACHE_MISS:` prefix on pool misses | | structured retry (the gist pool is an in-process LRU; a miss after eviction is not a client error) |

## 6. Verification recipes

- Smoke everything above against a live server:
  `python scripts/c2kv/smoke_c2kv_semantics.py --base-url http://127.0.0.1:PORT`.
- Projection A/B: launch twice (`--c2kv-query-proj base` / `gist`), run the
  same compressed request set, compare. Until that number exists, treat the
  effect size as unknown.
- Frame check for any client: sum `original_seq_len` over `c2kv_layout`
  entries of kind `gist` and compare with the client's own raw token count of
  the same history.
- Repair fidelity: the `messages` form with `in_place` placement on every
  history doc should reproduce the Full arm's generation for the same prompt
  (up to kernel rounding). If it does not, the frame or the tool prologue is
  off; check `rendered_prefix_len` against the gist ledger.

## 7. Things that are still open

- Effect size of `--c2kv-query-proj gist` vs `base`: no A/B yet.
- The `text`+`role` repair form still exists and is silently lower fidelity;
  clients in the `c2kv` repo were moved to the `messages` form.
- `c2kv_gist_seen` / `c2kv_position_correction` are not restored on
  retraction or recovery-checkpoint restore; C2KV requests are not expected
  to be retracted (`c2kv_early_process` path), but this is untested.
- `_c2kv_project_qkv` computes both projections for a mixed batch and
  selects per token; cost is one extra QKV GEMM on those tokens.
