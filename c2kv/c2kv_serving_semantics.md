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

Every `file:line` below without a repo prefix is a path in **this** tree,
read on 2026-09-05 in the reconciled worktree (section 9). Paths belonging to
the training repo are prefixed `c2kv` repo; `benchmarks/*` is the C2KV bench
(`tmp/bench-recover`); `yuhan_*` are the upstream client and run scripts, which
live with that client and not here.

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
initialised as a copy of the base weights (`c2kv` repo,
`python/models/gist_utils.py:906-927` -- *not* this tree's
`python/sglang/srt/mem_cache/gist_utils.py`) and the only attention weights
that are trained (`--only_train_gist True`).

| token class | paper (§3.2.2) | training code | this server |
|---|---|---|---|
| document tokens during extraction | base | base | base (`forward_with_gist`) |
| gist tokens during extraction | gist | gist | gist (`forward_with_gist`, `gist_qkv_proj`) |
| system prefix (before any gist) | base | base (`trainer.py:_build_system_kv`, prefilled without `use_gist`) | base |
| **query / current turn / decoded tokens (after gist KV)** | base ("applied only to the original document tokens") | **gist**: `modeling_qwen3.py:673` sets `use_gist=True` for the whole main forward whenever `context_input_ids` (gists) are present, and `:242-246` then routes every token's q/k/v through `gist_*_proj` for the parts listed in `gist_param` (`qkv` in every training script) | resolved per request (rules below): an explicit message-level `c2kv_use_gist_projection` wins, otherwise `--c2kv-query-proj {base,gist}` decides (default `gist`, `server_args.py:586`). `gist` = gist projections for every token from the request's **first C2KV segment** onward -- gist or repair-only alike since **D7** (`scheduler.py:3258-3267`); `base` = the behaviour before 2026-09-02 |
| raw repair KV (`/v1/c2kv/repair_extract`) | n/a (paper §3.3.2 "original-token invariance": raw KV must equal the frozen base model's) | HF harness computes it with base projections | base (`models/qwen3.py:1363` `generate_raw_repair_kv` calls `self.qkv_proj` directly; it never enters the gist path) |

So the checkpoint was trained with a query that reads gist K through
`W_q^g` and produces its own K/V through `W_k^g, W_v^g`. Serving with base
projections for the query (the pre-2026-09 behaviour, still available as
`--c2kv-query-proj base`) is a train/serve mismatch. Because `W^g` starts as a
copy of `W` and is trained with LR 5e-5 for O(10^3) steps, the two differ by a
small delta; the size of the effect on downstream numbers is **not
established** (2026-09-05: still no A/B run).

Which parts of q/k/v are swapped is decided by `--c2kv-gist-param`
(`server_args.py:578`, default `qkv`) and by nothing else:
`Qwen3Attention.c2kv_query_proj_parts` (`models/qwen3.py:193`, filled at
`:214`) is derived from it, and with the default the per-part select in
`_c2kv_project_qkv` (`models/qwen3.py:271-313`) reduces to swapping the whole
fused QKV row, which is what both parents did.

### Which mode a request gets

Resolved once, when the request enters the scheduler
(`Scheduler.handle_generate_request`, `scheduler.py:2089-2116`), and only
inside `if getattr(recv_req, "c2kv_segments", None):` (`scheduler.py:2089`) --
a request with no C2KV segments is never touched and keeps the `Req` default
`False` (`schedule_batch.py:936`):

1. an explicit **request-level** `c2kv_use_gist_projection` wins
   (`scheduler.py:2105-2107`; `c2kv_query_proj_source = "message"`).
   `GenerateReqInput` carries the field (`io_struct.py:250`) but
   `ChatCompletionRequest` does not, so `serving_chat.py:788-790` always sends
   `None`: this branch is unreachable over `/v1/chat/completions` today;
2. otherwise, if any annotated **message** carries `c2kv_use_gist_projection`
   (`protocol.py:506` / `:534`, forwarded verbatim into
   `C2KVSegmentInfo.use_gist_projection` at `serving_chat.py:574-591`,
   `io_struct.py:2057`, `:2068`), the per-message fields decide: the request
   uses gist if **any** of them asks for it, a message that carries no field
   following the flag (`scheduler.py:2108-2113`; source `"message"`);
3. otherwise `--c2kv-query-proj` decides (`scheduler.py:2114-2116`; source
   `"flag"`). The resolver's own fallback is `"gist"`
   (`scheduler.py:2097-2099`), the same as the flag default.

Both fields are tri-state on the wire: absent means "unset", not `false`
(`protocol.py:506`, `io_struct.py:250`, `io_struct.py:2064-2068`). Collapsing
absent to `false` anywhere on the path makes the flag unreachable, which is why
`serving_chat.py:574-591` forwards each message's value verbatim instead of
folding the segments into one bool, and why the tokenizer hop passes `None`
rather than `False` (`tokenizer_manager.py:987-994`, **D8**).

Implementation -- one mechanism, no second mask:
`Req.c2kv_use_gist_projection` (the effective mode) and
`Req.c2kv_gist_projection_start_pos` (`schedule_batch.py:936-938`) ->
`ModelWorkerBatch.c2kv_use_gist_projection` /
`.c2kv_gist_projection_start_positions` (`schedule_batch.py:2543-2554`,
`:2652-2656`) -> `ForwardBatch.c2kv_use_gist_projection`, a per-token mask
`position >= start` (`model_executor/forward_batch_info.py:437`, built at
`:618-641`) -> `Qwen3Attention._c2kv_project_qkv` (`models/qwen3.py:271-313`).
The mask is compared against the *corrected* RoPE positions, because the
position correction is applied first in the same function
(`forward_batch_info.py:601-612`); see the frame caveat in section 8.
`ForwardBatch.c2kv_gist_proj_mask` and `ModelWorkerBatch.c2kv_gist_seen` no
longer exist; `Req.c2kv_gist_seen` (`schedule_batch.py:911`, set at
`scheduler.py:3902`) survives as provenance only.

The start position is the `token_start` of the request's **first C2KV
segment** -- gist or repair-only alike (`scheduler.py:3258-3267`). That is
upstream `d42ce815f`'s rule verbatim, kept under **D7** so that an
upstream-fields-only request runs exactly the projections it ran there (D4).
Serve-align's extra rule -- a request whose history is entirely raw repair KV
falls back to base, keyed on `Req.c2kv_gist_seen` -- is **DROPPED**. Nothing
now reads `c2kv_gist_seen` to decide a projection; it is echoed in
`metadata.sglang_runtime` as provenance and nowhere else.

That is a real behaviour change against `fork/task/c2kv-serve-align`, not a
no-op. A request that has C2KV segments but **no gist** segment -- its history
is entirely raw repair KV, e.g. a runtime history-KV arm (section 7), or the
bench's `c2kv_repair_inplace` arm on a conversation whose only compressed doc
has had its gist replaced by the raw span -- still *resolves* to gist under
rule 3 with the default flag, and is now served with **gist** projections
where serve-align served base. Such a request is identifiable in the response:
`c2kv_query_proj_effective = "gist"` next to `c2kv_gist_seen = false`. Do not
compare its numbers against a serve-align (Sep-2) trace without carrying that
column.

### What the response reports

`metadata.sglang_runtime` (`serving_chat.py:1528`; non-streaming responses
only) is `meta_info.kv_runtime_stats`, built by
`Scheduler._get_kv_runtime_stats` (`scheduler_output_processor_mixin.py:85`).

`c2kv_tools_dump` is reported whether or not C2KV is enabled
(`scheduler_output_processor_mixin.py:105-114`, outside the `enable_c2kv` gate
at `:118`): the tool-serialization mode decides the rendered prompt of
`/v1/chat/completions`, `/v1/c2kv/extract` and the `messages` form of
`/v1/c2kv/repair_extract` alike (section 2), i.e. the token frame every
client-computed insertion point is measured in, so a client must be able to
check it even on a run with no gists. One caveat on "always": the whole
`kv_runtime_stats` dict is dropped earlier, at `:94-96`, when the physical KV
snapshot is unavailable (`_get_physical_kv_snapshot` returns `None` without an
allocator, `:227-232`), so the guarantee is "present whenever
`metadata.sglang_runtime` is present at all", not "present unconditionally".
On a normally initialised server the allocator always exists.

Whenever `--enable-c2kv` (`:118`), four more keys follow (**D6**: the
projection provenance is three keys, deliberately not merged into one):

- `c2kv_query_proj` -- the value of the `--c2kv-query-proj` **server flag**
  (`scheduler_output_processor_mixin.py:151`). Constant for the life of the
  run, byte-identical to what serve-align echoed. It is deliberately *not*
  per-request: a client that keys "one serving regime" off this field (the
  bench's `mixed_query_proj` check, `benchmarks/reqlog.py:72`, `:95`) must not
  see a single-flag run self-report as mixed just because turn 1 of a
  conversation carries no gisted history yet.
- `c2kv_query_proj_effective` -- what THIS request actually ran: `"gist"` iff
  some token of it was projected with `gist_{q,k,v}_proj` (`:153-157`), else
  `"base"` (`:162`). A request with no C2KV segments never reaches the
  resolver and never builds a mask (`schedule_batch.py:2543-2547` ->
  `forward_batch_info.py:618`), so it reports `"base"` whatever the flag says.
- `c2kv_query_proj_source` -- which of the three rules chose the mode:
  `"message"` (an explicit request- or message-level
  `c2kv_use_gist_projection`), `"flag"` (`--c2kv-query-proj`), or `"none"` --
  no C2KV segments, so no projection decision was ever made for this request
  (`:158-160`, `:163`). `"none"` is what separates a segment-less request from
  a genuinely flag-defaulted one; before the reconciliation both said
  `"flag"`.
- `c2kv_query_proj_decode_verified` (`:173-176`) -- see below.

and the per-request injection ledger -- `c2kv_layout`,
`c2kv_position_correction`, `c2kv_gist_seen`
(`scheduler_output_processor_mixin.py:177-196`). Entries are appended by
`Scheduler._c2kv_layout_append` (`scheduler.py:3995-4000`), called at
`scheduler.py:3903` for a gist and `:4256` for a repair entry.

**The three ledger keys are always present on a C2KV-enabled server**
(2026-09-05 change): when nothing was injected they read `[]`, `0` and `false`
(`scheduler_output_processor_mixin.py:185-196`) instead of being omitted, so
`c2kv_layout == []` is a positive "this request injected nothing" -- a
`full`-arm request, or turn 1 of every compression arm. They are still absent
in two other cases, and those are what an absent key now means: the whole
`kv_runtime_stats` dict was dropped at `:94-96` (no allocator), or the server
is not running `--enable-c2kv` (the `elif` is inside the gate at `:118`).
Before this change the keys were omitted whenever the ledger was empty, and
"nothing injected" was indistinguishable from "no ledger reported".

Finally, `c2kv_injection_error` -- a machine-readable `C2KV_*` string, present
only on a request whose C2KV injection failed
(`scheduler_output_processor_mixin.py:205-207`, written by
`Scheduler._set_c2kv_injection_error`, `scheduler.py:3973-3992`). Section 3
lists what produces it and how it reaches an OpenAI client.

Known limitation: the mask is built per forward from request state; it is not
part of CUDA/NPU graph capture, so a **replayed decode graph runs the base
projection whatever the flag says**, while prefill runs the flag's mode. Serve
C2KV with `--disable-cuda-graph` if decoded tokens must follow the flag (the
c2kv position correction has the same limitation).
`c2kv_query_proj_decode_verified` is `false` exactly when
`c2kv_query_proj_effective` is `gist` and graph capture is on
(`scheduler_output_processor_mixin.py:173-176`) -- keyed on the **effective**
mode, not the flag, because a request that ran base has nothing for graph
replay to have lost.

The scheduler logs once at startup (**D9**, `scheduler.py:899-927`) in exactly
the configuration where the flag path loses the mask: inside
`if server_args.enable_c2kv:` (`scheduler.py:851`) and gated on
`c2kv_query_proj == "gist" and not server_args.disable_cuda_graph`
(`scheduler.py:903-906`). A `--disable-cuda-graph` server is silent, which is
what the D9 change bought.

The text says only what the code shows (message built at
`scheduler.py:913-921`): graph-captured decode steps do not rebuild the C2KV
gist-projection mask; decoded tokens of a request that resolved to gist may run
base projections; `c2kv_query_proj_decode_verified` reports false for such
requests; pass `--disable-cuda-graph` for a projection A/B. It no longer
declares that such a run "must not be used as one side of a gist-vs-base A/B"
-- that was a verdict about experiments, not a reading of the code, and what
the code establishes is the four clauses above. On `--device npu` the same
sentence is logged at **INFO** with "(graph behaviour on Ascend not verified)"
appended (`scheduler.py:922-925`), because nothing in this tree establishes how
Ascend graph mode replays the mask.

**The startup line covers the flag path only.** Its gate reads
`server_args.c2kv_query_proj` (`scheduler.py:903-906`) and never the resolved
per-request mode: on a `--c2kv-query-proj base` server a message-level
`c2kv_use_gist_projection: true` still resolves that request to gist
(`scheduler.py:2105-2113`) and still builds a mask graph capture does not
carry, and no startup line says so. Read that server per request instead --
`c2kv_query_proj_effective="gist"` next to
`c2kv_query_proj_decode_verified=false`
(`scheduler_output_processor_mixin.py:153-157`, `:173-176`) is the per-request
form of the same statement, and it is what a base-flag server with per-message
gist overrides has to be judged on. Any server started with `--enable-c2kv` and
neither `--disable-cuda-graph` nor `--c2kv-query-proj base` prints the line;
that includes the upstream history-KV run script
(`yuhan_run_history_kv_baselines.sh:187`, client-side file, not in this tree),
which passes `--enable-c2kv` and never `--disable-cuda-graph`. Expect the line
there; it is D9 behaving as specified, not a misconfiguration introduced by the
merge. For the per-run statement read the startup line, or `c2kv_query_proj`
together with `--disable-cuda-graph`; for a single request read
`c2kv_query_proj_decode_verified`.

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

The tool prologue is part of the frame, and how it is rendered is a server
flag: `--c2kv-tools-dump` (`ServerArgs.c2kv_tools_dump`, `server_args.py:599`,
validated at `:789-792`, CLI at `:5290`).

- `full` (default): `model_dump()`, i.e. pydantic defaults are emitted
  (`"strict": false`, +4 tokens per tool with the Qwen3 template). This is what
  this server rendered before 2026-09-05, so it is the frame every already
  collected trajectory and every frozen reference was produced under. A client
  that tokenizes the same tool JSON itself cannot see the offset.
- `exclude_unset`: `model_dump(exclude_unset=True)`, i.e. the rendered prologue
  equals the client's tool JSON. Use it when the client predicts C2KV insertion
  points or repair positions itself, or compares its own token counts with the
  server's.

The flag governs the three renderings that define the served token frame,
through one helper (`chat_template_tools_dump`, `protocol.py:554-581`):

1. the `/v1/chat/completions` prompt -- `_c2kv_tools_dump_exclude_unset`
   (`serving_chat.py:357-371`) feeding `_chat_template_tools`
   (`serving_chat.py:373-388`), the single source used at `:544`, `:645` and
   `:818`;
2. `/v1/c2kv/extract` with `tools` -- `http_server._c2kv_flat_tools`
   (`http_server.py:1460-1485`), applied at `:1513`;
3. the `messages` form of `/v1/c2kv/repair_extract` -- `_c2kv_template_ids`
   (`http_server.py:1559-1575`), which calls the same `_c2kv_flat_tools` at
   `:1564`.

Both endpoints read the same server flag (`http_server.py:1476-1481`), so the
chat frame, the extract frame and the repair frame never disagree with each
other -- but they all move together when the flag changes, so trajectories
collected under the two values are not comparable and a frozen reference must
be regenerated after a change. `full` reproduces upstream `d42ce815f`;
`exclude_unset` reproduces `fork/task/c2kv-serve-align` (section 9).

Four holes in that parity, all read off the merged tree and none of them a
reconciliation regression -- listed so "all three renderings" is not read as
"all renderings ever":

- **DeepSeek-V3.2 encoding path.** Inside `_apply_jinja_template`,
  `if self.use_dpsk_v32_encoding:` re-serialises the tools with a plain
  `model_dump()` and bypasses the helper entirely
  (`serving_chat.py:895-896`). On such a model the chat prologue is always the
  `full` dump while the two C2KV endpoints follow the flag. Identical to
  upstream; unreachable for the Qwen3 C2KV target.
- **`tool_choice`.** `_chat_template_tools` returns `None` for
  `tool_choice == "none"` and narrows the list to the single named function for
  a `ToolChoice` object (`serving_chat.py:376-387`). `_c2kv_flat_tools` has no
  `tool_choice` concept and neither `C2KVExtractRequest`
  (`protocol.py:1559-1570`) nor `C2KVRepairExtractRequest`
  (`protocol.py:1583-1611`) declares the field. A client that pins
  `tool_choice` on the chat call but extracts against the full tool list
  measures `original_seq_len` in a longer prologue than it is served. BFCL
  leaves `tool_choice` at its default, so this is not hit today.
- **Function-only fallback.** When `apply_chat_template` rejects the
  OpenAI-wrapped tools the chat path retries with the flattened
  `t["function"]` form (`serving_chat.py:462`, `:965`); the two C2KV endpoints
  have no such retry, so on a template that needs it chat succeeds and extract
  returns `success=false`. Loud, not silent, and Qwen3 does not need it.
- **Raw-dict items.** `chat_template_tools_dump` passes a non-pydantic item
  through as `dict(item)` (`protocol.py:579-580`), i.e. always
  `exclude_unset` semantics whatever the flag says. Unreachable today -- both
  callers hand it `Tool` models (`serving_chat.py:380`, `:388`;
  `http_server.py:1483` validates dicts first) -- but a future raw-dict caller
  would silently render the wrong prologue.

`/v1/c2kv/extract` used to be an exception that passed its `tools` through as
the raw JSON the client sent (4 tokens per tool short of `full`). It is not any
more (`http_server.py:1505-1513`), so a client that measures `original_seq_len`
through `/v1/c2kv/extract` now measures it in the frame it will be served in,
under either value of the flag.

The active value is echoed per request as
`metadata.sglang_runtime.c2kv_tools_dump`
(`scheduler_output_processor_mixin.py:114`), outside the `enable_c2kv` gate at
`:118`, because the flag moves the prompt whether or not C2KV is on.

## 3. Repair KV: source, storage form and placement

`/v1/c2kv/repair_extract` stores the raw (base-projection) K/V of a span of
tokens so it can be injected next to, or instead of, gist KV.

Three knobs are involved and they are **orthogonal**. Do not derive one from
another:

| knob | where | values | decides |
|---|---|---|---|
| source | the request form of `/v1/c2kv/repair_extract`, plus `extract_source` (`protocol.py:1597`) | `messages`+`target_index` / `input_ids`+span / `text`+`role`; `model_prefill` / `serving_cache` | which K/V is captured |
| **storage form** | `raw_kv_position_mode` (`protocol.py:1594`, default `rotated`) | `rotated` / `pre_rope` | whether the stored K is post- or pre-RoPE, i.e. `already_rotated` |
| **placement** | `c2kv_repair_placement` on the chat message that carries the repair hashes (`protocol.py:509`, `:537`) | `in_place` / `append_keep_ledger` / `append_tail` | where the entry lands in the position ledger at injection |

The storage form is upstream `d42ce815f`'s knob and the placement is
serve-align's (section 9); the merged server keeps both. Their only coupling is
one guard: `append_tail` must re-rotate the span, so it needs a pre-RoPE entry
and refuses an already-rotated one (`scheduler.py:4173-4185`).

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

**Storage form.** Chosen by `raw_kv_position_mode` on the request, not by the
extract path:

- `pre_rope`: K is captured after `k_norm` and before `rotary_emb`
  (`models/qwen3.py:1363` `generate_raw_repair_kv`) and stored with the
  absolute position ids it was computed at (`already_rotated=false`); injection
  applies RoPE once, at whatever position the placement asks for.
  **`append_tail` requires this form**: on an already-rotated entry the
  injection is refused with
  `C2KV_APPEND_TAIL_REQUIRES_PRE_ROPE: ...` (`scheduler.py:4173-4185`, message
  built at `:4175-4181`), which names the fix (re-extract with
  `raw_kv_position_mode='pre_rope'`). There is no silent fallback, and since
  2026-09-05 the reason does reach an OpenAI client -- on
  `metadata.sglang_runtime.c2kv_injection_error` / `metadata.finish_message`,
  still inside an HTTP 200 abort. See the end of this section.
- `rotated` (the field default, `protocol.py:1594`): K is stored post-RoPE and
  copied verbatim at injection (`already_rotated=true`). Numerically the two
  differ only by the rounding of applying RoPE in `c2kv_injection.py` versus
  inside the model forward (same as the gist path already does), but a rotated
  entry can only be re-used at the position it was extracted at.

Because the field default is `rotated`, the `messages` + `target_index` form
overrides it to `pre_rope` when the caller does not send the field at all
(`http_server.py:1663-1677`, which inspects `model_fields_set`): that form
exists to produce entries that get re-placed, and it is the form the D-line arms
use. The `input_ids` and `text` forms keep the field default `rotated`, and an
explicit `raw_kv_position_mode` wins in every form. On the internal hop the
mode is folded into one boolean,
`already_rotated or raw_kv_position_mode != "pre_rope"`
(`tokenizer_communicator_mixin.py:476-478`), and `extract_source=serving_cache`
is post-RoPE by construction (`scheduler.py:2765-2766`). Asking for both at
once is rejected, not silently downgraded: `serving_cache` together with an
explicit or defaulted `raw_kv_position_mode="pre_rope"` returns
`success=false` with "serving_cache repair extraction cannot return pre-RoPE K"
(`scheduler.py:2550-2557`). Note the interaction with the `messages`-form
default above: that form defaults the mode to `pre_rope`, so `messages` +
`serving_cache` fails unless the caller sends `raw_kv_position_mode="rotated"`
explicitly. The stored form is echoed as `already_rotated` in the
repair_extract response (`protocol.py:1634`) and in every `c2kv_layout` repair
entry (`scheduler.py:4263-4266`).

**Placement** (`c2kv_repair_placement` on the chat message that carries the
repair hashes; `Scheduler._inject_c2kv_repair_entry`). With
`kv_start` = physical prefix length and `L` = logical position before the
repair (`kv_start + correction`):

| placement | K rotated at | ledger after | meaning / origin |
|---|---|---|---|
| `in_place` | its original absolute span `[p, p+n)` | `correction = (p+n) - (kv_start+n)`, i.e. the query continues at `p+n` | the raw span **replaces** its gist (gist not injected). Upstream `d_corr_replace_*` / `d_corr_recompute*` / `raw_all_replace*`; D-line `replaceG` |
| `append_keep_ledger` | its original span `[p, p+n)` | `correction -= n`, i.e. the query's logical position is unchanged | the raw span is a **duplicate** of a gist that stays. Upstream `d_corr`, `d_corr_w*`, `d_corr_all`; D-line `corr`, `keepG` |
| `append_tail` | fresh positions `[L, L+n)` | unchanged (logical and physical both grew by `n`) | the raw span is a **new note at the end of history**. D-line `raw_erratum_tail` (its best arm). Requires a pre-RoPE entry |

**Precedence.** An explicit `c2kv_repair_placement` on the message always wins.
When it is absent, the legacy rule applies
(`Scheduler._resolve_c2kv_repair_placement`, `scheduler.py:4002-4015`;
the legacy branch is `:4003-4009`): `in_place` for the modes listed in
`Scheduler._C2KV_IN_PLACE_REPAIR_MODES` (`scheduler.py:3954-3966`: the
`d_corr_recompute*` / `d_corr_replace*` / `raw_all_replace*` modes plus
`append_masked_w2`) or for any `repair_mode` starting with `history_kv_`
(`_C2KV_IN_PLACE_REPAIR_MODE_PREFIXES`, `scheduler.py:3970` -- this is what
keeps the upstream history-KV arms of section 7 on their pre-merge placement),
`append_keep_ledger` otherwise. The legacy rule is keyed on `repair_mode`
strings chosen by `bfcl-c2kv`; new clients should send the placement
explicitly.

An unrecognised placement string is caught twice. For a chat request it is
rejected at admission (**D10**), in the per-segment validation loop of
`_build_c2kv_prefill_rounds` (`scheduler.py:3284-3299`, checked against
`_C2KV_REPAIR_PLACEMENTS`, `scheduler.py:3971`), which returns an error string
that `handle_generate_request` turns into `req.set_finish_with_abort`
(`scheduler.py:2127-2129`) -- the same channel as the `C2KV_CACHE_MISS:` path,
and that one really is an HTTP 400 carrying the message:
`set_finish_with_abort` builds `FINISH_ABORT(..., HTTPStatus.BAD_REQUEST)`
(`schedule_batch.py:1428-1438`), which `tokenizer_manager.py:1214-1221` turns
into a `ValueError` and the serving layer into a 400 response.
`_resolve_c2kv_repair_placement` still raises the same `ValueError` at
injection time (`scheduler.py:4010-4014`); with the admission check in place
that is now a backstop rather than the path a client hits.

Every injection is recorded in `metadata.sglang_runtime.c2kv_layout` with both
knobs: `placement`, `already_rotated` and `raw_kv_position_mode` next to
`kv_start`, `repair_len`, `position_start`, `position_end`, `logical_before`
and `position_correction_after` (`scheduler.py:4256-4274`; the ledger arithmetic
that precedes it is `:4243-4255`).

**What a failed injection looks like to a client.** Everything above that fails
*during* prefill ends as
`FINISH_ABORT(req.c2kv_injection_error or "C2KV injection failed")` with no
status code (`scheduler_output_processor_mixin.py:431-461`; the `FINISH_ABORT`
is built at `:457-460` with no `status_code`, and `FINISH_ABORT` defaults it to
`None`, `schedule_batch.py:186-190`). Because `tokenizer_manager.py:1214-1221`
tests for `HTTPStatus.BAD_REQUEST`, neither error branch fires: that is an
**HTTP 200** whose `finish_reason.type` is `"abort"`, and the per-choice
`finish_reason` of `/v1/chat/completions` copies only `finish_reason["type"]`
(`serving_chat.py:1495`). The status code is deliberately unchanged (D4): an
upstream client sees the same 200 it always saw.

Since 2026-09-05 the *reason* is readable anyway, on two extra keys and one
prefix rule:

- `req.c2kv_injection_error` is set by **every** injection-time failure path,
  through `Scheduler._set_c2kv_injection_error` (`scheduler.py:3973-3992`,
  first writer wins so a nested repair failure keeps its own specific reason).
  In `_inject_c2kv_gist_segment` that is the pool-unavailable guard
  (`scheduler.py:3461-3467`), a repair-only segment with no keys (`:3471-3479`),
  the three cache misses (repair-only `:3482-3493`, gist `:3517-3525`,
  repair-attached `:3916-3926`), an invalid logical KV start (`:3533-3552`),
  allocation failure (`:3766-3783`), context overflow (`:3802-3824`), the
  `req_to_token` write (`:3826-3845`) and the injection call itself
  (`:3874-3889`); in `_inject_c2kv_repair_entry` the pool guard
  (`:4033-4039`), the placement guard (`:4040-4046`), allocation
  (`:4117-4130`), context overflow (`:4137-4153`), the `req_to_token` write
  (`:4155-4168`), the `append_tail` pre-RoPE guard (`:4173-4185`) and the
  injection call (`:4212-4220`).
- **The three cache misses now carry the `C2KV_CACHE_MISS:` prefix too**
  (`scheduler.py:3489`, `:3522`, `:3922`), the same prefix the admission-time
  misses use (`:3309`, `:3320`, `:3422`). A client's cache-miss retry
  therefore fires for an eviction discovered at injection, which it did not
  before. The other paths keep or gain their own `C2KV_*` codes
  (`C2KV_APPEND_TAIL_REQUIRES_PRE_ROPE`, `C2KV_REPAIR_PLACEMENT_INVALID`,
  `C2KV_ALLOC_FAILED`, `C2KV_CONTEXT_OVERFLOW`, ...).
- The string is echoed to the client twice, both as *extra* keys:
  `metadata.sglang_runtime.c2kv_injection_error`
  (`scheduler_output_processor_mixin.py:205-207`) and
  `metadata.finish_message` -- `finish_reason["message"]`, attached only when
  the finish is an abort (`serving_chat.py:1537-1539`, next to
  `sglang_runtime` in the same dict, `:1520-1548`). Two surfaces because
  `_get_kv_runtime_stats` returns `None`, and `metadata.sglang_runtime` with
  it, when the physical KV snapshot is unavailable
  (`scheduler_output_processor_mixin.py:94-96`); `finish_message` survives
  that.

Both surfaces are non-streaming only, like the rest of `metadata`
(`serving_chat.py:1541-1548`; section 8). The server log still carries the same
string, and an admission-time error (section 5) is still the louder channel:
it is a real HTTP 400.

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

## 5. Flags and fields: what each side owns

"Owner" is where the field entered this tree: **upstream** = `d42ce815f`
(the history-KV baseline line), **serve-align** =
`fork/task/c2kv-serve-align` (`b08172044` + `718a654e3`), **merge** = added by
the 2026-09-05 reconciliation itself (section 9). All of them are live in the
merged server.

### Server flags (`server_args.py`)

| flag | default | owner | meaning |
|---|---|---|---|
| `--enable-c2kv` | `False` (`:576`, CLI `:5247`) | pre-existing | gates the pool and everything below |
| `--c2kv-gist-type` | `dynamic-interleave` (`:577`, CLI `:5252`) | pre-existing | `GistConfig.gist_type` (`models/qwen3.py:1119-1120`); only `dynamic-interleave` is implemented (CLI help, `server_args.py:5255`) |
| `--c2kv-gist-param` | `qkv` (`:578`, CLI `:5258`) | pre-existing | which of q/k/v the gist projection replaces (`models/qwen3.py:214`) |
| `--c2kv-pool-fraction` | `0.01` (`:579`, validated `:783-784`, CLI `:5264`) | pre-existing | share of device memory given to the gist/repair pool (`scheduler.py:860-871`) |
| `--c2kv-max-tokens` | `65536` (`:580`, validated `:785-786`, CLI `:5271`) | pre-existing | per-entry cap in the pool |
| `--c2kv-query-proj {base,gist}` | `gist` (`:586`, validated `:787-788`, CLI `:5277`) | serve-align | section 1; in the merged server it is the per-request **default** only -- an explicit message field overrides it |
| `--c2kv-tools-dump {full,exclude_unset}` | `full` (`:599`, validated `:789-792`, CLI `:5290`) | merge | section 2; `full` = upstream's tool prologue, `exclude_unset` = serve-align's |

### Chat message fields (`ChatCompletionMessageGenericParam` / `...UserParam`)

| field | default | owner | meaning |
|---|---|---|---|
| `c2kv_key_hash` | `None` (`protocol.py:499`, `:527`) | pre-existing | the gist entry that replaces this message |
| `c2kv_repair_key_hashes` / `c2kv_repair_only_key_hashes` | `None` (`:500-501`, `:528-529`) | pre-existing | repair entries injected next to / instead of the gist |
| `c2kv_repair_token_start` | `None` (`:502`, `:530`) | pre-existing | |
| `c2kv_use_gist_projection` | `None` = unset (`:506`, `:534`) | upstream | section 1, rule 2; tri-state |
| `c2kv_repair_placement` | `None` = legacy rule (`:509`, `:537`) | serve-align | section 3 |

Both reach the scheduler as `C2KVSegmentInfo.use_gist_projection` /
`.repair_placement` (`io_struct.py:2057-2058`, assigned at `:2068` and `:2072`,
filled at `serving_chat.py:574-591`).

### Chat request / response

| field | default | owner | meaning |
|---|---|---|---|
| `ChatCompletionRequest.c2kv_kv_memory_hint` | `None` (`protocol.py:644`) | upstream | section 7 |
| `GenerateReqInput.c2kv_use_gist_projection` | `None` (`io_struct.py:250`) | upstream, retyped tri-state by the merge | section 1, rule 1 (no chat field feeds it today) |
| `metadata.sglang_runtime.c2kv_query_proj` | | serve-align | section 1 -- the `--c2kv-query-proj` **flag**, constant per run (`scheduler_output_processor_mixin.py:151`) |
| `metadata.sglang_runtime.c2kv_query_proj_effective` | | merge (D6) | section 1 -- `"gist"` / `"base"`, what this request actually ran (`:153-157`, `:162`) |
| `metadata.sglang_runtime.c2kv_query_proj_source` | | merge | `"message"` / `"flag"` / `"none"` (`:158-160`, `:163`); `"none"` = no C2KV segments |
| `metadata.sglang_runtime.c2kv_query_proj_decode_verified` | | merge | section 1, graph capture; keyed on `_effective` (`:173-176`) |
| `metadata.sglang_runtime.c2kv_tools_dump` | | merge (D5) | the active `--c2kv-tools-dump` (`:114`), emitted whether or not C2KV is enabled |
| `metadata.sglang_runtime.c2kv_layout`, `.c2kv_position_correction`, `.c2kv_gist_seen` | `[]`, `0`, `false` when nothing was injected; absent only off `--enable-c2kv` or with no allocator | serve-align, always-present since 2026-09-05 | per-request injection ledger (`:177-196`); since D7 `c2kv_gist_seen` is provenance only (section 1) |
| `metadata.sglang_runtime.c2kv_injection_error` | absent unless the injection failed | 2026-09-05 | machine-readable `C2KV_*` reason (`:205-207`); section 3 |
| `metadata.finish_message` | absent unless `finish_reason.type == "abort"` | 2026-09-05 | `finish_reason["message"]` lifted to response level so an abort's reason survives the choice, which keeps only the type (`serving_chat.py:1537-1539`, `:1495`); section 3 |
| `metadata.kv_memory_report` | `None` | upstream | section 7 (`serving_chat.py:1512`, `:1526`) |
| `meta_info.persistent_history_session` | absent | upstream | section 7 (`serving_chat.py:233-239`) |
| error prefix `C2KV_CACHE_MISS:` | | serve-align, extended 2026-09-05 | admission-time misses `scheduler.py:3309`, `:3320`, `:3422`; **injection-time misses now carry it too** -- repair-only `:3489`, gist `:3522`, repair-attached `:3922` -- so the structured retry (the gist pool is an in-process LRU; a miss after eviction is not a client error) fires for both. Before this change an injection-time miss returned a bare `False` with no text and reached the client as a text-less abort |
| error prefix `C2KV_APPEND_TAIL_REQUIRES_PRE_ROPE:` | | merge | `scheduler.py:4176` (guard `:4173-4185`); section 3. Readable by an OpenAI client since 2026-09-05, on `metadata.sglang_runtime.c2kv_injection_error` / `metadata.finish_message` |
| error prefix `C2KV_REPAIR_PLACEMENT_INVALID:` | | pre-2026-09-05 (injection), 2026-09-05 (admission) | `scheduler.py:4043` at injection and `:3296` at admission (D10) -- the same code on both channels, so a client can key on it whether the request is refused with HTTP 400 or aborted at 200. Which parent introduced the injection-time string was not traced |
| `metadata.sglang_runtime` on `/v1/chat/completions` | absent for `stream=true` | serve-align | attached only in `_build_chat_response` (`serving_chat.py:1520-1548`); `ChatCompletionStreamResponse` has no `metadata` field (`protocol.py:955-962`) |

**Batched `/generate` rejects C2KV fields.** `GenerateReqInput.__getitem__`
(`io_struct.py:635-699`) rebuilds each item field by field and copies none of
`c2kv_segments` / `c2kv_kv_memory_hint` / `c2kv_use_gist_projection`, so a
batched request used to reach the scheduler with `c2kv_segments=None` and be
served as a plain prompt with a normal 200. Since 2026-09-05
`normalize_batch_and_arguments` raises
`ValueError("C2KV fields are not supported on batched requests")` when the
request is not single and any of the three is set
(`io_struct._validate_c2kv_not_batched`, `io_struct.py:366-388`, called at
`:290`). The fields are **not** propagated: a loud error beats silent
plain-prompt serving, and propagating them would be new behaviour neither
parent branch had. Note the check runs after `_handle_parallel_sampling`
(`io_struct.py:340-364`), which turns `n > 1` into a non-single request, so a
C2KV chat request with `n > 1` (`ChatCompletionRequest.to_sampling_params`
sets `"n"`, `protocol.py:836`) is refused as well -- that combination took the
same `obj[i]` path and was equally broken before.

Note that `metadata.sglang_runtime` reaches the client only because
`kv_runtime_stats` is relayed across the detokenizer hop
(`detokenizer_manager.py:339-340`, serve-align `718a654e3`); the parallel
`kv_memory_reports` relay on the same line is upstream's.

### C2KV endpoints

| where | field | default | owner |
|---|---|---|---|
| `/v1/c2kv/extract` request | `tools` | `None` (`protocol.py:1570`) | serve-align |
| `/v1/c2kv/repair_extract` request | `repair_position_ids`, `raw_kv_position_mode` | `None`, `"rotated"` (`:1593-1594`) | upstream |
| | `history_kv_method`, `history_kv_target_tokens`, `history_kv_retention_ratio`, `history_kv_recent_window`, `history_kv_kernel_size`, `history_kv_pooling`, `history_kv_h2o_recent_fraction` | `None`, `None`, `None`, `64`, `5`, `"avgpool"`, `0.5` (`:1599-1605`) | upstream |
| | `messages`, `target_index`, `tools` | `None` (`:1609-1611`) | serve-align |
| `/v1/c2kv/repair_extract` response | `history_kv_method`, `requested_span_tokens`, `selected_token_count`, `selected_relative_indices` | `None`, `0`, `0`, `None` (`:1627-1630`) | upstream |
| | `already_rotated`, `span_start`, `span_end`, `rendered_prefix_len` | `False`, `0`, `0`, `0` (`:1634-1639`) | serve-align |

There is deliberately **no** `already_rotated` field on the repair_extract
**request**: the storage form is controlled only by `raw_kv_position_mode`
(section 3; the internal kwarg still exists on the tokenizer hop,
`tokenizer_communicator_mixin.py:447`, and no HTTP field reaches it). The
**response** does echo the resulting form as `already_rotated`
(`protocol.py:1634`).

## 6. Verification recipes

- Smoke part of the above against a live server:
  `python scripts/c2kv/smoke_c2kv_semantics.py --base-url http://127.0.0.1:PORT`.
  Read what it actually asserts before trusting a pass. It covers the extract
  ledger, the gist frame, the `messages`-form repair span and the three
  placements, and since 2026-09-05 also:
  the D6 three-key contract plus `c2kv_tools_dump` on a compressed request
  (`scripts/c2kv/smoke_c2kv_semantics.py:143-167`), including that
  `c2kv_query_proj_source` is `"flag"` and `c2kv_query_proj_effective` agrees
  with `c2kv_query_proj` when no message overrides it; `c2kv_layout` is a list
  (`:168-171`); a `full`-arm request reports the five projection keys,
  `base`/`none`, and `c2kv_layout == []` with
  `c2kv_gist_seen=false`/`c2kv_position_correction=0` (`:184-225`); and an
  invalid `c2kv_repair_placement` is refused with
  `C2KV_REPAIR_PLACEMENT_INVALID` in the response, on either surface --
  the HTTP 400 body or `metadata.finish_message` /
  `metadata.sglang_runtime.c2kv_injection_error` (`:284-333`, helper
  `post_allow_error` at `:74-102`). Still **not** covered: the D7 repair-only
  mask start (no repair-only-history turn is sent), and no assertion ties
  `c2kv_query_proj` to the `--c2kv-query-proj` the server was actually started
  with -- the script is not told the flag.
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
- History-KV arms (section 7): the measured numbers are
  `metadata.kv_memory_report.history_kv_physical_eviction`
  (`kept_history_tokens`, `freed_physical_slots`, `new_physical_kv_slots`,
  `mem_cache/history_kv_eviction.py:30-50`) and, for the extraction path,
  `selected_token_count` in the repair_extract response
  (`protocol.py:1629`). The client-side `active_history_kv_tokens` in the hint
  is an estimate and says so (`estimated: true`,
  `yuhan_client_history_kv_baselines.py:646`; cleared to `false` at `:660` only
  when a real runtime extract happened). Do not report it as measured.
- Comparability: record the value of `--c2kv-tools-dump` next to every run.
  Two runs under different values were served different prompts (section 2)
  and must not be compared, however identical their arm labels look.

## 7. History-KV eviction baselines (upstream `d42ce815f`)

These are the non-C2KV baselines the upstream client runs against the same
server: the completed history stays ordinary raw KV and is *shrunk*, instead of
being replaced by gists. They never inject a gist, but they share the C2KV
request plumbing, so they belong here.

Method names, after the server's aliasing (`snapkv` -> `snapkv_persistent`,
`pyramid` -> `pyramidkv`, `scheduler.py:2462-2465` on the extraction path and
`:2871-2874` on the eviction path):

| `history_kv_method` | accepted at extraction (`scheduler.py:2466-2472`) | accepted for physical eviction (`scheduler.py:2884-2886`) |
|---|---|---|
| `streamingllm` | yes | yes |
| `h2o` | yes | yes |
| `snapkv_persistent` | yes | yes |
| `pyramidkv` | yes | yes |
| `snapkv_refresh` | yes | no |
| `full`, `c2kv` | client-side arm labels only; the server never sees a `history_kv_method` for them | -- |

Two mechanisms, chosen by the client:

**(a) Selection at extraction time.** `/v1/c2kv/repair_extract` with
`history_kv_method` compresses the extracted span *before* storing it; the
client then injects the stored entry through a repair-only carrier message.
Budget: `history_kv_target_tokens`, else `history_kv_retention_ratio` x span
(`scheduler.py:2495-2506`). Shape parameters: `history_kv_recent_window` (64),
`history_kv_kernel_size` (5), `history_kv_pooling` (`avgpool`),
`history_kv_h2o_recent_fraction` (0.5) (`protocol.py:1599-1605`). Only
`extract_source="model_prefill"` supports it -- `serving_cache` is rejected
with an explicit error (`scheduler.py:2603-2613`). The response echoes
`history_kv_method`, `requested_span_tokens`, `selected_token_count` and
`selected_relative_indices` (`protocol.py:1627-1630`, filled at
`scheduler.py:2799-2807`), which is the only place the *actual* number of kept
tokens is reported.

**(b) Physical eviction of the already-prefilled history.** The chat request
carries `c2kv_kv_memory_hint.history_kv_eviction`; the prefill is split into a
round ending at `history_end` and a round for the rest
(`scheduler.py:2938-2946`), and after the first round
`PhysicalHistoryKVEvictor` compacts the surviving tokens into the request's
leading pages and frees only the pages that no longer hold a destination slot
(`mem_cache/history_kv_eviction.py:53-63`, driven from
`scheduler.py:2974-2991`). The server picks the kept indices itself when the
client sends none (`scheduler.py:2981-2983`); the attention-score methods
(`h2o`, `snapkv*`, `pyramidkv`) require a selection and fail without one
(`mem_cache/history_kv_eviction.py:65`, `:125-131`). Prefix sharing is refused
outright: `PHYSICAL_HISTORY_KV_EVICTION_SHARED_PREFIX_UNSUPPORTED`
(`scheduler.py:2888-2892`) -- run these arms with `--disable-radix-cache`.

**Hint shape, as the upstream client sends it**
(`yuhan_client_history_kv_baselines.py:634-660`; the inner eviction block is
built at `:514-530`, the session block at `:650-653`, and
`active_raw_repair_tokens` / `estimated: false` are added at `:654-660` when the
arm went through `/v1/c2kv/repair_extract`):

```json
{
  "full_equivalent_history_tokens": 0,
  "active_history_kv_tokens": 0,
  "active_full_raw_tokens": 0,
  "active_c2kv_gist_tokens": 0,
  "active_raw_repair_tokens": 0,
  "history_kv_method": "h2o",
  "estimated": false,
  "history_kv_eviction": {
    "method": "h2o",
    "history_message_count": 12,
    "target_tokens": 512,
    "retention_ratio": 0.25,
    "history_kv_recent_window": 64,
    "history_kv_kernel_size": 5,
    "history_kv_pooling": "avgpool",
    "history_kv_h2o_recent_fraction": 0.5,
    "persistent_session": true
  },
  "persistent_history_session": {"enabled": true}
}
```

The client deliberately sends **no token offsets**, only
`history_message_count`: the server re-renders `messages[:count]` through its
own chat template and resolves `history_start` / `history_end` in its own token
frame, refusing the request when the span cannot be located exactly
(`serving_chat.py:613-682`), and writes `server_tokenized: true` plus
`full_equivalent_history_tokens` back into the hint (`:678`, `:681`). This is
the second place where the tool-prologue flag of section 2 moves a
measurement.

**Persistent history sessions.** With `persistent_history_session.enabled`
plus `session_params.id`, the chat request is turned into the exact append
delta against the stored canonical prompt (`serving_chat.py:143-213`):
streaming is refused (`:152`), a prompt that does not extend the previous one
raises `PERSISTENT_HISTORY_SESSION_PREFIX_MISMATCH` (`:185-189`), and the canonical
prompt is stored **pre-decode**, so the generated suffix is re-prefilled in its
canonical serialization on the next turn (`:215-239`). `/close_session`
releases the entry (`http_server.py:1341-1348`).

**What the server echoes.** `metadata.kv_memory_report`
(`serving_chat.py:1512`, `:1532`) is the hint itself with the six token
counters coerced to ints and `source="sglang_c2kv_runtime_injection"`
(`Scheduler._init_c2kv_kv_memory_report`, `scheduler.py:2849-2870`), plus,
after a physical eviction, `history_kv_physical_eviction` -- the whole
`HistoryKVEvictionResult` (`mem_cache/history_kv_eviction.py:30-50`) -- and the
flattened `active_history_kv_tokens`, `active_full_raw_tokens`,
`history_kv_runtime_status`, `physical_slots_freed`, `logical_total_len`,
`physical_kv_len`, `next_rope_position`, `selected_history_indices`
(`scheduler.py:2999-3015`). A failed eviction aborts the request instead of
silently serving the uncompacted prompt (`scheduler.py:3031-3034`).
`meta_info.persistent_history_session` reports `session_id`,
`canonical_prompt_tokens`, `generated_tokens`, `next_prefix_tokens` and
`generated_suffix_reprefill_required` (`serving_chat.py:233-239`).

**Interaction with section 1.** The runtime-eviction arms attach the stored
entry to a carrier message that sets `c2kv_use_gist_projection: false`
(`yuhan_client_history_kv_baselines.py:471-476`), so rule 2 resolves them to
base -- and after **D7** that explicit `false` is the *only* thing keeping
them there. The carrier carries no `c2kv_key_hash`, but the no-gist-segment
fallback that used to catch that case is gone (section 1): a runtime
history-KV arm whose carrier omits `c2kv_use_gist_projection` now resolves to
gist under the default flag and is served with gist projections. Check
`c2kv_query_proj_effective` on every such run rather than assuming base. The
`full` arm sends no C2KV annotation at all, so it never reaches the resolver
and reports `c2kv_query_proj_effective="base"` with
`c2kv_query_proj_source="none"` and an empty `c2kv_layout` (`[]`, not an
absent key, since 2026-09-05 -- section 1).

**These servers print the D9 line.** The upstream run script passes
`--enable-c2kv` (`yuhan_run_history_kv_baselines.sh:187`; client-side file, not
in this tree) and neither `--disable-cuda-graph` nor `--c2kv-query-proj`, so
`c2kv_query_proj` is its default `gist` (`server_args.py:586`) and the startup
line at `scheduler.py:899-927` fires on every such server. That is D9 behaving
as specified, not a merge defect: graph capture really is on. It says nothing
about these arms' own numbers -- their carrier message forces
`c2kv_use_gist_projection: false`, so they resolve to base and report
`c2kv_query_proj_decode_verified=true` -- but any request on that server that
resolves to gist has decode steps the mask was not rebuilt for. On an
Ascend/NPU server the same sentence is emitted at INFO with "(graph behaviour
on Ascend not verified)" appended (`scheduler.py:922-925`): the wording has not
been checked against the device's actual graph mode, so read it as "graph
capture is not disabled", not as a verified statement about NPU replay.

## 8. Things that are still open

Carried over:

- Effect size of `--c2kv-query-proj gist` vs `base`: no A/B yet.
- The `text`+`role` repair form still exists and is silently lower fidelity;
  clients in the `c2kv` repo were moved to the `messages` form.
- `c2kv_gist_seen` / `c2kv_position_correction` are not restored on
  retraction or recovery-checkpoint restore; C2KV requests are not expected
  to be retracted (`c2kv_early_process` path), but this is untested.
  `Req.c2kv_layout` is not cleared by `Req.reset_for_retract` either
  (`schedule_batch.py:1329-1367`), so a retracted-and-retried C2KV request
  would report its injections twice.
- `_c2kv_project_qkv` computes both projections for a mixed batch and
  selects per token; cost is one extra QKV GEMM on those tokens
  (`models/qwen3.py:271-313`).

Found while reconciling the two branches (2026-09-05). Items marked
**Resolved by** or **Closed** were fixed in the D5-D10 close-out and are kept
for traceability; the rest are still open:

- **Resolved by D6**, kept here for traceability: `c2kv_query_proj` used to be
  the effective mode for a request with segments and the flag for one without,
  so a turn-1 or `full`-arm request that provably ran base self-reported
  `"gist"` under the default flag. It is now three keys (section 1): the flag,
  `_effective`, and `_source` with a third value `"none"`. Residual: a
  consumer that read `c2kv_query_proj_decode_verified` as a per-RUN flag now
  sees `true` on the segment-less requests of a gist-flag run (they genuinely
  ran base). No such consumer exists in `tmp/bench-recover`
  (`benchmarks/backends/sglang.py:252-273` only records the keys into the
  per-request cost dict).
- `--c2kv-tools-dump` defaults to `full` (`server_args.py:599`), i.e. upstream's
  frame. Traces collected against `fork/task/c2kv-serve-align` were produced
  under `exclude_unset`; comparing them with a default-flag run compares two
  different prompts (+4 tokens per tool, `protocol.py:563-568`). A serve-align
  client must pass the flag explicitly.
- `c2kv_gist_projection_start_pos` is a compressed-prompt token index
  (`scheduler.py:3258-3267`) but is compared against corrected RoPE positions
  (`forward_batch_info.py:601-641`). The two frames coincide only while
  `c2kv_position_correction` is still 0 when the first segment lands; a
  repair-only segment injected before the first gist changes the correction
  (`scheduler.py:4243-4255`) and breaks the invariant. D7 widens the exposure:
  the start position is now the first C2KV segment of any kind, so a
  repair-only segment can itself be the anchor. Not covered by the smoke
  script, which puts the repair target at `target_index = 2`
  (`scripts/c2kv/smoke_c2kv_semantics.py:230`).
- **Closed, kept for traceability**: `extract_source="serving_cache"` is
  post-RoPE by construction (`scheduler.py:2765-2766`), but it no longer
  *silently* overrides a requested `pre_rope` -- the combination is rejected
  with an explicit error before extraction (`scheduler.py:2550-2557`). The
  live consequence is the opposite of the old one: because the
  `messages`/`target_index` form defaults the mode to `pre_rope`
  (`http_server.py:1663-1677`), that form plus `serving_cache` now fails unless
  the caller sends `raw_kv_position_mode="rotated"` explicitly.
- `TokenizedRepairExtractReqInput.already_rotated` defaults to `True`
  (`io_struct.py:2108`) while the response model defaults to `False`
  (`io_struct.py:2139`). Checked under D10 and left as is: `True` is the
  correct partner of the sibling default `raw_kv_position_mode="rotated"`
  (`io_struct.py:2105`) under the storage-form rule `already_rotated ==
  (raw_kv_position_mode != "pre_rope")`, and the single construction site
  always passes it explicitly (`tokenizer_communicator_mixin.py:476-478`), so
  the asymmetry is latent, not live. The docstring at
  `tokenizer_communicator_mixin.py:449-455` still describes the pre-merge rule
  and is now wrong: no HTTP field reaches the `already_rotated` kwarg, and it
  can only ever force *rotated*.
- **Resolved by D10**: an invalid `c2kv_repair_placement` is now rejected at
  admission, in the per-segment validation loop of
  `_build_c2kv_prefill_rounds` (`scheduler.py:3284-3299`), and reaches the
  client as a request abort (`scheduler.py:2127-2129`) the way
  `handle_repair_extract_request` rejects an unknown `raw_kv_position_mode`
  (`scheduler.py:2479-2484`), rather than as a per-entry injection error after
  KV has been allocated (`scheduler.py:4010-4014`).
- `metadata.sglang_runtime` is attached on the non-streaming path only
  (`serving_chat.py:1520-1548`; `ChatCompletionStreamResponse` has no
  `metadata` field, `protocol.py:955-962`, and none of the chunk builders reads
  `meta_info["kv_runtime_stats"]`): a streaming client gets no projection or
  layout provenance, so C2KV runs must be served with `stream=false`. Known and
  accepted, not a bug to rediscover.
- **Closed 2026-09-05, kept for traceability**: a batched `/generate` request
  silently lost every C2KV field. `GenerateReqInput.__getitem__`
  (`io_struct.py:635-699`) still rebuilds each item field by field and copies
  none of `c2kv_segments` / `c2kv_kv_memory_hint` /
  `c2kv_use_gist_projection`, although all three are declared on the class
  (`io_struct.py:246-250`); the per-item `TokenizedGenerateReqInput` reached
  the scheduler with `c2kv_segments=None`, the resolver and the round builder
  are both behind `if getattr(recv_req, "c2kv_segments", None):`
  (`scheduler.py:2089`), and the client got a normal 200 with no injection.
  Pre-existing in both parents (the field list is upstream's), so D4 was never
  violated -- a batched upstream request is exactly as broken on `d42ce815f`.
  The combination is now **rejected** rather than propagated
  (`io_struct._validate_c2kv_not_batched`, `io_struct.py:366-388`, called at
  `:290`; section 5). Residual: the rejection also fires for a C2KV request
  with `n > 1`, which `_handle_parallel_sampling` (`io_struct.py:340-364`)
  turns into a non-single request -- previously that combination was served as
  a plain prompt, so a client that was silently getting no injection now gets
  a 400. Single `/v1/chat/completions` requests with `n == 1` are unaffected:
  that path tokenizes the original object (`tokenizer_manager.py:518`), while
  every batched entry point goes through `obj[i]`
  (`tokenizer_manager.py:1025`, `:1030`, `:1290`, `:1303`, `:1322`).
- **Closed 2026-09-05, kept for traceability**: an injection-time failure was
  not readable by an OpenAI client. `FINISH_ABORT` without a status code
  (`scheduler_output_processor_mixin.py:431-461`) is still HTTP 200 with
  `finish_reason="abort"`, and the choice still carries only
  `finish_reason["type"]` (`serving_chat.py:1495`) -- that is deliberate (D4).
  What changed is that every injection-time failure now records a `C2KV_*`
  reason (`Scheduler._set_c2kv_injection_error`, `scheduler.py:3973-3992`),
  the three injection-time cache misses gained the `C2KV_CACHE_MISS:` prefix
  (`scheduler.py:3489`, `:3522`, `:3922`) so the client retry fires for them,
  and the reason is echoed as `metadata.sglang_runtime.c2kv_injection_error`
  (`scheduler_output_processor_mixin.py:205-207`) and `metadata.finish_message`
  (`serving_chat.py:1537-1539`). Section 3. Residual: both surfaces are
  non-streaming only, and neither is a status-code change -- a client that
  keys on HTTP status alone still sees 200 for an injection failure.
- Two `ForwardBatch` fields are declared and never assigned:
  `c2kv_position_corrections` (`forward_batch_info.py:432`) and
  `c2kv_gist_projection_start_positions` (`:438`). Every producer writes them on
  the `ModelWorkerBatch` (`schedule_batch.py:2652`, `:2654-2656`) and every real
  consumer reads them off `batch`, not `ret` (`forward_batch_info.py:601-612`,
  `:618-641`), so both are permanently `None` on `ForwardBatch`. This is
  **diagnostic-only**, not a correctness hole: the two Ascend reads of the
  former (`hardware_backend/npu/attention/ascend_backend.py:396` inside a debug
  print, `:1108-1110` as the gate for a `[C2KV QLENS CALL]` warning at
  `:1222-1226`) only suppress a log line; the position corrections themselves do
  reach the model. Either delete the two declarations or assign them in
  `ForwardBatch.init_new`; touching the NPU backend needs an owner.
- Two code comments still describe the dropped serve-align rule and say the
  mask starts at the first **gist** segment: `models/qwen3.py:281` and
  `forward_batch_info.py:617` ("pre-gist prologue"). Under D7 it starts at the
  first C2KV segment of any kind (`scheduler.py:3258-3267`). The third,
  `schedule_batch.py:2848-2852`, was corrected on 2026-09-05. Comment-only, no
  behavioural effect -- flagged because that stale mental model is exactly what
  produced the deviation D7 had to undo. A fourth, `schedule_policy.py:199`
  ("Before the first gist injection, the first round is just a normal
  real-token prefix"), is about round 0 rather than the mask and is only
  loosely worded: round 0 precedes any C2KV injection, gist or repair-only.

## 9. Reconciliation 2026-09-05

This tree is a merge of two independent C2KV serving lines, not a rewrite:

- base: `d42ce815f` "fix append" -- history-KV eviction baselines and the
  physical evictor, `c2kv_kv_memory_hint`, persistent history sessions, the
  `history_kv_*` fields on `/v1/c2kv/repair_extract`, the per-message
  `c2kv_use_gist_projection` and the `raw_kv_position_mode` storage knob;
- merged in: `fork/task/c2kv-serve-align` = `b08172044` (query-projection
  switch, explicit repair placement, full-context repair extraction, layout
  provenance, `C2KV_CACHE_MISS:` prefix, `exclude_unset` tool dump) +
  `718a654e3` (`kv_runtime_stats` carried across the detokenizer hop);
- common ancestor: `7de9e8105`.

Decisions taken (implemented; not open for re-litigation here):

- **D1** upstream's projection mechanism is the implementation
  (`Req.c2kv_use_gist_projection` + `Req.c2kv_gist_projection_start_pos` -> one
  per-token mask); serve-align's `--c2kv-query-proj` becomes the per-request
  **default**, and a message-level field overrides it (section 1). There is one
  mask, not two: `c2kv_gist_proj_mask` is gone.
- **D2** `raw_kv_position_mode` (storage form) and `c2kv_repair_placement`
  (ledger placement) are orthogonal and both survive; explicit placement beats
  the legacy `repair_mode` rule, and `append_tail` on an already-rotated entry
  is refused rather than silently downgraded (section 3 -- with the caveat,
  documented there, that over `/v1/chat/completions` this stays an HTTP 200
  abort; since 2026-09-05 the reason string itself does reach the client, on
  `metadata.sglang_runtime.c2kv_injection_error` / `metadata.finish_message`).
- **D3** everything else from both sides is kept: upstream's history-KV
  eviction, persistent sessions and KV-memory reporting (section 7);
  serve-align's full-context repair extraction, `tools` on the extract
  endpoints, layout provenance and error prefixes (sections 3 and 5).
- **D4** a request that sends none of the serve-align fields must behave as it
  did on `d42ce815f`, and a request from the bench proxy must get serve-align
  behaviour. The one place the two frames genuinely differ is
  `--c2kv-tools-dump` (section 2): its default `full` is upstream's frame, so a
  serve-align client has to ask for `exclude_unset` explicitly.
- **D5** `--c2kv-tools-dump {full,exclude_unset}` stays, default `full`: the
  shared default is upstream's frame (D4 beats D3 here), and the flag is
  honoured on all three renderings that define the token frame -- the chat
  prompt (`serving_chat._chat_template_tools`, `serving_chat.py:357-388`),
  `/v1/c2kv/extract` with `tools` (`http_server.py:1513`) and the `messages`
  form of `/v1/c2kv/repair_extract` (`http_server.py:1564`), both endpoints
  through the same `http_server._c2kv_flat_tools` (`:1460-1485`) -- so extract
  == chat == repair frame under either value, with the four documented
  exceptions listed in section 2. The active value is echoed as
  `metadata.sglang_runtime.c2kv_tools_dump`
  (`scheduler_output_processor_mixin.py:114`). A serve-align client must pass
  `--c2kv-tools-dump exclude_unset` explicitly (the bench launcher does:
  `tmp/bench-recover/benchmarks/ops/launch_sgl1088.sh:69`).
- **D6** projection provenance is three keys, not one: `c2kv_query_proj` (the
  server flag, constant per run), `c2kv_query_proj_effective` (what this
  request ran) and `c2kv_query_proj_source` (`"message"` / `"flag"` /
  `"none"`). Section 1.
- **D7** repair-only segments follow upstream: once the resolved mode is gist,
  the mask starts at `segments[0].token_start` whether that segment is a gist
  or a repair-only one. Serve-align's `c2kv_gist_seen`-keyed fallback to base
  is dropped, and with it the guarantee that a fully-raw-repair history is
  served on base projections. Section 1 spells out which bench arm this
  changes.
- **D8** `c2kv_use_gist_projection` travels as a tri-state (`None` = unset =>
  the flag decides) the whole way: `protocol.py:506`/`:534` ->
  `serving_chat.py:588-590`/`:788-790` -> `io_struct.py:246-250`/`:787-791` ->
  `tokenizer_manager.py:992-994` -> the resolver at `scheduler.py:2100`. A
  `getattr(..., False)` collapse anywhere on that path makes
  `--c2kv-query-proj` unreachable and is a bug, not a default. The
  `getattr(..., False)` reads that remain (`schedule_batch.py:2544`, `:2546`,
  `scheduler_output_processor_mixin.py:155`, `scheduler.py:3266`) are all on
  `Req` *after* resolution, where the value is a real bool.
- **D9** the startup line fires only where the flag path actually loses the
  mask on decode replay: `--enable-c2kv` (`scheduler.py:851`) and
  `--c2kv-query-proj gist` and graph capture on (`scheduler.py:903-906`, inside
  the block at `:899-927`). A `--disable-cuda-graph` server is silent. The gate
  reads the flag, not the resolved per-request mode -- section 1 says what that
  leaves uncovered. Its wording was narrowed on 2026-09-05 to the four clauses
  the code supports and demoted to INFO on `--device npu`
  (`scheduler.py:913-925`); the "must not be used as one side of a
  gist-vs-base A/B" verdict was removed.
- **D10** hygiene: an invalid `c2kv_repair_placement` is rejected at admission
  (`scheduler.py:3284-3299`) instead of at injection;
  `TokenizedRepairExtractReqInput.already_rotated` was audited against its
  construction site and left unchanged (section 8). Since 2026-09-05 the
  admission error carries the same `C2KV_REPAIR_PLACEMENT_INVALID:` code as
  the injection-time backstop (`scheduler.py:3296` and `:4043`), so a client
  keys on one string whichever channel refuses it.

Closed after D1-D10, same day, same evidence standard (**S1**-**S7**):
every injection-time failure records a `C2KV_*` reason and the reason reaches
an OpenAI client on two extra metadata keys, with no status-code change (S1,
section 3); the `c2kv_layout` ledger keys are always present on a
C2KV-enabled server and empty means "nothing injected" (S2, section 1);
batched `/generate` requests carrying C2KV fields are rejected instead of
silently served as plain prompts (S3, section 5); the D9 startup line says
only what the code shows and is INFO on NPU (S4, section 1); the stale
"first gist segment" comment in `schedule_batch.py` is corrected (S5,
`schedule_batch.py:2848-2852`); the smoke script covers the D6 keys, the
empty-ledger contract and the placement rejection (S6, section 6).

**Unverified.** No part of this tree has been run against a model or a server
since the merge: no model load, no server start, no forward pass, and in
particular **no `scripts/c2kv/smoke_c2kv_semantics.py` run** -- there is no NPU
and no checkpoint in the environment the merge was resolved in. Verification so
far is limited to conflict-marker checks (0 markers), `python -m py_compile` on
the changed files, and reading the merged functions. Every behavioural
statement in this file, every `file:line` in it, and every claim about what a
response contains is a claim about code as read on 2026-09-05, not an
observation of a run.

Before quoting any number produced by this build, on the NPU box:

1. run the smoke script (section 6) against a live server, and read section 6
   on what it does *not* assert -- a pass still does not cover the D7
   repair-only mask start, and nothing ties the reported `c2kv_query_proj` to
   the flag the server was started with;
2. check the startup log for the D9 line and record whether it fired (and at
   which level -- INFO on `--device npu`), next to the `--c2kv-query-proj` and
   `--c2kv-tools-dump` values the server was started with;
3. on one real request of each arm, record `c2kv_query_proj`,
   `c2kv_query_proj_effective`, `c2kv_query_proj_source`,
   `c2kv_query_proj_decode_verified`, `c2kv_tools_dump`, whether `c2kv_layout`
   is `[]` or populated, and whether `c2kv_injection_error` is present -- that
   is the whole reconciled contract in one response.

The open items in section 8 are the ones already known to be wrong or fragile;
they are read off the code too, and none of them has been reproduced on a
server either.
