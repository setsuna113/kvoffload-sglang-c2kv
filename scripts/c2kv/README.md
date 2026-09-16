# Native C2KV on CUDA

`launch_cuda_native.sh` provides the eager `torch_native` CUDA lane for the
native packed generation endpoint. Use a C2KV checkpoint, the matching fork,
and a CUDA Python environment with the fork's dependencies installed. This is
not a throughput configuration or a guarantee of cross-device bitwise equality.

## Launch

```bash
CKPT=/absolute/checkpoint/path \
PYTHON_BIN=/absolute/venv/bin/python \
CUDA_HOME=/opt/cuda \
bash scripts/c2kv/launch_cuda_native.sh
```

`CKPT` is required and is passed verbatim. Native chunk handles include the
model-path string in their identity. When replaying a frozen journal, use the
exact `shadow_features.bindings.model.model_path` recorded in its responses;
an existing symlink can supply that alias. Passing the resolved target instead
changes the identity even if the weight files are identical. A fresh controller
run should instead use the path expected by that controller (`Path.resolve()`
in the current event-native runner).

The matching event-native adapter exposes `decode_strategy="incremental"` in
its health response and translates controller `seed` to SGLang `sampling_seed`.
Use the corrected adapter when starting a fresh BFCL run; old frozen runtime
archives can predate these fixes even when their saved journals are valid.

Defaults are `MAX_TOKENS=12800`, `MEM_FRAC=0.85`, `PORT=36100`, and
`PAGE_SIZE=1`. Memory requirements depend on the checkpoint and hardware.
The lane fixes one running request, disables radix caching, CUDA graphs,
piecewise graphs and overlap scheduling. CUDA native C2KV currently requires
token-granularity allocation; the launcher rejects `PAGE_SIZE` other than `1`.

## Replay and retain evidence

```bash
python scripts/c2kv/replay_native_journal.py \
  --journal /absolute/frozen/sglang_http.jsonl \
  --base-url http://127.0.0.1:36100 \
  --out /absolute/results/replay.json \
  --controller-config /absolute/history_system/runtime/configs/controller.json \
  --runtime-root /absolute/history_system/runtime
```

Keep the controller config and runtime from the reference run. The replay
stores the original requests, reference responses, actual responses and errors,
so results can be recomputed independently. All expected tokens remain in the
denominator, including requests that fail. Malformed journals, transport errors,
token/text/finish mismatches, invalid requested features, and detector decision
mismatches fail the command. Hidden-state numerical differences are reported;
the detector comparison uses the unchanged reference head and threshold.

Replay validates engine responses for the supplied inputs. It does not execute
the BFCL environment, controller decisions or official scorer. Complete BFCL
validation additionally requires running the matching event-native controller
and official worker, checking successful engine requests and controller
decisions, and comparing official task results. Scorer completion alone is
insufficient: BFCL can score a failed server request as an incorrect answer.
A single-task run is a functional smoke test (`preliminary, n=1`), not
a full-benchmark quality or speed result.

## Tests and source versions

Run tests from the same checkout as the engine. Do not copy newer tests onto an
older deployment snapshot and then exclude their failures from the pass count.

```bash
PYTHONPATH="$PWD/python" python -m pytest -q \
  test/registered/unit/test_c2kv_journal_replay.py \
  test/registered/unit/test_c2kv_cuda_launcher.py \
  test/registered/unit/test_c2kv_native_packed.py \
  test/registered/unit/test_c2kv_hidden_capture_mode.py
```

BFCL's `num_threads=1` and the server's single-request limit constrain scheduling;
neither promises identical tokens across hardware or library versions. Artifact
hashes establish file identity, not cross-device numerical equivalence.
