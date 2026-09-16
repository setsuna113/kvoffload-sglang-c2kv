#!/usr/bin/env bash
# Eager CUDA lane for C2KV native generation. CKPT is a protocol identity:
# preserve its spelling (including a symlink alias) when replaying saved handles.
set -euo pipefail

: "${CKPT:?Set CKPT to the exact model_path used by the controller or frozen journal}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="${SGLANG_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-36100}"
MAX_TOKENS="${MAX_TOKENS:-12800}"
MEM_FRAC="${MEM_FRAC:-0.85}"

if [[ ! -f "${CKPT}/config.json" ]]; then
    printf 'Checkpoint config not found: %s/config.json\n' "$CKPT" >&2
    exit 2
fi
if [[ "${PAGE_SIZE:-1}" != 1 ]]; then
    printf 'The CUDA native C2KV lane requires PAGE_SIZE=1.\n' >&2
    exit 2
fi

export PYTHONPATH="${SGLANG_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

exec "$PYTHON_BIN" -m sglang.launch_server \
    --model-path "$CKPT" --served-model-name "${SERVED_MODEL_NAME:-d3-c1000}" \
    --model-impl sglang --device cuda --dtype bfloat16 \
    --attention-backend torch_native \
    --enable-c2kv --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
    --c2kv-query-proj base --c2kv-pool-fraction 0.05 \
    --c2kv-shadow-feature-layer -2 --enable-return-hidden-states \
    --mem-fraction-static "$MEM_FRAC" --max-total-tokens "$MAX_TOKENS" \
    --context-length 131072 --max-running-requests 1 --page-size 1 \
    --chunked-prefill-size 256 --disable-radix-cache --disable-cuda-graph \
    --disable-overlap-schedule --disable-piecewise-cuda-graph \
    --host 127.0.0.1 --port "$PORT"
