#!/usr/bin/env bash
# Serve the released GenEvolve checkpoint with SGLang (OpenAI-compatible).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH to the GenEvolve checkpoint directory}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GenEvolve-8B}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tp "${TP}" \
  --context-length "${MAX_MODEL_LEN}" \
  --trust-remote-code
