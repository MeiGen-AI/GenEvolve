#!/usr/bin/env bash
# Serve the released GenEvolve checkpoint with vLLM (OpenAI-compatible).
#
# Edit MODEL_PATH to point at the released GenEvolve checkpoint
# (Qwen3-VL-8B-based; download from the project page or HuggingFace).
#
# Then run the agent against http://localhost:8000/v1.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH to the GenEvolve checkpoint directory}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GenEvolve}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
DP="${DP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --data-parallel-size "${DP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${MEM_FRACTION}" \
  --trust-remote-code
