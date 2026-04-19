#!/bin/bash
# Start a local vLLM server for interactive testing on a GPU machine.
# Usage:
#   bash jobs/start_vllm_local.sh Qwen/Qwen3-8B
#
# Environment variables you may override:
#   LLM_PORT=8000
#   GPU_MEMORY_UTILIZATION=0.90
#   MAX_MODEL_LEN=8192

set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen3-8B}"
LLM_PORT="${LLM_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

export LLM_BASE_URL="http://127.0.0.1:${LLM_PORT}/v1"
export LLM_MODEL="${MODEL_NAME}"
export LLM_API_KEY="local-token"

vllm serve "${MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${LLM_PORT}" \
  --api-key "${LLM_API_KEY}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}"
