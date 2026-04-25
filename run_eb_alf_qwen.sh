#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
REMOTE_URL="${REMOTE_URL:-http://127.0.0.1:23333/v1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "${LOG_DIR}"

source /home/xirui/miniconda3/etc/profile.d/conda.sh

X_PID=""
SERVER_PID=""

activate_env() {
  set +u
  conda activate "$1"
  set -u
}

cleanup() {
  echo "[cleanup] stopping background services..."
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${X_PID}" ]] && kill -0 "${X_PID}" 2>/dev/null; then
    kill "${X_PID}" 2>/dev/null || true
    wait "${X_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local retries=120
  local delay=5

  for ((i = 1; i <= retries; i++)); do
    if python - <<'PY' >/dev/null 2>&1
import socket

with socket.create_connection(("127.0.0.1", 23333), timeout=1):
    pass
PY
    then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[error] vLLM server exited before becoming ready. See ${LOG_DIR}/qwen_server.log"
      return 1
    fi
    sleep "${delay}"
  done

  echo "[error] vLLM server did not become ready after $((retries * delay)) seconds. See ${LOG_DIR}/qwen_server.log"
  return 1
}

cd "${ROOT_DIR}"

echo "[start] launching ALFRED X server on DISPLAY=:1..."
activate_env embench
python -m embodiedbench.envs.eb_alfred.scripts.startx 1 >"${LOG_DIR}/alfred_startx.log" 2>&1 &
X_PID=$!

sleep 5

echo "[start] launching vLLM server on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} with MODEL_PATH=${MODEL_PATH}..."
activate_env hf
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 --port 23333 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-model-len 8192 \
  >"${LOG_DIR}/qwen_server.log" 2>&1 &
SERVER_PID=$!

echo "[wait] waiting for Qwen server..."
wait_for_server

echo "[run] starting EB-ALFRED evaluation..."
activate_env embench
export remote_url="${REMOTE_URL}"
python -m embodiedbench.main env=eb-alf model_name="${MODEL_PATH}" model_type=remote exp_name=vllm_baseline >"${LOG_DIR}/eb_alf_eval.log" 2>&1

echo "[done] EB-ALFRED evaluation finished. Logs are in ${LOG_DIR}"
