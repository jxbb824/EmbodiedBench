#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
DISPLAY_NUM="${DISPLAY_NUM:-1}"
VLLM_PORT="${VLLM_PORT:-23333}"
X_GPU="${X_GPU:-0}"
VLLM_GPU="${VLLM_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"
NAV_MODEL_PATH="${NAV_MODEL_PATH:-${ROOT_DIR}/sft_output/nav_0426_0040/merged}"
NAV_MODEL_NAME="${NAV_MODEL_NAME:-Qwen3.5-Nav-SFT}"
mkdir -p "${LOG_DIR}"

source /home/xirui/miniconda3/etc/profile.d/conda.sh

X_PID=""
SERVER_PID=""
TRAIN_PID=""

activate_env() {
  set +u
  conda activate "$1"
  set -u
}

cleanup() {
  echo "[cleanup] stopping background services..."
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${X_PID}" ]] && kill -0 "${X_PID}" 2>/dev/null; then
    kill "${X_PID}" 2>/dev/null || true
    wait "${X_PID}" 2>/dev/null || true
  fi
  cleanup_x_display "${DISPLAY_NUM}"
}
trap cleanup EXIT INT TERM

cleanup_x_display() {
  local display="$1"
  pkill -TERM -f "Xorg .*:${display}( |$)" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "Xorg .*:${display}( |$)" 2>/dev/null || true
  rm -f "/tmp/.X${display}-lock" "/tmp/.X11-unix/X${display}" 2>/dev/null || true
}

wait_for_server() {
  local retries=120
  local delay=5
  for ((i = 1; i <= retries; i++)); do
    if python - <<PY >/dev/null 2>&1
import socket
with socket.create_connection(("127.0.0.1", ${VLLM_PORT}), timeout=1):
    pass
PY
    then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[error] vLLM server exited. See ${LOG_DIR}/qwen_nav_sft_server.log"
      return 1
    fi
    sleep "${delay}"
  done
  echo "[error] vLLM server timed out. See ${LOG_DIR}/qwen_nav_sft_server.log"
  return 1
}

cd "${ROOT_DIR}"

# ── Step 1: X server for AI2-THOR navigation rendering ─────────────────────
echo "[start] launching X server on GPU ${X_GPU}, DISPLAY=:${DISPLAY_NUM}..."
activate_env embench
cleanup_x_display "${DISPLAY_NUM}"
CUDA_VISIBLE_DEVICES="${X_GPU}" python -m embodiedbench.envs.eb_alfred.scripts.startx "${DISPLAY_NUM}" \
  >"${LOG_DIR}/alfred_startx.log" 2>&1 &
X_PID=$!
sleep 5

# ── Step 2: vLLM serve Nav SFT model ────────────────────────────────────────
echo "[start] launching vLLM server (Nav SFT model) on GPU ${VLLM_GPU}..."
activate_env hf
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" vllm serve "${NAV_MODEL_PATH}" \
  --served-model-name "${NAV_MODEL_NAME}" \
  --host 0.0.0.0 --port "${VLLM_PORT}" \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --max-model-len 32768 \
  >"${LOG_DIR}/qwen_nav_sft_server.log" 2>&1 &
SERVER_PID=$!

# ── Step 3: ALFRED multi-step SFT training on the other GPU ────────────────
echo "[start] launching ALFRED multi-step SFT training on GPU ${TRAIN_GPU} (background)..."
activate_env hf
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python sft/train.py \
  --task alf \
  --step-mode multi \
  --max-episodes 0 \
  >"${LOG_DIR}/alf_multi_sft_train.log" 2>&1 &
TRAIN_PID=$!

echo "[wait] waiting for vLLM server..."
wait_for_server

# ── Step 4: EB-Navigation evaluation with Nav SFT model ────────────────────
echo "[run] starting EB-Navigation evaluation with Nav SFT model..."
activate_env embench_nav
export DISPLAY=":${DISPLAY_NUM}"
export remote_url="http://127.0.0.1:${VLLM_PORT}/v1"
python -m embodiedbench.main \
  env=eb-nav \
  model_name="${NAV_MODEL_NAME}" \
  model_type=remote \
  exp_name=nav_sft_vllm \
  >"${LOG_DIR}/eb_nav_sft_eval.log" 2>&1

echo "[done] EB-Navigation SFT evaluation finished."
echo "[done] ALFRED multi-step SFT training should still be running (or finished) in background."
wait "${TRAIN_PID}" 2>/dev/null || true
echo "[done] All tasks finished. Logs in ${LOG_DIR}"
