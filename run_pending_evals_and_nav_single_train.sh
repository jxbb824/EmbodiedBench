#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

DISPLAY_NUM="${DISPLAY_NUM:-1}"
VLLM_PORT="${VLLM_PORT:-23333}"
EVAL_GPU="${EVAL_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"

ALF_ENV="${ALF_ENV:-embench}"
NAV_ENV="${NAV_ENV:-embench_nav}"
TRAIN_ENV="${TRAIN_ENV:-hf}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3.5-9B}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3.5-Base}"
ALF_SFT_MODEL_PATH="${ALF_SFT_MODEL_PATH:-${ROOT_DIR}/sft_output/alf_0426_1540/merged}"
ALF_SFT_MODEL_NAME="${ALF_SFT_MODEL_NAME:-Qwen3.5-Alf-SFT}"

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

cleanup_x_display() {
  local display="$1"
  pkill -TERM -f "Xorg .*:${display}( |$)" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "Xorg .*:${display}( |$)" 2>/dev/null || true
  rm -f "/tmp/.X${display}-lock" "/tmp/.X11-unix/X${display}" 2>/dev/null || true
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

cleanup() {
  echo "[cleanup] stopping background services..."
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
  stop_server
  if [[ -n "${X_PID}" ]] && kill -0 "${X_PID}" 2>/dev/null; then
    kill "${X_PID}" 2>/dev/null || true
    wait "${X_PID}" 2>/dev/null || true
  fi
  cleanup_x_display "${DISPLAY_NUM}"
}
trap cleanup EXIT INT TERM

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
      echo "[error] vLLM server exited. See current server log in ${LOG_DIR}"
      return 1
    fi
    sleep "${delay}"
  done
  echo "[error] vLLM server timed out. See current server log in ${LOG_DIR}"
  return 1
}

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local log_name="$3"

  echo "[start] vLLM ${served_name} on GPU ${EVAL_GPU}"
  activate_env "${TRAIN_ENV}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" vllm serve "${model_path}" \
    --served-model-name "${served_name}" \
    --host 0.0.0.0 --port "${VLLM_PORT}" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-model-len 32768 \
    >"${LOG_DIR}/${log_name}" 2>&1 &
  SERVER_PID=$!
  wait_for_server
}

run_eval() {
  local env_name="$1"
  local conda_env="$2"
  local model_name="$3"
  local exp_name="$4"
  local log_name="$5"

  echo "[run] ${env_name} eval with ${model_name}"
  activate_env "${conda_env}"
  export DISPLAY=":${DISPLAY_NUM}"
  export remote_url="http://127.0.0.1:${VLLM_PORT}/v1"
  python -m embodiedbench.main \
    env="${env_name}" \
    model_name="${model_name}" \
    model_type=remote \
    exp_name="${exp_name}" \
    >"${LOG_DIR}/${log_name}" 2>&1
}

cd "${ROOT_DIR}"

if [[ ! -d "${ALF_SFT_MODEL_PATH}" ]]; then
  echo "[error] Missing merged ALF SFT model: ${ALF_SFT_MODEL_PATH}"
  echo "        Merge it first with:"
  echo "        CUDA_VISIBLE_DEVICES=${EVAL_GPU} python sft/export_model.py --lora sft_output/alf_0426_1540/lora --output sft_output/alf_0426_1540/merged"
  exit 1
fi

echo "[start] X server on GPU ${EVAL_GPU}, DISPLAY=:${DISPLAY_NUM}"
activate_env "${NAV_ENV}"
cleanup_x_display "${DISPLAY_NUM}"
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" python -m embodiedbench.envs.eb_alfred.scripts.startx "${DISPLAY_NUM}" \
  >"${LOG_DIR}/pending_startx.log" 2>&1 &
X_PID=$!
sleep 5

echo "[start] Nav single-step successful-path SFT training on GPU ${TRAIN_GPU}"
activate_env "${TRAIN_ENV}"
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python sft/train.py \
  --task nav \
  --step-mode single \
  --max-episodes 0 \
  >"${LOG_DIR}/nav_single_sft_train.log" 2>&1 &
TRAIN_PID=$!

start_vllm "${BASE_MODEL_PATH}" "${BASE_MODEL_NAME}" "base_repo_server.log"
run_eval "eb-alf" "${ALF_ENV}" "${BASE_MODEL_NAME}" "repo_tuned_base" "eb_alf_repo_tuned_base_eval.log"
run_eval "eb-nav" "${NAV_ENV}" "${BASE_MODEL_NAME}" "repo_tuned_base" "eb_nav_repo_tuned_base_eval.log"
stop_server

start_vllm "${ALF_SFT_MODEL_PATH}" "${ALF_SFT_MODEL_NAME}" "alf_sft_server.log"
run_eval "eb-alf" "${ALF_ENV}" "${ALF_SFT_MODEL_NAME}" "alf_sft_vllm" "eb_alf_sft_eval.log"
stop_server

echo "[done] pending evals finished; waiting for Nav single-step SFT training."
wait "${TRAIN_PID}" 2>/dev/null || true
echo "[done] all tasks finished. Logs are in ${LOG_DIR}"
