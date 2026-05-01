#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
DISPLAY_NUM="${DISPLAY_NUM:-1}"
X_GPU="${X_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"
VLLM_GPU="${VLLM_GPU:-1}"
VLLM_PORT="${VLLM_PORT:-23333}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3.5-9B}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-Nav-RL}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/rl_data/nav_grpo_states.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/rl_output/nav_grpo}"
MERGED_DIR="${MERGED_DIR:-${OUTPUT_DIR}/merged}"

STEP_MODE="${STEP_MODE:-multi}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
INCLUDE_FAILED="${INCLUDE_FAILED:-1}"
IMAGE_ROOTS="${IMAGE_ROOTS:-sft_data/EB-Nav_trajectory_dataset/images/images/claude-3-5-sonnet-20241022_additional_nav_no_c_h,sft_data/EB-Nav_trajectory_dataset/images/images/claude-3-7-sonnet-20250219_no_c_h,sft_data/EB-Nav_trajectory_dataset/images/images/gemini-1.5-pro_additional_nav_no_c_h,sft_data/EB-Nav_trajectory_dataset/images/images/gpt-4o_additional_nav_no_c_h,sft_data/EB-Nav_trajectory_dataset/images/images/qwen-vl-max-2025-01-25_navigation_baseline}"

MAX_STEPS="${MAX_STEPS:-850}"
SAVE_STEPS="${SAVE_STEPS:-200}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-8e-7}"
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-0.95}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-7168}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_PLAN_ACTIONS="${MAX_PLAN_ACTIONS:-3}"
REWARD_EXP_NAME="${REWARD_EXP_NAME:-nav_grpo_reward}"
EXP_NAME="${EXP_NAME:-nav_rl_vllm}"
SMOKE="${SMOKE:-0}"
SKIP_DATASET="${SKIP_DATASET:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "${LOG_DIR}" "$(dirname "${DATASET_PATH}")" "${OUTPUT_DIR}"
source /home/xirui/miniconda3/etc/profile.d/conda.sh

X_PID=""
SERVER_PID=""

activate_env() {
  set +u
  conda activate "$1"
  set -u
}

cleanup_x_display() {
  local display="$1"
  pkill -TERM -f "[X]org .*:${display}( |$)" 2>/dev/null || true
  sleep 2
  pkill -KILL -f "[X]org .*:${display}( |$)" 2>/dev/null || true
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
      echo "[error] vLLM server exited early. See ${LOG_DIR}/nav_grpo_server.log"
      return 1
    fi
    sleep "${delay}"
  done
  echo "[error] vLLM server timed out. See ${LOG_DIR}/nav_grpo_server.log"
  return 1
}

cd "${ROOT_DIR}"

if [[ "${SKIP_DATASET}" != "1" ]]; then
  echo "[data] building EB-Nav ${STEP_MODE} GRPO states -> ${DATASET_PATH}"
  activate_env hf
  DATA_ARGS=(--step-mode "${STEP_MODE}" --output "${DATASET_PATH}" --max-examples "${MAX_EXAMPLES}")
  if [[ -n "${IMAGE_ROOTS}" ]]; then
    DATA_ARGS+=(--image-roots "${IMAGE_ROOTS}")
  fi
  if [[ "${INCLUDE_FAILED}" == "1" ]]; then
    DATA_ARGS+=(--all-episodes)
  fi
  python rl/nav_state_dataset.py "${DATA_ARGS[@]}" >"${LOG_DIR}/nav_grpo_data.log" 2>&1
fi

echo "[start] X server on GPU ${X_GPU}, DISPLAY=:${DISPLAY_NUM}"
activate_env embench_nav
cleanup_x_display "${DISPLAY_NUM}"
CUDA_VISIBLE_DEVICES="${X_GPU}" python -m embodiedbench.envs.eb_alfred.scripts.startx "${DISPLAY_NUM}" \
  >"${LOG_DIR}/nav_grpo_startx.log" 2>&1 &
X_PID=$!
sleep 5

if ! kill -0 "${X_PID}" 2>/dev/null; then
  echo "[error] X server exited early. See ${LOG_DIR}/nav_grpo_startx.log"
  exit 1
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[train] Nav GRPO from ${BASE_MODEL_PATH} on GPU ${TRAIN_GPU}"
  activate_env hf
  TRAIN_ARGS=(
    --dataset "${DATASET_PATH}"
    --model "${BASE_MODEL_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --max-steps "${MAX_STEPS}"
    --save-steps "${SAVE_STEPS}"
    --num-generations "${NUM_GENERATIONS}"
    --grad-accum "${GRAD_ACCUM}"
    --learning-rate "${LEARNING_RATE}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-prompt-length "${MAX_PROMPT_LENGTH}"
    --max-completion-length "${MAX_COMPLETION_LENGTH}"
    --max-plan-actions "${MAX_PLAN_ACTIONS}"
    --reward-exp-name "${REWARD_EXP_NAME}"
    --reward-worker-cmd "env -u CUDA_VISIBLE_DEVICES DISPLAY=:${DISPLAY_NUM} conda run --no-capture-output -n embench_nav python -m rl.nav_reward_worker"
  )
  if [[ "${SMOKE}" == "1" ]]; then
    TRAIN_ARGS+=(--smoke)
  fi
  DISPLAY=":${DISPLAY_NUM}" CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python rl/train_nav_grpo.py "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/nav_grpo_train.log"
fi

if [[ "${SKIP_MERGE}" != "1" ]]; then
  echo "[merge] ${OUTPUT_DIR}/lora -> ${MERGED_DIR}"
  activate_env hf
  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" python sft/export_model.py \
    --lora "${OUTPUT_DIR}/lora" \
    --output "${MERGED_DIR}" \
    2>&1 | tee "${LOG_DIR}/nav_grpo_merge.log"
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  echo "[serve] ${MERGED_DIR} as ${MODEL_NAME} on GPU ${VLLM_GPU}"
  activate_env hf
  CUDA_VISIBLE_DEVICES="${VLLM_GPU}" vllm serve "${MERGED_DIR}" \
    --served-model-name "${MODEL_NAME}" \
    --host 0.0.0.0 --port "${VLLM_PORT}" \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-model-len 32768 \
    >"${LOG_DIR}/nav_grpo_server.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server

  echo "[eval] EB-Nav with ${MODEL_NAME}, exp=${EXP_NAME}"
  activate_env embench_nav
  export DISPLAY=":${DISPLAY_NUM}"
  export remote_url="http://127.0.0.1:${VLLM_PORT}/v1"
  python -m embodiedbench.main \
    env=eb-nav \
    model_name="${MODEL_NAME}" \
    model_type=remote \
    exp_name="${EXP_NAME}" \
    >"${LOG_DIR}/eb_nav_rl_eval.log" 2>&1
fi

echo "[done] Nav GRPO train/merge/eval finished."
