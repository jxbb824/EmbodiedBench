#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
DISPLAY_NUM="${DISPLAY_NUM:-1}"
X_GPU="${X_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-0}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/sft_output/alf_0426_1540/lora}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/rl_data/alfred_grpo_states.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/rl_output/alf_grpo}"
STEP_MODE="${STEP_MODE:-multi}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
INCLUDE_FAILED="${INCLUDE_FAILED:-1}"
IMAGE_ROOTS="${IMAGE_ROOTS:-sft_data/EB-Alfred_trajectory_dataset/images/images/claude-3-5-sonnet-20241022_vlm_subset_10shots_imgsize500,sft_data/EB-Alfred_trajectory_dataset/images/images/claude-3-7-sonnet-20250219_vlm_10shots_imgsize500,sft_data/EB-Alfred_trajectory_dataset/images/images/gemini-1.5-pro_vlm_subset_10shots_imgsize500,sft_data/EB-Alfred_trajectory_dataset/images/images/gpt-4o_vlm_subset_10shots_imgsize500,sft_data/EB-Alfred_trajectory_dataset/images/images/qwen-vl-max-2025-01-25_vlm_10shots_imgsize500}"

MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-200}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-8e-7}"
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-0.95}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-7168}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_PLAN_ACTIONS="${MAX_PLAN_ACTIONS:-3}"
REWARD_EXP_NAME="${REWARD_EXP_NAME:-alf_grpo_reward}"
SMOKE="${SMOKE:-0}"
SKIP_DATASET="${SKIP_DATASET:-0}"

mkdir -p "${LOG_DIR}" "$(dirname "${DATASET_PATH}")" "${OUTPUT_DIR}"
source /home/xirui/miniconda3/etc/profile.d/conda.sh

X_PID=""

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

cleanup() {
  echo "[cleanup] stopping X server on DISPLAY=:${DISPLAY_NUM}..."
  if [[ -n "${X_PID}" ]] && kill -0 "${X_PID}" 2>/dev/null; then
    kill "${X_PID}" 2>/dev/null || true
    wait "${X_PID}" 2>/dev/null || true
  fi
  cleanup_x_display "${DISPLAY_NUM}"
}
trap cleanup EXIT INT TERM

cd "${ROOT_DIR}"

if [[ "${SKIP_DATASET}" != "1" ]]; then
  echo "[data] building ALFRED ${STEP_MODE} GRPO states -> ${DATASET_PATH}"
  activate_env hf
  DATA_ARGS=(
    --step-mode "${STEP_MODE}"
    --output "${DATASET_PATH}"
    --max-examples "${MAX_EXAMPLES}"
  )
  if [[ -n "${IMAGE_ROOTS}" ]]; then
    DATA_ARGS+=(--image-roots "${IMAGE_ROOTS}")
  fi
  if [[ "${INCLUDE_FAILED}" == "1" ]]; then
    DATA_ARGS+=(--all-episodes)
  fi
  python rl/alfred_state_dataset.py \
    "${DATA_ARGS[@]}" \
    >"${LOG_DIR}/alf_grpo_data.log" 2>&1
fi

echo "[start] launching ALFRED X server on GPU ${X_GPU}, DISPLAY=:${DISPLAY_NUM}"
activate_env embench
cleanup_x_display "${DISPLAY_NUM}"
CUDA_VISIBLE_DEVICES="${X_GPU}" python -m embodiedbench.envs.eb_alfred.scripts.startx "${DISPLAY_NUM}" \
  >"${LOG_DIR}/alf_grpo_startx.log" 2>&1 &
X_PID=$!
sleep 5

if ! kill -0 "${X_PID}" 2>/dev/null; then
  echo "[error] X server exited early. See ${LOG_DIR}/alf_grpo_startx.log"
  exit 1
fi

TRAIN_ARGS=(
  --dataset "${DATASET_PATH}"
  --model "${MODEL_PATH}"
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
)

if [[ "${SMOKE}" == "1" ]]; then
  TRAIN_ARGS+=(--smoke)
fi

echo "[run] starting ALFRED GRPO on GPU ${TRAIN_GPU}; logs -> ${LOG_DIR}/alf_grpo_train.log"
activate_env hf
DISPLAY=":${DISPLAY_NUM}" CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python rl/train_alf_grpo.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/alf_grpo_train.log"

echo "[done] ALFRED GRPO finished. LoRA should be in ${OUTPUT_DIR}/lora"
