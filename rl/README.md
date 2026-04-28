# ALFRED GRPO

This is a lightweight RL path for ALFRED planner actions. It does not modify the
environment or evaluator. The reward function resets `EBAlfEnv`, replays the
stored action history for a trajectory state, executes the model JSON plan, and
scores JSON validity, invalid actions, task progress, and task success.

## 1. Build replay states

```bash
python rl/alfred_state_dataset.py \
  --step-mode multi \
  --output rl_data/alfred_grpo_states.jsonl \
  --max-examples 2000
```

## 2. Smoke test

Run this first. It only trains for two GRPO steps and catches most integration
issues.

Start the ALFRED X server in the `embench` environment. `EBAlfEnv` uses display
`:1`.

```bash
source /home/xirui/miniconda3/etc/profile.d/conda.sh
conda activate embench
CUDA_VISIBLE_DEVICES=0 python -m embodiedbench.envs.eb_alfred.scripts.startx 1 \
  > logs/alf_grpo_startx.log 2>&1 &
```

Run training from the `hf` environment:

```bash
conda activate hf
DISPLAY=:1 \
CUDA_VISIBLE_DEVICES=0 python rl/train_alf_grpo.py \
  --dataset rl_data/alfred_grpo_states.jsonl \
  --model sft_output/alf_0426_1540/lora \
  --output-dir rl_output/alf_grpo_smoke \
  --smoke
```

## 3. Train

```bash
DISPLAY=:1 \
CUDA_VISIBLE_DEVICES=0 python rl/train_alf_grpo.py \
  --dataset rl_data/alfred_grpo_states.jsonl \
  --model sft_output/alf_0426_1540/lora \
  --output-dir rl_output/alf_grpo \
  --max-steps 200 \
  --num-generations 4 \
  --grad-accum 4 \
  --max-prompt-length 7168 \
  --max-completion-length 1024
```

Use one GPU for this first version. The reward calls AI2-THOR directly and
expects the normal ALFRED X server/display setup to already be available. The
training process runs in `hf`; the reward worker runs in `embench` through
`conda run --no-capture-output -n embench python -m rl.alfred_reward_worker`.
The trainer disables Qwen thinking mode by default to match the vLLM eval
server and keep completions short enough for JSON rewards.

## 4. Merge for vLLM

```bash
python sft/export_model.py \
  --lora rl_output/alf_grpo/lora \
  --output rl_output/alf_grpo/merged
```
