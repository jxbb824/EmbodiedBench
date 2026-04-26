"""
EB Vision SFT with Unsloth LoRA for Qwen3.5-9B.

Supports ALFRED, Navigation, or both (mixed).

Usage:
    python sft/train.py --task alf
    python sft/train.py --task nav
    python sft/train.py --task both
    python sft/train.py --task nav --step-mode multi

Prerequisites:
    1. bash sft/download_data.sh
    2. pip install unsloth trl pillow
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

# ═══════════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_NAME   = "Qwen/Qwen3.5-9B"
OUTPUT_DIR   = Path("sft_output")
MAX_SEQ_LEN  = 8192

LORA_R       = 16
LORA_ALPHA   = 16
BATCH_SIZE   = 1
GRAD_ACCUM   = 8
LR           = 5e-5
NUM_EPOCHS   = 1
WARMUP_STEPS = 50
SAVE_STEPS   = 2000
LOGGING_STEPS = 1

MAX_PLAN_ACTIONS    = 5
ONLY_SUCCESSFUL     = True
MAX_EPISODES        = 0        # 0 = use all episodes
MIN_RESPONSE_TOKENS = 16
IMAGE_TOKEN_MARGIN  = 512

# ═══════════════════════════════════════════════════════════════════════════════
# JSON format template — appended for Qwen3.5 at inference
# (planner_utils.template, shared by ALFRED & NAV)
# ═══════════════════════════════════════════════════════════════════════════════
JSON_TEMPLATE = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, 
4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

TEMPLATE_MARKER = "The output json format should be"

# ═══════════════════════════════════════════════════════════════════════════════
# Task-specific: feedback formatting
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_fb_alf(step_idx: int, entry: dict) -> str:
    """ALFRED: Step N, action id X, name, env feedback: ..."""
    aid, aname = entry["action"]
    parts = [f"Step {step_idx}, action id {aid}, {aname}"]
    fb = entry.get("env_feedback")
    if fb:
        parts.append(f"env feedback: {fb}")
    return ", ".join(parts)


def _fmt_fb_nav(step_idx: int, entry: dict) -> str:
    """Nav: Step N, action id X, name, env feedback: ..., distance: ..., delta: ..."""
    aid, aname = entry["action"]
    parts = [f"Step {step_idx}, action id {aid}, {aname}"]
    fb = entry.get("env_feedback")
    if fb:
        parts.append(f"env feedback: {fb}")
    if "distance" in entry:
        parts.append(f"distance to target: {entry['distance']:.2f}")
    if "distance_delta" in entry:
        parts.append(f"distance change after action: {entry['distance_delta']:+.2f}")
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Task-specific: replan suffix (text between action history and JSON template)
# ═══════════════════════════════════════════════════════════════════════════════

def _replan_alf(instruction: str, max_aid: int) -> str:
    """Mirrors vlm_planner.process_prompt() non-first, non-chat_history branch."""
    return (
        f"\n\n Considering the above interaction history and the current image state,"
        f" to achieve the human instruction: '{instruction}',"
        f" you are supposed to output in json."
        f" You need to describe current visual state from the image,"
        f" summarize interaction history and environment feedback"
        f" and reason why the last action or plan failed and did not finish the task,"
        f" output your new plan to achieve the goal from current state."
        f" At the end, output only the next 1-{MAX_PLAN_ACTIONS}"
        f" excutable action id(s)(0 ~ {max_aid}) from the available actions."
        f" The task is NOT finished yet — you MUST output at least one action."
    )


def _replan_nav(instruction: str, max_aid: int) -> str:
    """Mirrors nav_planner.following_prompt (without the trailing JSON template)."""
    return (
        f"\n\nTo achieve the task,"
        f" 1. Reason about the current visual state and your final goal,"
        f" and 2. Reflect on the effect of previous actions."
        f" 3. Summarize how you learn from the Strategy and Examples provided "
        f"\nAim for about 1-{MAX_PLAN_ACTIONS} actions in this step"
        f" to be closer to the target object."
        f" !!!Notice: you cannot assess the situation until the whole plan"
        f" in this planning step is finished executed, so plan accordingly."
        f"\nAt last, output the action id(s) (0 ~ {max_aid})"
        f" from the available actions to execute. "
        f"\n\nThe input given to you is an first person view observation"
        f" . Plan accordingly based on the visual observation."
        f"\nWhen distance change is available in the history,"
        f" negative means the last move got closer to the target"
        f" and positive means it got farther away."
        f"\n\nYou are supposed to output in JSON."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Task configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskConfig:
    name: str
    data_dir: Path
    json_files: dict[str, str]                     # {"single": ..., "multi": ...}
    fmt_feedback: Callable[[int, dict], str]
    replan_suffix: Callable[[str, int], str]
    history_line_prefix: str                        # "\n" for alf, "\n " for nav


TASKS: dict[str, TaskConfig] = {
    "alf": TaskConfig(
        name="alf",
        data_dir=Path("sft_data/EB-Alfred_trajectory_dataset"),
        json_files={
            "single": "eb-alfred_dataset_single_step.json",
            "multi":  "eb-alfred_dataset_multi_step.json",
        },
        fmt_feedback=_fmt_fb_alf,
        replan_suffix=_replan_alf,
        history_line_prefix="\n",
    ),
    "nav": TaskConfig(
        name="nav",
        data_dir=Path("sft_data/EB-Nav_trajectory_dataset"),
        json_files={
            "single": "eb-nav_dataset_single_step.json",
            "multi":  "eb-nav_dataset_multi_step.json",
        },
        fmt_feedback=_fmt_fb_nav,
        replan_suffix=_replan_nav,
        history_line_prefix="\n ",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Shared data conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_base_prompt(ep_input: str) -> tuple[str, str]:
    """Split ep["input"] into (base_prompt, instruction) at the marker."""
    marker = "## Now the human instruction is:"
    idx = ep_input.find(marker)
    if idx == -1:
        return ep_input, ""
    base = ep_input[:idx]
    rest = ep_input[idx + len(marker):]
    instruction = rest.split(".")[0].strip()
    return base, instruction


def _extract_max_action_id(ep_input: str) -> int:
    m = re.search(r"\(0 ~ (\d+)\)", ep_input)
    return int(m.group(1)) if m else 7


def _enrich_distance_deltas(trajectory: list[dict]) -> None:
    """If steps contain numeric 'distance', compute 'distance_delta' in-place."""
    prev_dist = None
    for step in trajectory:
        for entry in step.get("executable_plan", []):
            dist = entry.get("distance")
            if dist is not None:
                dist = float(dist)
                entry["distance"] = dist
                if prev_dist is not None:
                    entry["distance_delta"] = dist - prev_dist
                prev_dist = dist


def _resolve_image_path(data_dir: Path, img_path: str) -> Path | None:
    for candidate in [
        data_dir / "images" / img_path,
        data_dir / img_path,
    ]:
        if candidate.exists():
            return candidate
    return None


def _build_user_text_step0(ep_input: str) -> str:
    """First step: stored prompt + JSON template (avoid duplication)."""
    if TEMPLATE_MARKER in ep_input:
        return ep_input
    return ep_input + JSON_TEMPLATE


def _build_user_text_replan(
    base_prompt: str,
    instruction: str,
    history: list[tuple[int, dict]],
    max_action_id: int,
    cfg: TaskConfig,
) -> str:
    """Subsequent steps: system prompt + instruction + history + replan suffix."""
    prompt = base_prompt
    prompt += f"## Now the human instruction is: {instruction}."
    prompt += "\n\n The action history:"
    for step_idx, entry in history:
        prompt += cfg.history_line_prefix + cfg.fmt_feedback(step_idx, entry)
    prompt += cfg.replan_suffix(instruction, max_action_id)
    prompt += JSON_TEMPLATE
    return prompt


def _build_assistant_text(step: dict) -> str:
    """Build target JSON from dataset fields (works for single & multi step)."""
    actions = []
    for entry in step["executable_plan"]:
        aid, aname = entry["action"]
        actions.append({"action_id": aid, "action_name": aname})
    return json.dumps(
        {
            "visual_state_description": step.get("visual_description", ""),
            "reasoning_and_reflection": step.get("reasoning_and_reflection", ""),
            "language_plan": step.get("language_plan", ""),
            "executable_plan": actions,
        },
        ensure_ascii=False,
    )


def convert_episode(episode: dict, cfg: TaskConfig) -> list[dict]:
    trajectory = episode.get("trajectory", [])
    if not trajectory:
        return []

    base_prompt, instruction = _extract_base_prompt(episode["input"])
    max_action_id = _extract_max_action_id(episode["input"])
    _enrich_distance_deltas(trajectory)

    examples: list[dict] = []
    history: list[tuple[int, dict]] = []

    for step in trajectory:
        if not step.get("executable_plan"):
            continue

        img = _resolve_image_path(cfg.data_dir, step.get("input_image_path", ""))
        if img is None:
            continue

        if not history:
            user_text = _build_user_text_step0(episode["input"])
        else:
            user_text = _build_user_text_replan(
                base_prompt, instruction, history, max_action_id, cfg,
            )

        asst_text = _build_assistant_text(step)

        examples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img)},
                            {"type": "text", "text": user_text},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": asst_text}],
                    },
                ]
            }
        )

        for entry in step["executable_plan"]:
            history.append((len(history), entry))

    return examples


def load_task_episodes(cfg: TaskConfig, step_mode: str) -> list[dict]:
    json_file = cfg.json_files[step_mode]
    json_path = cfg.data_dir / json_file
    if not json_path.exists():
        sys.exit(f"Dataset not found: {json_path}. Run: bash sft/download_data.sh")

    with open(json_path) as f:
        episodes = json.load(f)
    print(f"[{cfg.name}] Loaded {len(episodes)} episodes from {json_file}")

    if ONLY_SUCCESSFUL:
        episodes = [ep for ep in episodes if ep.get("success", 0) == 1.0]
        print(f"[{cfg.name}]   → {len(episodes)} successful")

    if MAX_EPISODES:
        episodes = episodes[:MAX_EPISODES]
        print(f"[{cfg.name}]   → capped to {MAX_EPISODES}")

    return episodes


def build_dataset(task_names: list[str], step_mode: str) -> list[dict]:
    dataset: list[dict] = []
    for name in task_names:
        cfg = TASKS[name]
        episodes = load_task_episodes(cfg, step_mode)
        skipped = 0
        for i, ep in enumerate(episodes):
            if (i + 1) % 100 == 0 or i + 1 == len(episodes):
                print(
                    f"[{cfg.name}] Converting: {i+1}/{len(episodes)}, "
                    f"{len(dataset)} examples ...",
                    flush=True,
                )
            converted = convert_episode(ep, cfg)
            if converted:
                dataset.extend(converted)
            else:
                skipped += 1
        print(f"[{cfg.name}] {len(dataset)} total examples ({skipped} skipped)")
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Response-window filter (prevents NaN from all-masked samples)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_subsequence(seq: list[int], pattern: list[int]) -> int:
    for i in range(len(seq) - len(pattern) + 1):
        if seq[i : i + len(pattern)] == pattern:
            return i
    return -1


def filter_visible_response(dataset: list[dict], processor, max_seq_len: int) -> list[dict]:
    tokenizer = getattr(processor, "tokenizer", processor)
    response_ids = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids

    kept, skip_trunc, skip_miss = [], 0, 0
    for ex in dataset:
        text = processor.apply_chat_template(ex["messages"], tokenize=False)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        pos = _find_subsequence(ids, response_ids)
        if pos < 0:
            skip_miss += 1
            continue
        if pos + len(response_ids) + MIN_RESPONSE_TOKENS + IMAGE_TOKEN_MARGIN > max_seq_len:
            skip_trunc += 1
            continue
        kept.append(ex)

    print(
        f"Response filter: kept {len(kept)}/{len(dataset)} "
        f"(truncated {skip_trunc}, missing marker {skip_miss})"
    )
    if not kept:
        sys.exit("No examples survived the response-window filter.")
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EB Vision SFT")
    p.add_argument("--task", choices=["alf", "nav", "both"], default="alf",
                    help="Which task(s) to train on")
    p.add_argument("--step-mode", choices=["single", "multi"], default="single",
                    help="single-step or multi-step trajectory data")
    p.add_argument("--max-episodes", type=int, default=None,
                    help="Override MAX_EPISODES (per task)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    global MAX_EPISODES
    if args.max_episodes is not None:
        MAX_EPISODES = args.max_episodes

    task_names = ["alf", "nav"] if args.task == "both" else [args.task]

    from datetime import datetime
    run_tag = f"{args.task}_{datetime.now().strftime('%m%d_%H%M')}"
    run_dir = OUTPUT_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    dataset = build_dataset(task_names, args.step_mode)
    if not dataset:
        sys.exit("No training examples. Check data paths / images.")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )

    dataset = filter_visible_response(dataset, tokenizer, MAX_SEQ_LEN)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(
            model,
            tokenizer,
            train_on_responses_only=True,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        ),
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LR,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=str(run_dir),
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_SEQ_LEN,
        ),
    )

    print("Starting training ...")
    trainer.train()

    lora_dir = run_dir / "lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"LoRA adapter saved to {lora_dir}")


if __name__ == "__main__":
    main()
