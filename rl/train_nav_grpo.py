"""Lightweight GRPO training for EB-Navigation planner actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, Image as DatasetImage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.nav_replay import score_completion
from rl.train_alf_grpo import (
    AlfredRewardClient,
    disable_thinking_by_default,
    ensure_lora_trainable,
    patch_trl_optional_imports,
)


DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_DATASET = "rl_data/nav_grpo_states.jsonl"


def load_state_dataset(path: Path, max_examples: int = 0) -> Dataset:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "prompt": [{"role": "user", "content": item["prompt_text"]}],
                    "image": item["image_path"],
                    "eval_set": item["eval_set"],
                    "episode_idx": int(item["episode_idx"]),
                    "history_actions": item["history_actions"],
                }
            )
            if max_examples and len(rows) >= max_examples:
                break

    if not rows:
        raise ValueError(f"No RL samples found in {path}")
    return Dataset.from_list(rows).cast_column("image", DatasetImage())


def make_reward_func(exp_name: str, max_plan_actions: int, worker: AlfredRewardClient | None):
    def reward_func(completions, eval_set, episode_idx, history_actions, **_: Any) -> list[float]:
        rewards: list[float] = []
        for completion, cur_set, cur_episode, cur_history in zip(
            completions, eval_set, episode_idx, history_actions
        ):
            if worker is None:
                rewards.append(
                    score_completion(
                        completion,
                        cur_set,
                        int(cur_episode),
                        cur_history,
                        exp_name=exp_name,
                        max_plan_actions=max_plan_actions,
                    )
                )
            else:
                rewards.append(
                    worker.score(
                        completion,
                        cur_set,
                        int(cur_episode),
                        cur_history,
                        exp_name=exp_name,
                        max_plan_actions=max_plan_actions,
                    )
                )
        return rewards

    return reward_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("rl_output/nav_grpo"))
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=7168)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-plan-actions", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=850)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reward-exp-name", default="nav_grpo_reward")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument(
        "--reward-worker-cmd",
        default="conda run --no-capture-output -n embench_nav python -m rl.nav_reward_worker",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_examples = args.max_examples or 4
        args.max_steps = min(args.max_steps, 2)
        args.save_steps = min(args.save_steps, args.max_steps)
        args.num_generations = 2
        args.grad_accum = 2

    if args.grad_accum % args.num_generations != 0:
        raise ValueError("--grad-accum must be divisible by --num-generations for per-device batch size 1.")

    dataset = load_state_dataset(args.dataset, args.max_examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = None
    if args.reward_worker_cmd:
        worker = AlfredRewardClient(args.reward_worker_cmd, args.output_dir / "reward_worker.log")

    from unsloth import FastVisionModel

    patch_trl_optional_imports()
    from trl import GRPOConfig, GRPOTrainer

    print(f"Loading model from {args.model}")
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    if not args.enable_thinking:
        disable_thinking_by_default(processor)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        target_modules="all-linear",
    )

    FastVisionModel.for_training(model)
    trainable, total = ensure_lora_trainable(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total:.2%})")

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=make_reward_func(args.reward_exp_name, args.max_plan_actions, worker),
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            warmup_steps=0,
            optim="adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_steps=args.save_steps,
            save_total_limit=3,
            report_to="none",
            seed=3407,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            remove_unused_columns=False,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
            use_vllm=False,
            beta=0.0,
            log_completions=args.log_completions,
            num_completions_to_print=2,
        ),
    )

    trainer.train()

    lora_dir = args.output_dir / "lora"
    model.save_pretrained(str(lora_dir))
    processor.save_pretrained(str(lora_dir))
    print(f"LoRA adapter saved to {lora_dir}")


if __name__ == "__main__":
    main()
