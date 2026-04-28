"""Lightweight GRPO training for ALFRED planner actions."""

from __future__ import annotations

import argparse
import atexit
import importlib
import json
import shlex
import sys
import subprocess
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, Image as DatasetImage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.alfred_replay import score_completion


DEFAULT_MODEL = "sft_output/alf_0426_1540/lora"
DEFAULT_DATASET = "rl_data/alfred_grpo_states.jsonl"


def patch_trl_optional_imports() -> None:
    """Avoid optional TRL imports that are not used by this local GRPO path."""
    import_utils = importlib.import_module("trl.import_utils")
    for name in (
        "_llm_blender_available",
        "_mergekit_available",
        "_vllm_available",
        "_weave_available",
    ):
        if hasattr(import_utils, name):
            setattr(import_utils, name, False)


def disable_thinking_by_default(processor) -> None:
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if apply_chat_template is None:
        return

    def wrapped_apply_chat_template(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        try:
            return apply_chat_template(*args, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return apply_chat_template(*args, **kwargs)

    processor.apply_chat_template = wrapped_apply_chat_template


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


class AlfredRewardClient:
    def __init__(self, command: str, log_path: Path):
        self.command = command
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_path.open("a")
        self.proc = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_file,
            text=True,
            cwd=str(ROOT),
            bufsize=1,
        )
        atexit.register(self.close)
        self.request_id = 0

    def score(
        self,
        completion: Any,
        eval_set: str,
        episode_idx: int,
        history_actions: Any,
        *,
        exp_name: str,
        max_plan_actions: int,
    ) -> float:
        if self.proc.poll() is not None:
            raise RuntimeError(f"ALFRED reward worker exited with code {self.proc.returncode}.")
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        self.request_id += 1
        request = {
            "id": self.request_id,
            "completion": completion,
            "eval_set": eval_set,
            "episode_idx": int(episode_idx),
            "history_actions": history_actions,
            "exp_name": exp_name,
            "max_plan_actions": max_plan_actions,
        }
        self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

        response = None
        while response is None:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("ALFRED reward worker closed stdout.")
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                self.log_file.write(line)
                self.log_file.flush()
                continue
            if candidate.get("id") == self.request_id:
                response = candidate
            else:
                self.log_file.write(line)
                self.log_file.flush()

        if response.get("error"):
            raise RuntimeError(response["error"])
        return float(response["score"])

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.log_file.close()


def make_reward_func(exp_name: str, max_plan_actions: int, worker: AlfredRewardClient | None):
    def reward_func(
        completions,
        eval_set,
        episode_idx,
        history_actions,
        **_: Any,
    ) -> list[float]:
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


def ensure_lora_trainable(model) -> tuple[int, int]:
    for name, param in model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            param.requires_grad_(True)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    if trainable == 0:
        raise RuntimeError("No trainable parameters found. Check the model path or pass --add-lora.")
    return trainable, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("rl_output/alf_grpo"))
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=7168)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-plan-actions", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--reward-exp-name", default="alf_grpo_reward")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Keep Qwen thinking mode in chat templates. Default disables it like the vLLM eval server.",
    )
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument(
        "--reward-worker-cmd",
        default="conda run --no-capture-output -n embench python -m rl.alfred_reward_worker",
        help="Command used to run ALFRED reward replay. Empty string runs reward in-process.",
    )
    parser.add_argument(
        "--add-lora",
        action="store_true",
        help="Add a fresh LoRA adapter. Leave off when continuing from an SFT LoRA path.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny config to validate imports, generation, and reward replay.",
    )
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
        worker = AlfredRewardClient(
            args.reward_worker_cmd,
            args.output_dir / "reward_worker.log",
        )

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

    is_adapter = (Path(args.model) / "adapter_config.json").exists()
    if args.add_lora or not is_adapter:
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
