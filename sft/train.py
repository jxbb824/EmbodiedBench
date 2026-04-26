"""
EB-ALFRED Vision SFT with Unsloth LoRA for Qwen3.5-9B.

Usage:
    CUDA_VISIBLE_DEVICES=0 python sft/train.py

Prerequisites:
    1. bash sft/download_data.sh
    2. pip install unsloth trl pillow
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_NAME   = "Qwen/Qwen3.5-9B"
DATA_DIR     = Path("sft_data/EB-Alfred_trajectory_dataset")
OUTPUT_DIR   = Path("sft_output")
MAX_SEQ_LEN  = 8192

LORA_R       = 16
LORA_ALPHA   = 16
BATCH_SIZE   = 1
GRAD_ACCUM   = 4
LR           = 5e-5
NUM_EPOCHS   = 1
WARMUP_STEPS = 50
SAVE_STEPS   = 2000
LOGGING_STEPS = 1

MAX_PLAN_ACTIONS = 5
ONLY_SUCCESSFUL  = True  # True = only use successful episodes
MAX_EPISODES     = 700    # 0 = use all; ~700 episodes ≈ 10000 examples
IMAGE_RESIZE     = 0      # resize images to save RAM (0 = no resize)
MIN_RESPONSE_TOKENS = 16
IMAGE_TOKEN_MARGIN  = 512

# ═══════════════════════════════════════════════════════════════════════════════
# JSON template appended for Qwen3.5 at inference (vlm_planner.py L245-246)
# ═══════════════════════════════════════════════════════════════════════════════
VLM_JSON_TEMPLATE = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, 
4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''

# ═══════════════════════════════════════════════════════════════════════════════
# Data conversion — mirrors vlm_planner.act() exactly
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_base_prompt(ep_input: str) -> tuple[str, str]:
    """
    ep["input"] = full first-step prompt (system prompt + actions + examples + instruction).
    Split into (base_prompt, instruction) at the "## Now the human instruction is:" marker.
    """
    marker = "## Now the human instruction is:"
    idx = ep_input.find(marker)
    if idx == -1:
        return ep_input, ""
    base = ep_input[:idx]
    rest = ep_input[idx + len(marker):]
    instruction = rest.split(".")[0].strip()
    return base, instruction


def _format_feedback(step_idx: int, plan_entry: dict) -> str:
    """Same format as VLMPlanner._format_action_feedback() — dict branch."""
    aid, aname = plan_entry["action"]
    parts = [f"Step {step_idx}, action id {aid}, {aname}"]
    fb = plan_entry.get("env_feedback")
    if fb:
        parts.append(f"env feedback: {fb}")
    return ", ".join(parts)


def _build_user_text_step0(ep_input: str) -> str:
    """First step: use the stored prompt verbatim + append JSON template."""
    return ep_input + VLM_JSON_TEMPLATE


def _build_user_text_replan(
    base_prompt: str,
    instruction: str,
    prev_steps: list[dict],
    max_action_id: int,
) -> str:
    """Subsequent steps: system prompt + instruction + action history + replan."""
    prompt = base_prompt
    prompt += f"## Now the human instruction is: {instruction}."
    prompt += "\n\n The action history:"

    for i, step in enumerate(prev_steps):
        plan_entry = step["executable_plan"][0]
        prompt += "\n" + _format_feedback(i, plan_entry)

    prompt += (
        f"\n\n Considering the above interaction history and the current image state,"
        f" to achieve the human instruction: '{instruction}',"
        f" you are supposed to output in json."
        f" You need to describe current visual state from the image,"
        f" summarize interaction history and environment feedback"
        f" and reason why the last action or plan failed and did not finish the task,"
        f" output your new plan to achieve the goal from current state."
        f" At the end, output only the next 1-{MAX_PLAN_ACTIONS}"
        f" excutable action id(s)(0 ~ {max_action_id}) from the available actions."
        f" The task is NOT finished yet — you MUST output at least one action."
    )
    prompt += VLM_JSON_TEMPLATE
    return prompt


def _build_assistant_text(step: dict) -> str:
    """Build target JSON from the dataset's pre-existing model output."""
    plan_entry = step["executable_plan"][0]
    aid, aname = plan_entry["action"]

    return json.dumps(
        {
            "visual_state_description": step["visual_description"],
            "reasoning_and_reflection": step["reasoning_and_reflection"],
            "language_plan": step["language_plan"],
            "executable_plan": [{"action_id": aid, "action_name": aname}],
        },
        ensure_ascii=False,
    )


def _resolve_image_path(data_dir: Path, img_path: str) -> Path | None:
    """Find the actual file path on disk."""
    for candidate in [
        data_dir / "images" / img_path,   # images/images/model/...
        data_dir / img_path,               # images/model/...
    ]:
        if candidate.exists():
            return candidate
    return None




def convert_episode(episode: dict, data_dir: Path) -> list[dict]:
    trajectory = episode.get("trajectory", [])
    if not trajectory:
        return []

    base_prompt, instruction = _extract_base_prompt(episode["input"])

    # extract max action id from the first-step prompt (e.g. "0 ~ 207")
    import re
    m = re.search(r"\(0 ~ (\d+)\)", episode["input"])
    max_action_id = int(m.group(1)) if m else 200

    examples: list[dict] = []
    for idx, step in enumerate(trajectory):
        if not step.get("executable_plan"):
            continue

        img_file = _resolve_image_path(data_dir, step["input_image_path"])
        if img_file is None:
            continue
        image = str(img_file)

        if idx == 0:
            user_text = _build_user_text_step0(episode["input"])
        else:
            user_text = _build_user_text_replan(
                base_prompt, instruction, trajectory[:idx], max_action_id
            )

        asst_text = _build_assistant_text(step)

        examples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
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
    return examples


def load_dataset(data_dir: Path) -> list[dict]:
    json_path = data_dir / "eb-alfred_dataset_single_step.json"
    if not json_path.exists():
        sys.exit(f"Dataset not found: {json_path}. Run download_data.sh first.")

    with open(json_path) as f:
        episodes = json.load(f)
    print(f"Loaded {len(episodes)} episodes")

    if ONLY_SUCCESSFUL:
        episodes = [ep for ep in episodes if ep.get("success", 0) == 1.0]
        print(f"  Filtered to {len(episodes)} successful episodes")

    if MAX_EPISODES:
        episodes = episodes[:MAX_EPISODES]
        print(f"  Capped to {len(episodes)} episodes")

    dataset: list[dict] = []
    skipped = 0
    for i, ep in enumerate(episodes):
        if (i + 1) % 100 == 0 or i + 1 == len(episodes):
            print(f"  Converting: {i+1}/{len(episodes)} episodes, {len(dataset)} examples ...", flush=True)
        converted = convert_episode(ep, data_dir)
        if converted:
            dataset.extend(converted)
        else:
            skipped += 1

    print(f"Converted to {len(dataset)} training examples ({skipped} episodes skipped)")
    return dataset


def _find_subsequence(sequence: list[int], pattern: list[int]) -> int:
    for idx in range(len(sequence) - len(pattern) + 1):
        if sequence[idx:idx + len(pattern)] == pattern:
            return idx
    return -1


def filter_examples_with_visible_response(dataset: list[dict], processor, max_seq_len: int) -> list[dict]:
    """
    Response-only training gives NaN loss when truncation removes all assistant
    tokens and the collator masks the whole sample to -100 labels.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    response_ids = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids

    kept = []
    skipped_missing_response = 0
    skipped_truncated_response = 0
    for example in dataset:
        text = processor.apply_chat_template(example["messages"], tokenize=False)
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        response_start = _find_subsequence(token_ids, response_ids)
        if response_start < 0:
            skipped_missing_response += 1
            continue

        min_required_len = response_start + len(response_ids) + MIN_RESPONSE_TOKENS + IMAGE_TOKEN_MARGIN
        if min_required_len > max_seq_len:
            skipped_truncated_response += 1
            continue

        kept.append(example)

    skipped = skipped_missing_response + skipped_truncated_response
    print(
        f"Response-window filter kept {len(kept)}/{len(dataset)} examples "
        f"(skipped {skipped_truncated_response} truncated, "
        f"{skipped_missing_response} missing response marker)."
    )
    if not kept:
        sys.exit("No examples keep assistant labels inside max sequence length.")
    if skipped:
        print("Skipped examples would likely create all -100 labels and NaN response-only loss.")
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    dataset = load_dataset(DATA_DIR)
    if not dataset:
        sys.exit("No training examples. Check data paths / images.")

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
    )
    dataset = filter_examples_with_visible_response(dataset, tokenizer, MAX_SEQ_LEN)

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
            # max_grad_norm=1.0,
            seed=3407,
            output_dir=str(OUTPUT_DIR),
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

    lora_dir = OUTPUT_DIR / "lora"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    print(f"LoRA adapter saved to {lora_dir}")


if __name__ == "__main__":
    main()
