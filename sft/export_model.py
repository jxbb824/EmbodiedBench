"""
Merge LoRA adapter into base model and save for vLLM deployment.

Usage:
    python sft/export_model.py                        # use defaults
    python sft/export_model.py --lora sft_output/lora --output sft_output/merged

Then serve with:
    vllm serve sft_output/merged --max-model-len 8192
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA → vLLM-ready model")
    parser.add_argument("--lora", type=str, default="sft_output/lora",
                        help="Path to saved LoRA adapter directory")
    parser.add_argument("--output", type=str, default="sft_output/merged",
                        help="Output directory for merged 16-bit model")
    args = parser.parse_args()

    lora_path = Path(args.lora)
    out_path = Path(args.output)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")

    from unsloth import FastVisionModel

    print(f"Loading LoRA adapter from {lora_path} ...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=str(lora_path),
        load_in_4bit=False,
    )

    print(f"Merging and saving 16-bit model to {out_path} ...")
    model.save_pretrained_merged(
        str(out_path),
        tokenizer,
        save_method="merged_16bit",
    )

    print("Done.")
    print(f"  Serve with:  vllm serve {out_path} --max-model-len 8192")


if __name__ == "__main__":
    main()
