"""Build replayable ALFRED state samples for GRPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sft.train import (
    TASKS,
    _build_user_text_replan,
    _build_user_text_step0,
    _enrich_distance_deltas,
    _extract_base_prompt,
    _extract_max_action_id,
    _resolve_image_path,
)


DEFAULT_OUTPUT = Path("rl_data/alfred_grpo_states.jsonl")


def _episode_index(episode: dict[str, Any]) -> int:
    """SFT data stores 1-based episode ids; EBAlfEnv selected_indexes are 0-based."""
    return max(0, int(episode["episode_id"]) - 1)


def _normalize_image_roots(raw_roots: str, data_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for item in raw_roots.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        candidates = [
            path,
            data_dir / path,
            data_dir / "images" / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                roots.append(candidate.resolve())
                break
        else:
            roots.append(path.resolve())
    return roots


def _episode_matches_image_roots(
    episode: dict[str, Any],
    image_roots: list[Path],
    cfg,
) -> bool:
    if not image_roots:
        return True

    for step in episode.get("trajectory", []):
        image = _resolve_image_path(cfg.data_dir, step.get("input_image_path", ""))
        if image is None:
            continue
        resolved = image.resolve()
        return any(resolved.is_relative_to(root) for root in image_roots)
    return False


def iter_state_rows(episode: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = TASKS["alf"]
    trajectory = episode.get("trajectory", [])
    if not trajectory:
        return []

    base_prompt, instruction = _extract_base_prompt(episode["input"])
    max_action_id = _extract_max_action_id(episode["input"])
    _enrich_distance_deltas(trajectory)

    rows: list[dict[str, Any]] = []
    history: list[tuple[int, dict[str, Any]]] = []
    history_actions: list[int] = []

    for step in trajectory:
        executable_plan = step.get("executable_plan") or []
        if not executable_plan:
            continue

        image = _resolve_image_path(cfg.data_dir, step.get("input_image_path", ""))
        if image is None:
            continue

        if history:
            prompt_text = _build_user_text_replan(
                base_prompt, instruction, history, max_action_id, cfg
            )
        else:
            prompt_text = _build_user_text_step0(episode["input"])

        oracle_actions = [int(entry["action"][0]) for entry in executable_plan]
        rows.append(
            {
                "prompt_text": prompt_text,
                "image_path": str(image),
                "eval_set": episode["eval_set"],
                "episode_idx": _episode_index(episode),
                "history_actions": list(history_actions),
                "instruction": episode.get("instruction", instruction),
                "oracle_action_ids": oracle_actions,
            }
        )

        for entry in executable_plan:
            history.append((len(history), entry))
            history_actions.append(int(entry["action"][0]))

    return rows


def build_rows(
    source: Path,
    *,
    eval_sets: set[str] | None,
    only_successful: bool,
    max_examples: int,
    image_roots: list[Path] | None = None,
) -> list[dict[str, Any]]:
    with source.open() as f:
        episodes = json.load(f)

    cfg = TASKS["alf"]
    image_roots = image_roots or []
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        if eval_sets and episode.get("eval_set") not in eval_sets:
            continue
        if only_successful and float(episode.get("success", 0.0)) < 1.0:
            continue
        if not _episode_matches_image_roots(episode, image_roots, cfg):
            continue

        rows.extend(iter_state_rows(episode))
        if max_examples and len(rows) >= max_examples:
            return rows[:max_examples]

    return rows


def parse_args() -> argparse.Namespace:
    cfg = TASKS["alf"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-mode",
        choices=("single", "multi"),
        default="multi",
        help="Trajectory file to use when --source is omitted.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="ALFRED SFT trajectory JSON.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--eval-sets",
        default="",
        help="Comma-separated subset, empty means all sets.",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Use failed episodes too. Default uses successful trajectories only.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=2000,
        help="0 means no cap. Keep this small for the first RL run.",
    )
    parser.add_argument(
        "--image-roots",
        default="",
        help="Comma-separated image root paths. Empty means all trajectory sources.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source or TASKS["alf"].data_dir / TASKS["alf"].json_files[args.step_mode]
    eval_sets = {item.strip() for item in args.eval_sets.split(",") if item.strip()}
    image_roots = _normalize_image_roots(args.image_roots, TASKS["alf"].data_dir)
    rows = build_rows(
        source,
        eval_sets=eval_sets or None,
        only_successful=not args.all_episodes,
        max_examples=args.max_examples,
        image_roots=image_roots,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} ALFRED GRPO state samples to {args.output}")


if __name__ == "__main__":
    main()
