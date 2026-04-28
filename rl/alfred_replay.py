"""Replay-based ALFRED rewards for lightweight GRPO training."""

from __future__ import annotations

import json
import re
from typing import Any


MAX_REWARD_PLAN_ACTIONS = 3


def completion_to_text(completion: Any) -> str:
    """Normalize TRL/transformers completion objects to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if completion and isinstance(completion[-1], dict):
            return completion_to_text(completion[-1].get("content", ""))
        return "".join(completion_to_text(item) for item in completion)
    if isinstance(completion, dict):
        return completion_to_text(completion.get("content", ""))
    return str(completion)


def normalize_history_actions(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return [int(action_id) for action_id in value]


def parse_action_ids(text: str, max_actions: int = MAX_REWARD_PLAN_ACTIONS) -> tuple[list[int], bool]:
    """Parse action ids from the planner JSON response."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return [], False
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], False

    plan = data.get("executable_plan")
    if not isinstance(plan, list):
        return [], False

    action_ids: list[int] = []
    for item in plan[:max_actions]:
        if isinstance(item, dict):
            raw_action = item.get("action_id")
        elif isinstance(item, (list, tuple)) and item:
            raw_action = item[0]
        else:
            continue
        try:
            action_ids.append(int(raw_action))
        except (TypeError, ValueError):
            return [], False

    return action_ids, bool(action_ids)


def score_completion(
    completion: Any,
    eval_set: str,
    episode_idx: int,
    history_actions: Any,
    *,
    exp_name: str = "alf_grpo_reward",
    max_plan_actions: int = MAX_REWARD_PLAN_ACTIONS,
) -> float:
    """Replay a stored state, execute a candidate plan, and return a scalar reward."""
    action_ids, format_ok = parse_action_ids(completion_to_text(completion), max_plan_actions)
    if not format_ok:
        return -1.0

    from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv

    env = EBAlfEnv(eval_set=eval_set, exp_name=exp_name, selected_indexes=[int(episode_idx)])
    score = 0.2
    best_progress = 0.0

    try:
        env.reset()
        done = False

        for action_id in normalize_history_actions(history_actions):
            if action_id < 0 or action_id >= len(env.language_skill_set):
                return -1.0
            _, _, done, info = env.step(action_id, {})
            best_progress = max(best_progress, float(info.get("task_progress", 0.0)))
            if done:
                break

        if done:
            return 3.0

        for action_id in action_ids:
            if action_id < 0 or action_id >= len(env.language_skill_set):
                score -= 0.5
                break

            _, _, done, info = env.step(action_id, {})
            score -= 0.02

            if float(info.get("last_action_success", 0.0)) < 1.0:
                score -= 0.5

            progress = float(info.get("task_progress", 0.0))
            if progress > best_progress + 1e-6:
                score += 1.0
                best_progress = progress

            if float(info.get("task_success", 0.0)) >= 1.0:
                score += 3.0
                break

            if done:
                break

        return float(score)
    finally:
        env.close()
