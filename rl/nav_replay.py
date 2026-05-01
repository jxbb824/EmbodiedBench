"""Replay-based EB-Navigation rewards for lightweight GRPO training."""

from __future__ import annotations

from typing import Any

from rl.alfred_replay import completion_to_text, parse_action_ids


MAX_REWARD_PLAN_ACTIONS = 3


def _history_actions(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        import json

        value = json.loads(value)
    return [int(action_id) for action_id in value]


def _distance_reward(prev_dist: float, new_dist: float) -> float:
    delta = prev_dist - new_dist
    if delta >= 0:
        return min(delta, 0.5)
    return max(delta * 0.5, -0.3)


def score_completion(
    completion: Any,
    eval_set: str,
    episode_idx: int,
    history_actions: Any,
    *,
    exp_name: str = "nav_grpo_reward",
    max_plan_actions: int = MAX_REWARD_PLAN_ACTIONS,
) -> float:
    action_ids, format_ok = parse_action_ids(completion_to_text(completion), max_plan_actions)
    if not format_ok:
        return -1.0

    from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv

    env = EBNavigationEnv(eval_set=eval_set, exp_name=exp_name, selected_indexes=[int(episode_idx)])
    score = 0.1

    try:
        env.reset()
        done = False

        for action_id in _history_actions(history_actions):
            if action_id < 0 or action_id >= len(env.language_skill_set):
                return -1.0
            _, _, done, _ = env.step(action_id, {}, 0)
            if done:
                break

        _, start_dist = env.measure_success()
        prev_dist = float(start_dist)
        best_dist = prev_dist

        if done:
            return 3.0

        for action_id in action_ids:
            if action_id < 0 or action_id >= len(env.language_skill_set):
                score -= 0.5
                break

            _, _, done, info = env.step(action_id, {}, 0)
            new_dist = float(info.get("distance", prev_dist))
            score -= 0.01

            if not info.get("last_action_success", False):
                score -= 0.2

            score += _distance_reward(prev_dist, new_dist)
            best_dist = min(best_dist, new_dist)
            prev_dist = new_dist

            if float(info.get("task_success", 0.0)) >= 1.0:
                score += 3.0
                break
            if done:
                break

        score += max(0.0, float(start_dist) - best_dist)
        return float(score)
    finally:
        env.close()
