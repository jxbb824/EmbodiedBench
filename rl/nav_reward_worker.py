"""JSONL reward worker that runs EB-Navigation env code in embench_nav."""

from __future__ import annotations

import json
import sys
import traceback


PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

from rl.nav_replay import score_completion  # noqa: E402


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue

        request = json.loads(line)
        try:
            score = score_completion(
                request["completion"],
                request["eval_set"],
                int(request["episode_idx"]),
                request.get("history_actions", []),
                exp_name=request.get("exp_name", "nav_grpo_reward"),
                max_plan_actions=int(request.get("max_plan_actions", 3)),
            )
            response = {"id": request.get("id"), "score": score}
        except Exception:
            response = {"id": request.get("id"), "error": traceback.format_exc()}

        PROTOCOL_OUT.write(json.dumps(response, ensure_ascii=False) + "\n")
        PROTOCOL_OUT.flush()


if __name__ == "__main__":
    main()
