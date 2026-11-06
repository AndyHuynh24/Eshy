"""
Record a trained policy's actions in sim as a replayable sequence.

By default recording stops when the episode ends (sit, turn, ...). Use
--seconds for tasks with no natural end, such as walk.

    python record_sequence.py sit
    python record_sequence.py walk --seconds 10

Output is a JSON list of [FL, FR, BL, BR] integer robot-frame degrees at
25 Hz — the scale of deploy.py's A: command:

    python deploy.py --sequence sequences/walk_10.0s.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO

from eshy_env       import CONTROL_HZ
from eshy_walk_env  import EshyWalkEnv
from eshy_sit_env   import EshySitEnv
from eshy_lay_env   import EshyLayEnv
from eshy_turn_env  import EshyTurnEnv
from eshy_arc_env   import EshyArcEnv


TASKS = {
    "walk": EshyWalkEnv,
    "sit":  EshySitEnv,
    "lay":  EshyLayEnv,
    "turn": EshyTurnEnv,
    "arc":  EshyArcEnv,
}


def default_model_path(task: str) -> str:
    return os.path.join("models", task, "final_model.zip")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Record an action sequence from a trained policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "task",
        choices=sorted(TASKS),
        help="Action to record. Auto-resolves env + default model path "
             "(models/<task>/final_model.zip).",
    )
    ap.add_argument("--model", default=None,
                    help="Override the model path. Default: "
                         "models/<task>/final_model.zip")
    ap.add_argument("--out", default=None,
                    help="Output JSON path. Default: "
                         "sequences/<task>_<actual-seconds>s.json")
    ap.add_argument("--seconds", type=float, default=None,
                    help="Force a fixed recording length, ignoring "
                         "termination. Default: auto — record until the "
                         "policy completes (term or trunc) so the output "
                         "is exactly the length of one action.")
    ap.add_argument("--action-deg", type=float, default=None,
                    help="Override the servo command range used by deploy.py's "
                         "A: cmd. Default: read from the env (per-task).")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    model_path = args.model or default_model_path(args.task)
    if not os.path.isfile(model_path):
        sys.exit(f"Model not found: {model_path}\n"
                 f"Pass --model explicitly, or train it first with:\n"
                 f"  python train_with_video.py --task {args.task}")

    env_cls = TASKS[args.task]
    env = env_cls()
    action_deg = args.action_deg if args.action_deg is not None else env.joint_range_deg
    model = PPO.load(model_path, device="cpu")

    if env.observation_space.shape != model.observation_space.shape:
        sys.exit(f"Obs-space mismatch: env={env.observation_space.shape} "
                 f"model={model.observation_space.shape}. The model was "
                 f"trained against a different env than '{args.task}'.")

    auto_length = args.seconds is None
    if auto_length:
        ceiling_ticks = env.max_steps
        ceiling_label = f"≤{env.max_steps} ticks ({env.max_steps/CONTROL_HZ:.1f}s) — auto-stop on completion"
    else:
        ceiling_ticks = int(args.seconds * CONTROL_HZ)
        ceiling_label = f"{args.seconds:.1f}s ({ceiling_ticks} ticks) — fixed"

    print(f"Task   : {args.task}")
    print(f"Env    : {env_cls.__name__}")
    print(f"Model  : {model_path}")
    print(f"Range  : ±{action_deg:.0f}° (joint command scale)")
    print(f"Length : {ceiling_label}")
    print("-" * 60)

    obs, _ = env.reset(seed=args.seed)
    actions: list[list[int]] = []
    stop_reason = "ceiling reached"
    for _ in range(ceiling_ticks):
        action, _ = model.predict(obs, deterministic=True)
        deg = np.clip(action, -1.0, 1.0) * action_deg
        actions.append(deg.astype(int).tolist())
        obs, _, term, trunc, _ = env.step(action)
        if auto_length and (term or trunc):
            stop_reason = (
                "policy completed (terminated)" if term
                else "env truncated (max_steps)"
            )
            break

    actual_seconds = len(actions) / CONTROL_HZ
    out_path = args.out or os.path.join(
        "sequences", f"{args.task}_{actual_seconds:.1f}s.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(actions, f)

    print(f"Stopped: {stop_reason}")
    print(f"Wrote {len(actions)} actions ({actual_seconds:.2f}s) to {out_path}")


if __name__ == "__main__":
    main()
