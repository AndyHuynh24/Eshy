"""
Replay a recorded sequence in the MuJoCo viewer before running it on the
robot (same XML, control rate, and slew limit).

    python view_sequence.py --sequence sequences/walk_10.0s.json --task walk
"""
from __future__ import annotations
import argparse
import json
import os
import time

import numpy as np
import mujoco
import mujoco.viewer

from eshy_env import CONTROL_HZ, MAX_SPEED_DEG_PER_SEC


# Mirrors each task env's JOINT_RANGE_DEG.
TASK_JOINT_RANGE = {
    "walk": 45,
    "sit":  70,
    "lay":  90,
    "turn": 50,
    "arc":  45,
}
DEFAULT_RANGE = 45


LEG_NAMES = ["FL", "FR", "BL", "BR"]   # XML actuator order == JSON order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence",  required=True)
    ap.add_argument("--xml",       default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "eshy.xml"))
    ap.add_argument("--loop",      action="store_true",
                    help="Loop the sequence forever instead of stopping.")
    ap.add_argument("--task", choices=sorted(TASK_JOINT_RANGE),
                    help="Auto-resolve --action-deg from the task. "
                         "Equivalent to passing --action-deg manually.")
    ap.add_argument("--action-deg", type=float, default=None,
                    help="Servo command range used by record_sequence.py / "
                         "deploy.py. Defaults to --task's range, else 45°.")
    args = ap.parse_args()

    if args.action_deg is None:
        args.action_deg = (
            TASK_JOINT_RANGE[args.task] if args.task else DEFAULT_RANGE
        )

    with open(args.sequence) as f:
        sequence = json.load(f)
    n = len(sequence)
    print(f"Loaded {n} actions ({n/CONTROL_HZ:.1f}s @ {CONTROL_HZ} Hz)")
    print(f"Leg index legend: 0={LEG_NAMES[0]}  1={LEG_NAMES[1]}  "
          f"2={LEG_NAMES[2]}  3={LEG_NAMES[3]}")
    print("Trot (diagonals) = indices [0,3] in phase with each other, "
          "opposite phase to [1,2].")

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    xml_order = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(model.nu)]
    print(f"XML actuator order: {xml_order}")
    expected = [f"{ln}_act" for ln in LEG_NAMES]
    assert xml_order == expected, (
        f"XML actuator order {xml_order} does not match LEG_NAMES "
        f"{expected} — sequence indices would be wrong!"
    )

    frame_skip = int(1.0 / (CONTROL_HZ * model.opt.timestep))
    joint_range_rad = np.radians(args.action_deg)

    max_delta_per_tick_rad = np.radians(MAX_SPEED_DEG_PER_SEC / CONTROL_HZ)
    prev_ctrl = np.zeros(4, dtype=np.float32)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        log_every = int(CONTROL_HZ)
        t_start = time.monotonic()
        i = 0
        while viewer.is_running():
            if i >= n:
                if not args.loop:
                    break
                i = 0
                mujoco.mj_resetData(model, data)
                prev_ctrl[:] = 0.0
                t_start = time.monotonic()

            target_rad = np.clip(
                np.asarray(sequence[i], dtype=np.float32),
                -args.action_deg, args.action_deg,
            ) * (joint_range_rad / args.action_deg)
            delta = np.clip(target_rad - prev_ctrl,
                            -max_delta_per_tick_rad,
                            +max_delta_per_tick_rad)
            prev_ctrl = prev_ctrl + delta
            data.ctrl[:] = prev_ctrl

            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            viewer.sync()

            if i % log_every == 0:
                deg = [int(d) for d in sequence[i]]
                labelled = "  ".join(
                    f"{LEG_NAMES[k]}={deg[k]:+3d}" for k in range(4)
                )
                print(f"t={i/CONTROL_HZ:5.1f}s  {labelled}")

            elapsed = time.monotonic() - t_start
            target = (i + 1) / CONTROL_HZ
            if target > elapsed:
                time.sleep(target - elapsed)
            i += 1

        print("Viewer closed.")


if __name__ == "__main__":
    main()
