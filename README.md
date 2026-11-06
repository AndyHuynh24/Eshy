# Eshy — RL-Trained Quadruped Robot

https://github.com/user-attachments/assets/afd2cbcb-715b-4916-834a-9237a6f70c31

A 116 g, 4-DOF quadruped that learns to walk and sit with reinforcement
learning. Each skill is trained with PPO in a MuJoCo model of the robot, then
run on the real hardware over BLE.

## Hardware

- Seeed XIAO ESP32-S3 Sense
- 4× MG90S servos (hip pitch only, 40 mm rigid legs)
- MPU-6050 IMU, 3× WS2812B status LEDs
- 3.7 V 1000 mAh LiPo + 5 V boost

## How it works

1. **Simulation.** `eshy.xml` models the robot with actuator torque limits
   and slew rates taken from the MG90S datasheet. Mass, friction, and servo
   strength are randomized every episode.
2. **Training.** Each skill is a subclass of `EshyEnv` with its own reward
   (`eshy_walk_env.py`, `eshy_sit_env.py`, ...). `train_with_video.py` trains
   a 2×128 MLP with PPO and logs every reward term to TensorBoard.
3. **Recording.** `record_sequence.py` runs the trained policy in sim and
   saves its actions as a 25 Hz joint-angle sequence.
4. **Deployment.** `deploy.py` streams the sequence to the robot over BLE.
   The firmware (`firmware/`) clips each command, slew-limits it, and drives
   one servo per command so the 5 V boost doesn't brown out.

## Repository layout

| Path | Purpose |
|---|---|
| `eshy.xml` | MuJoCo robot model |
| `eshy_env.py` | Base Gymnasium env: observation, action rate limiting, domain randomization |
| `eshy_{walk,sit,turn,arc,lay}_env.py` | Task rewards and termination |
| `train_with_video.py` | PPO training with videos and per-term reward logging |
| `record_sequence.py` | Record a trained policy's actions to `sequences/` |
| `view_sequence.py` | Replay a sequence in the MuJoCo viewer |
| `deploy.py` | BLE host: sequence replay, closed-loop policy, or leg range test |
| `controller.py` | Manual slider controller and telemetry viewer (pygame) |
| `export_model.py` | Export policy weights to a C header |
| `firmware/` | Main ESP-IDF firmware: BLE command interface and telemetry |
| `firmware_walk/` | Experimental firmware that runs the policy on the ESP32 |
| `models/<task>/final_model.zip` | Trained policies |
| `sequences/` | Recorded action sequences |
| `docs/VISION_NAV_PLAN.md` | Plan for camera-based navigation |

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Train, inspect, and record:

```bash
python train_with_video.py --task walk          # logs/, models/, videos/
tensorboard --logdir logs/
python record_sequence.py walk --seconds 10     # -> sequences/walk_10.0s.json
python view_sequence.py --sequence sequences/walk_10.0s.json --task walk
```

Flash the firmware and run it on the robot:

```bash
cd firmware && idf.py set-target esp32s3 && idf.py build flash monitor
cd .. && python deploy.py --sequence sequences/walk_10.0s.json
python deploy.py --range-test                   # check leg signs and ranges
```

On macOS, add `--sim-mirror` and launch with `mjpython` to watch the
simulation next to the real robot.

## BLE protocol

Nordic UART Service, newline-terminated text commands:

| Command | Action |
|---|---|
| `c` / `i` / `s` | center / IMU-only / sweep |
| `A:fl,fr,bl,br` | robot-frame hip angles in degrees (+ = forward) |
| `M:fl,fr,bl,br` | raw servo angles in degrees |
| `O:fl,fr,bl,br` | per-leg trim offsets |
| `R:n` | slew limit, degrees per command |

Telemetry at 50 Hz: `R:roll P:pitch Y:yaw S:fl,fr,bl,br G:gx,gy,gz Q:qw,qx,qy,qz`

## Status and known issues

- Walking and sitting work on the robot as open-loop playback of sequences
  recorded from the trained policies. `firmware/` ships with
  `DISABLE_MPU 1`, so closed-loop inference needs the IMU turned back on.
- `firmware_walk/` is behind the current training code. Its weights come
  from an older model and `ACTION_RANGE_DEG` is 32, but the walk env uses 45.
  Re-export with `export_model.py` and update the constant before using it.
- `firmware/` and `firmware_walk/` swap the BL and BR pins (GPIO3 / GPIO4).
  Check against your wiring before flashing.
- Turning and camera-based navigation are in progress.
