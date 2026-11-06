"""
Drive the real robot over BLE from the laptop.

Modes:
    --sequence FILE   open-loop replay of a sequence from record_sequence.py
    (default)         closed-loop: build the 16-dim observation from robot
                      telemetry, run the PPO policy, stream A: commands
    --range-test      sweep each leg through ±action-deg to verify the sign map

Observation layout (matches EshyEnv._get_obs):
    joint_pos (4)      rad, + = leg forward
    joint_vel (4)      rad/s, finite-differenced from servo telemetry
    body_quat_xyz (3)  from the on-robot Madgwick filter
    body_angvel (3)    gyro, rad/s
    phase (2)          [sin 2πφ, cos 2πφ], φ advancing at GAIT_FREQ_HZ

Usage:
    python deploy.py --sequence sequences/walk_10.0s.json
    python deploy.py --model models/walk/final_model.zip
    python deploy.py --range-test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import signal
import time

import numpy as np
from bleak import BleakClient, BleakScanner
from stable_baselines3 import PPO

# Must match eshy_env.py / the task env.
JOINT_RANGE_DEG        = 45.0
GAIT_FREQ_HZ           = 1.0
CONTROL_HZ             = 25.0
MAX_SPEED_DEG_PER_SEC  = 120.0

# Firmware applies servo_deg = 90 + SERVO_SIGN[i] * robot_deg.
SERVO_SIGN = np.array([-1, 1, -1, 1], dtype=np.float32)  # [FL, FR, BL, BR]

DEVICE_NAME  = "Eshy"
UART_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

# G: and Q: fields are optional.
TELEM_RE = re.compile(
    r"R:(?P<r>[+-]?[\d.]+)\s+"
    r"P:(?P<p>[+-]?[\d.]+)\s+"
    r"Y:(?P<y>[+-]?[\d.]+)\s+"
    r"S:(?P<s0>\d+),(?P<s1>\d+),(?P<s2>\d+),(?P<s3>\d+)"
    r"(?:\s+G:(?P<gx>[+-]?[\d.]+),(?P<gy>[+-]?[\d.]+),(?P<gz>[+-]?[\d.]+))?"
    r"(?:\s+Q:(?P<qw>[+-]?[\d.]+),(?P<qx>[+-]?[\d.]+),"
          r"(?P<qy>[+-]?[\d.]+),(?P<qz>[+-]?[\d.]+))?"
)


class LiveState:
    """Latest telemetry, written by the BLE notify callback and read by the
    control loop (same event loop, so no locking needed)."""

    def __init__(self):
        self.servos_deg = np.array([90, 90, 90, 90], dtype=np.float32)
        self.prev_servos_deg = np.array([90, 90, 90, 90], dtype=np.float32)
        self.last_servo_t = time.monotonic()
        self.gyro = np.zeros(3, dtype=np.float32)
        self.quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.rpy_deg = np.zeros(3, dtype=np.float32)
        self.got_packet = False

    def feed_line(self, line: str) -> None:
        m = TELEM_RE.search(line)
        if not m:
            return
        d = m.groupdict()
        now = time.monotonic()
        servos = np.array(
            [float(d["s0"]), float(d["s1"]), float(d["s2"]), float(d["s3"])],
            dtype=np.float32,
        )
        self.prev_servos_deg = self.servos_deg
        self.servos_deg = servos
        self.last_servo_t_prev = getattr(self, "last_servo_t", now)
        self.last_servo_t = now

        self.rpy_deg = np.array(
            [float(d["r"]), float(d["p"]), float(d["y"])], dtype=np.float32
        )
        if d["gx"] is not None:
            self.gyro = np.array(
                [float(d["gx"]), float(d["gy"]), float(d["gz"])],
                dtype=np.float32,
            )
        if d["qw"] is not None:
            self.quat_wxyz = np.array(
                [float(d["qw"]), float(d["qx"]),
                 float(d["qy"]), float(d["qz"])],
                dtype=np.float32,
            )
        else:
            self.quat_wxyz = rpy_deg_to_quat_wxyz(*self.rpy_deg)
        self.got_packet = True


def rpy_deg_to_quat_wxyz(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg); p = math.radians(pitch_deg); y = math.radians(yaw_deg)
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ], dtype=np.float32)


def servo_deg_to_robot_rad(servos_deg: np.ndarray) -> np.ndarray:
    """Inverse of the firmware sign map. servo=90 → robot=0 rad."""
    robot_deg = (servos_deg - 90.0) * SERVO_SIGN
    return np.deg2rad(robot_deg).astype(np.float32)


def robot_rad_to_action_deg(action_rad: np.ndarray, action_range_deg: float) -> np.ndarray:
    """Policy output (rad) → clipped robot-frame degrees for the A: command."""
    deg = np.rad2deg(action_rad)
    return np.clip(deg, -action_range_deg, action_range_deg).astype(np.int32)


def build_obs(live: LiveState, phase: float) -> np.ndarray:
    joint_pos = servo_deg_to_robot_rad(live.servos_deg)             # (4,)

    dt = max(live.last_servo_t - getattr(live, "last_servo_t_prev",
                                         live.last_servo_t - 0.02), 1e-3)
    joint_vel_deg = (live.servos_deg - live.prev_servos_deg) * SERVO_SIGN / dt
    joint_vel = np.deg2rad(joint_vel_deg).astype(np.float32)        # (4,)

    body_quat_xyz = live.quat_wxyz[1:4].astype(np.float32)          # (3,)
    body_angvel   = live.gyro.astype(np.float32)                    # (3,)
    phase_sig = np.array(
        [math.sin(2*math.pi*phase), math.cos(2*math.pi*phase)],
        dtype=np.float32,
    )                                                                # (2,)

    return np.concatenate(
        [joint_pos, joint_vel, body_quat_xyz, body_angvel, phase_sig]
    ).astype(np.float32)


async def find_device(name: str, timeout: float = 10.0):
    print(f"Scanning for '{name}'...")
    dev = await BleakScanner.find_device_by_name(name, timeout=timeout)
    if dev is None:
        raise SystemExit(f"Could not find '{name}' — is it powered and advertising?")
    print(f"Found {dev.name} @ {dev.address}")
    return dev


async def run(args: argparse.Namespace) -> None:
    model = None
    if not args.range_test and not args.sequence:
        print(f"Loading policy: {args.model}")
        model = PPO.load(args.model, device="cpu")
        print(f"Policy obs_dim = {model.observation_space.shape}, "
              f"action_dim = {model.action_space.shape}")

    live = LiveState()
    dev = await find_device(DEVICE_NAME, args.scan_timeout)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async with BleakClient(dev) as client:
        print("Connected. Waiting for GATT service discovery...")
        # CoreBluetooth can return from connect() before discovery finishes.
        svc_ok = False
        for _ in range(50):                       # up to ~5 s
            svcs = client.services
            if svcs is not None:
                uuids = {str(s.uuid).lower() for s in svcs}
                if any(u.startswith("6e400001") for u in uuids):
                    svc_ok = True
                    break
            await asyncio.sleep(0.1)
        if not svc_ok:
            raise SystemExit("UART service never appeared — is the firmware running?")

        print("Services discovered. Subscribing to telemetry...")
        rx_buf = bytearray()

        def on_tx(_, data: bytearray) -> None:
            nonlocal rx_buf
            rx_buf += data
            while True:
                idx = rx_buf.find(b"\n")
                if idx < 0:
                    break
                line = bytes(rx_buf[:idx]).decode("utf-8", errors="ignore")
                del rx_buf[: idx + 1]
                live.feed_line(line)

        await client.start_notify(UART_TX_UUID, on_tx)

        t0 = time.monotonic()
        while not live.got_packet:
            if time.monotonic() - t0 > 8.0:
                raise SystemExit("No telemetry received within 8s.")
            await asyncio.sleep(0.05)
        print("Telemetry OK.")

        # response=False: macOS sometimes never ACKs the first
        # write-with-response and the call hangs.
        await client.write_gatt_char(UART_RX_UUID, b"i\n", response=False)
        await asyncio.sleep(0.1)

        if args.range_test:
            print(f"RANGE TEST: sweeping each leg through ±{args.action_deg:.0f}° "
                  f"(expect servo {90 - int(args.action_deg)}..{90 + int(args.action_deg)}).")
            leg_names = ["FL", "FR", "BL", "BR"]
            # Re-send each target for hold_secs so the slew-limited servo
            # converges before we check it.
            hold_secs = 1.5
            hold_dt   = 1.0 / args.hz
            hold_n    = int(hold_secs / hold_dt)
            for leg_idx in range(4):
                print(f"--- Leg {leg_idx} ({leg_names[leg_idx]}) ---")
                for target in (+args.action_deg, -args.action_deg, 0.0):
                    cmd_deg = np.zeros(4, dtype=np.int32)
                    cmd_deg[leg_idx] = int(target)
                    cmd = (
                        f"A:{cmd_deg[0]},{cmd_deg[1]},{cmd_deg[2]},{cmd_deg[3]}\n"
                    ).encode()
                    for _ in range(hold_n):
                        await client.write_gatt_char(UART_RX_UUID, cmd, response=False)
                        await asyncio.sleep(hold_dt)
                    expected = 90 + int(SERVO_SIGN[leg_idx] * target)
                    actual = int(live.servos_deg[leg_idx])
                    ok = "OK" if abs(actual - expected) <= 3 else "MISMATCH"
                    print(f"  cmd={int(target):+4d}°  expect_servo={expected:3d}  "
                          f"actual_servo={actual:3d}  [{ok}]")
            await client.write_gatt_char(UART_RX_UUID, b"c\n", response=False)
            await asyncio.sleep(0.1)
            print("Range test complete.  Sent 'c' to center.")
            return

        sequence: list[list[int]] | None = None
        if args.sequence:
            with open(args.sequence) as f:
                sequence = json.load(f)
            print(f"Loaded {len(sequence)} actions from {args.sequence} "
                  f"({len(sequence)/args.hz:.1f}s @ {args.hz} Hz, open-loop replay)")
        else:
            print("Starting inference.")

        dt = 1.0 / args.hz
        phase = 0.0
        tick_count = 0
        log_interval = int(args.hz)
        # Same slew limit as the env, so real servos move like sim.
        max_delta_deg = args.max_speed / args.hz
        prev_cmd_deg = np.zeros(4, dtype=np.float32)

        sim_model = sim_data = sim_viewer = None
        sim_frame_skip = 0
        if args.sim_mirror:
            import mujoco
            import mujoco.viewer
            sim_model = mujoco.MjModel.from_xml_path(args.sim_xml)
            sim_data  = mujoco.MjData(sim_model)
            sim_frame_skip = int(1.0 / (args.hz * sim_model.opt.timestep))
            mujoco.mj_resetData(sim_model, sim_data)
            sim_data.qpos[2] = 0.045   # EshyEnv.reset() start height
            try:
                sim_viewer = mujoco.viewer.launch_passive(sim_model, sim_data)
            except RuntimeError as e:
                raise SystemExit(
                    f"MuJoCo viewer failed: {e}\n"
                    f"On macOS, run deploy.py under mjpython, e.g.:\n"
                    f"  mjpython deploy.py "
                    f"--sim-mirror --sequence sequences/walk_10.0s.json"
                )
            print("Sim-mirror: MuJoCo viewer open.  Sim left / real right.")
        try:
            next_t = time.monotonic()
            while not stop.is_set():
                now = time.monotonic()
                if now < next_t:
                    await asyncio.sleep(next_t - now)
                next_t += dt

                if sequence is not None:
                    if tick_count >= len(sequence):
                        print("Sequence finished.")
                        break
                    target_deg = np.asarray(sequence[tick_count], dtype=np.float32)
                else:
                    obs = build_obs(live, phase)
                    action, _ = model.predict(obs, deterministic=True)
                    if args.mirror:
                        # [FL, FR, BL, BR] -> [FR, FL, BR, BL]
                        action = np.array([action[1], action[0],
                                           action[3], action[2]],
                                          dtype=action.dtype)
                    target_deg = np.clip(action, -1.0, 1.0) * args.action_deg

                delta = np.clip(target_deg - prev_cmd_deg, -max_delta_deg, max_delta_deg)
                prev_cmd_deg = prev_cmd_deg + delta
                cmd_deg = np.clip(prev_cmd_deg, -args.action_deg, args.action_deg).astype(np.int32)
                cmd = (
                    f"A:{cmd_deg[0]},{cmd_deg[1]},{cmd_deg[2]},{cmd_deg[3]}\n"
                ).encode()
                try:
                    await client.write_gatt_char(UART_RX_UUID, cmd, response=False)
                except Exception as e:
                    # Transient CoreBluetooth errors: log and keep going.
                    if tick_count % log_interval == 0:
                        print(f"write failed: {e}")
                    continue

                if sim_viewer is not None and sim_viewer.is_running():
                    sim_data.ctrl[:] = np.deg2rad(prev_cmd_deg)
                    for _ in range(sim_frame_skip):
                        mujoco.mj_step(sim_model, sim_data)
                    sim_viewer.sync()

                phase = (phase + GAIT_FREQ_HZ * dt) % 1.0
                tick_count += 1
                if tick_count % log_interval == 0:
                    print(
                        f"t={tick_count/args.hz:5.1f}s  "
                        f"act={cmd_deg.tolist()}  "
                        f"servo={live.servos_deg.astype(int).tolist()}  "
                        f"rpy={live.rpy_deg.round(1).tolist()}  "
                        f"gyro={live.gyro.round(2).tolist()}"
                    )
        finally:
            # Best-effort park; the firmware also centers on disconnect.
            try:
                if client.is_connected:
                    await client.write_gatt_char(UART_RX_UUID, b"c\n", response=False)
                    await asyncio.sleep(0.1)
                    await client.stop_notify(UART_TX_UUID)
            except Exception as e:
                print(f"cleanup: {e}")
            if sim_viewer is not None:
                try:
                    sim_viewer.close()
                except Exception:
                    pass
            print("Stopped. Servos centered.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/walk/final_model.zip")
    ap.add_argument("--hz", type=float, default=CONTROL_HZ,
                    help="Command rate in Hz (training: 25)")
    ap.add_argument("--action-deg", type=float, default=JOINT_RANGE_DEG,
                    help="Policy action scale in robot-frame degrees "
                         "(the task env's JOINT_RANGE_DEG)")
    ap.add_argument("--max-speed", type=float, default=MAX_SPEED_DEG_PER_SEC,
                    help="Max servo slew rate in deg/s (training: 120)")
    ap.add_argument("--scan-timeout", type=float, default=10.0)
    ap.add_argument("--range-test", action="store_true",
                    help="Skip the policy and sweep each leg through "
                         "±action-deg to verify sign map and scaling.")
    ap.add_argument("--sequence", default=None,
                    help="JSON sequence from record_sequence.py, replayed "
                         "open-loop (no policy inference).")
    ap.add_argument("--mirror", action="store_true",
                    help="Swap left/right legs on every policy action "
                         "(turn-left policy -> turn-right).")
    ap.add_argument("--sim-mirror", action="store_true",
                    help="Step a MuJoCo viewer with the same commands for a "
                         "side-by-side sim/real comparison (run under "
                         "mjpython on macOS).")
    ap.add_argument("--sim-xml", default="eshy.xml")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
