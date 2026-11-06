"""
Arc-walking env — an alternative to in-place turning.

Every hip pitches about the same axis, so the robot has no joint that
produces body yaw directly; in-place rotation relies on asymmetric friction.
Walking in an arc only needs a left/right bias in the gait.

  + body-frame forward velocity
  + yaw rate in the requested direction
  + arc bonus: forward velocity × yaw, paid only when both are positive
  − wrong-way yaw, side-slip, tilt, smoothness, jitter

Direction (+1 left, -1 right) is randomized at reset; force one with
options={"direction": ±1}.
"""
import numpy as np
from eshy_env import EshyEnv


FORWARD_REWARD       = 40.0   # per (m/s) body-frame forward velocity
YAW_REWARD           = 8.0    # per (rad/s) yaw in the desired direction
ARC_BONUS            = 80.0   # per (m/s · rad/s)

WRONG_YAW_COST       = 12.0   # yaw in the opposite direction
LATERAL_COST         = 4.0    # body-frame side-slip
TILT_COST            = 8.0    # max(|pitch_z|, |roll_z|)
TILT_FREE_ZONE       = 0.10
SMOOTH_COST          = 2.0    # L2 on action diff
SMOOTH_L1_COST       = 1.5    # L1 on action diff
JITTER_COST          = 1.0    # L2 on leg qvel
JITTER_L1_COST       = 1.0    # L1 on leg qvel

HEALTHY_REWARD       = 0.2

MIN_HEIGHT           = 0.025
PITCH_TERM           = 0.55   # |fwd[2]|
ROLL_TERM            = 0.45   # |right[2]|

MAX_STEPS            = 250    # 10 s @ 25 Hz


class EshyArcEnv(EshyEnv):

    JOINT_RANGE_DEG = 45

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = MAX_STEPS

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if options and "direction" in options:
            self._arc_direction = float(np.sign(options["direction"]) or 1)
        else:
            self._arc_direction = 1.0 if self.np_random.uniform() < 0.5 else -1.0
        return obs, info

    def _get_reward(self, action):
        xmat = self._xmat()
        fwd = xmat[:, 0]
        right = xmat[:, 1]

        v_world = self.data.qvel[0:3]
        forward_vel_bf = float(np.dot(fwd, v_world))
        lateral_vel_bf = float(np.dot(right, v_world))

        # Yaw rate about the body's own up axis.
        omega_world = self.data.qvel[3:6]
        yaw_rate = float(np.dot(xmat[:, 2], omega_world))

        yaw_signed = yaw_rate * self._arc_direction
        yaw_pos = max(yaw_signed, 0.0)
        yaw_neg = max(-yaw_signed, 0.0)

        # Both factors clamped at 0 so reversing while spinning earns nothing.
        arc_bonus = ARC_BONUS * max(forward_vel_bf, 0.0) * yaw_pos

        wrong_yaw_cost = WRONG_YAW_COST * yaw_neg
        lateral_cost = LATERAL_COST * abs(lateral_vel_bf)

        tilt = max(abs(fwd[2]), abs(right[1]))
        tilt_cost = TILT_COST * max(tilt - TILT_FREE_ZONE, 0.0)

        action_diff = action - self.prev_action
        smooth_cost = (SMOOTH_COST * float(np.sum(np.square(action_diff)))
                       + SMOOTH_L1_COST * float(np.sum(np.abs(action_diff))))

        leg_qvel = self.data.qvel[6:10]
        leg_speed = float(np.sum(np.abs(leg_qvel)))
        jitter_cost = (JITTER_COST * float(np.sum(np.square(leg_qvel)))
                       + JITTER_L1_COST * leg_speed)

        forward_reward = FORWARD_REWARD * max(forward_vel_bf, 0.0)
        yaw_reward = YAW_REWARD * yaw_pos

        self._reward_components = {
            "healthy": HEALTHY_REWARD,
            "forward": forward_reward,
            "yaw": yaw_reward,
            "arc_bonus": arc_bonus,
            "wrong_yaw_cost": -wrong_yaw_cost,
            "lateral_cost": -lateral_cost,
            "tilt_cost": -tilt_cost,
            "smooth_cost": -smooth_cost,
            "jitter_cost": -jitter_cost,
            "_diag_direction": self._arc_direction,
            "_diag_forward_vel_bf": forward_vel_bf,
            "_diag_yaw_rate": yaw_rate,
        }

        return (HEALTHY_REWARD
                + forward_reward
                + yaw_reward
                + arc_bonus
                - wrong_yaw_cost
                - lateral_cost
                - tilt_cost
                - smooth_cost
                - jitter_cost)

    def _is_terminated(self):
        if self.data.qpos[2] < MIN_HEIGHT:
            return True
        xmat = self._xmat()
        if abs(xmat[:, 0][2]) > PITCH_TERM:
            return True
        if abs(xmat[:, 1][2]) > ROLL_TERM:
            return True
        return False
