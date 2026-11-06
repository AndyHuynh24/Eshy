"""
Turn-in-place env: end in the canonical standing pose, rotated by TARGET_DEG.

1. World-frame yaw (lean-invariant). Yaw is the angle between the start and
   current torso-forward vectors projected onto the world XY plane, so
   leaning cannot masquerade as rotation.

2. Multiplicative goal reward: GOAL_BONUS * yaw_term * pose_term * still_term.
   Peak reward requires being at the target yaw, in the canonical pose, and
   at rest simultaneously. This makes turns chainable: each call ends in the
   same state the next one starts from.
"""
import numpy as np
from gymnasium import spaces

from eshy_env import EshyEnv


TARGET_DEG       = 20.0
NOMINAL_HEIGHT   = 0.045

GOAL_BONUS       = 30.0
YAW_SIGMA        = np.radians(15) # exp(-|yaw_err|/σ)
JOINT_SCALE      = 0.5            # Σjoints² scale in pose_err
HEIGHT_SCALE     = 0.02           # m
UPRIGHT_SCALE    = 5.0            # weight on (1 - up_z)² in pose_err
STILL_SIGMA      = 0.35
JOINT_VEL_WEIGHT = 0.6

# Cold-start shaping: world yaw rate toward the target, while the exp on
# yaw error is still flat.
YAW_PROGRESS     = 4.0            # per rad/s

POS_COST         = 5.0            # per metre of XY drift from start
SMOOTH_COST      = 1.0            # per Σ(Δaction)²
ALIVE            = 0.2

# Completion: canonical pose at target, held for STABLE_HOLD_STEPS.
COMP_YAW_TOL     = np.radians(4.0)
COMP_JOINT_TOL   = np.radians(10.0)
COMP_HEIGHT_TOL  = 0.010
COMP_UP_MIN      = 0.97           # ~14° tilt
COMP_VEL_MAX     = 0.5
STABLE_HOLD_STEPS= 10

# Completion bonus scales with steps remaining. A flat bonus makes finishing
# early forfeit per-step goal reward, so the policy learns to hover at the
# threshold instead of completing.
COMPLETION_BASE         = 100.0
COMPLETION_PER_REMAINING = 30.0

MIN_HEIGHT       = 0.025
UP_MIN_TERMINATE = 0.50           # up[2] below this = toppled

MAX_STEPS        = 200            # 8 s @ 25 Hz

OBS_EXTRA_DIM    = 2              # [tanh(2·yaw_remaining), |yaw_remaining|/π]


class EshyTurnEnv(EshyEnv):

    JOINT_RANGE_DEG = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = MAX_STEPS
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + OBS_EXTRA_DIM,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)

        target_deg = float(options["target_deg"]) if (
            options and "target_deg" in options) else TARGET_DEG

        # Trained in one direction only. The robot is left/right symmetric and
        # all hips share one axis, so the opposite turn is obtained at deploy
        # time by permuting actions [FL, FR, BL, BR] -> [FR, FL, BR, BL]
        # (deploy.py --mirror).
        if options and "direction" in options:
            direction = float(np.sign(options["direction"]) or 1.0)
        else:
            direction = 1.0

        self._target_yaw   = direction * np.radians(target_deg)
        self._cum_yaw      = 0.0
        self._stable_count = 0
        self._completed    = False
        self._start_xy     = self.data.qpos[0:2].copy()

        fwd_xy = self._xmat()[:2, 0].copy()
        n = float(np.linalg.norm(fwd_xy))
        self._start_fwd_xy = fwd_xy / n if n > 1e-6 else np.array([1.0, 0.0])

        return self._get_obs(), info

    def _get_obs(self):
        base = super()._get_obs()
        if not hasattr(self, "_target_yaw"):
            return np.concatenate([
                base, np.zeros(OBS_EXTRA_DIM, dtype=np.float32)
            ]).astype(np.float32)
        rem = self._target_yaw - self._cum_yaw
        return np.concatenate([
            base,
            np.array([np.tanh(2.0 * rem), abs(rem) / np.pi], dtype=np.float32),
        ]).astype(np.float32)

    def _world_yaw(self, xmat):
        """Signed angle from start_fwd to current fwd in the world XY plane."""
        fwd_xy = xmat[:2, 0]
        n = float(np.linalg.norm(fwd_xy))
        if n < 1e-3:
            # Body almost vertical — yaw signal undefined, hold last value.
            return self._cum_yaw
        f = fwd_xy / n
        s = self._start_fwd_xy[0] * f[1] - self._start_fwd_xy[1] * f[0]
        c = float(np.dot(self._start_fwd_xy, f))
        return float(np.arctan2(s, c))

    def _get_reward(self, action):
        xmat = self._xmat()
        up = xmat[:, 2]

        self._cum_yaw = self._world_yaw(xmat)
        yaw_err = self._target_yaw - self._cum_yaw
        abs_err = abs(yaw_err)

        joints = self.data.qpos[7:11]
        joints_max = float(np.max(np.abs(joints)))
        height = float(self.data.qpos[2])

        pose_err = (
            float(np.sum(joints ** 2)) / JOINT_SCALE
            + ((height - NOMINAL_HEIGHT) / HEIGHT_SCALE) ** 2
            + UPRIGHT_SCALE * (1.0 - up[2]) ** 2
        )

        body_vel_mag = float(np.linalg.norm(self.data.qvel[0:6]))
        joint_vel    = float(np.linalg.norm(self.data.qvel[6:10]))
        stillness    = body_vel_mag + JOINT_VEL_WEIGHT * joint_vel

        yaw_term   = float(np.exp(-abs_err / YAW_SIGMA))
        pose_term  = float(np.exp(-pose_err))
        still_term = float(np.exp(-stillness / STILL_SIGMA))
        goal_reward = GOAL_BONUS * yaw_term * pose_term * still_term

        # World-frame ω_z, so leaning can't fake progress.
        world_yaw_rate = float(self.data.qvel[5])
        err_sign = float(np.sign(yaw_err)) if yaw_err != 0.0 else 0.0
        yaw_progress = YAW_PROGRESS * max(world_yaw_rate * err_sign, 0.0)

        dx = float(self.data.qpos[0] - self._start_xy[0])
        dy = float(self.data.qpos[1] - self._start_xy[1])
        radial = float(np.hypot(dx, dy))
        pos_cost = POS_COST * radial

        action_diff = action - self.prev_action
        smooth_cost = SMOOTH_COST * float(np.sum(action_diff ** 2))

        completion_bonus = 0.0
        pose_match = (abs_err     < COMP_YAW_TOL
                      and joints_max < COMP_JOINT_TOL
                      and abs(height - NOMINAL_HEIGHT) < COMP_HEIGHT_TOL
                      and up[2]      > COMP_UP_MIN
                      and body_vel_mag < COMP_VEL_MAX)
        if pose_match and not self._completed:
            self._stable_count += 1
            if self._stable_count >= STABLE_HOLD_STEPS:
                remaining = max(self.max_steps - self.step_count, 0)
                completion_bonus = (
                    COMPLETION_BASE + COMPLETION_PER_REMAINING * remaining
                )
                self._completed = True
        elif not self._completed:
            self._stable_count = 0

        total = (ALIVE
                 + goal_reward
                 + yaw_progress
                 + completion_bonus
                 - pos_cost
                 - smooth_cost)

        self._reward_components = {
            "alive":              ALIVE,
            "goal_reward":        goal_reward,
            "yaw_progress":       yaw_progress,
            "completion":         completion_bonus,
            "pos_cost":           -pos_cost,
            "smooth_cost":        -smooth_cost,
            "_diag_target_deg":   float(np.degrees(self._target_yaw)),
            "_diag_cum_yaw_deg":  float(np.degrees(self._cum_yaw)),
            "_diag_yaw_err_deg":  float(np.degrees(yaw_err)),
            "_diag_yaw_term":     yaw_term,
            "_diag_pose_term":    pose_term,
            "_diag_still_term":   still_term,
            "_diag_joints_max_deg": float(np.degrees(joints_max)),
            "_diag_up_z":         float(up[2]),
            "_diag_height":       height,
            "_diag_radial":       radial,
            "_diag_stable_count": float(self._stable_count),
            "_diag_pose_match":   float(pose_match),
        }
        return total

    def _is_terminated(self):
        if self.data.qpos[2] < MIN_HEIGHT:
            return True
        if self._xmat()[2, 2] < UP_MIN_TERMINATE:
            return True
        if self._completed:
            return True
        return False
