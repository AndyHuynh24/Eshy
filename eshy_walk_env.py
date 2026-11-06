"""
Walk env — straight-line forward locomotion.

Shaping terms are kept small relative to forward velocity:

    forward_vel              +1.0 per m/s
    alive                    +0.1 per step
  − smoothness_cost           0.5 * Σ(Δa)²
  − lateral_cost              1.5 * v_y²
  − heading_cost              1.5 * max(|drift| − 5°, 0)

A walking gait is, by definition, joints reversing direction every step,
which is exactly what the base env's anti-jitter cost punishes. With it,
the policy learns to twitch instead of stepping, so walk disables it.
"""
import numpy as np

from eshy_env import EshyEnv, HEALTHY_REWARD


SMOOTHNESS_COST  = 0.5
LATERAL_COST     = 1.5
HEADING_COST     = 1.5             # per rad past free zone
HEADING_FREE     = np.radians(5)   # free zone — natural sway is fine


class EshyWalkEnv(EshyEnv):

    JOINT_RANGE_DEG = 45

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        fwd_xy = self._xmat()[:2, 0].copy()
        n = float(np.linalg.norm(fwd_xy))
        self._start_fwd_xy = fwd_xy / n if n > 1e-6 else np.array([1.0, 0.0])
        return obs, info

    def _heading_drift(self):
        """Signed world-XY angle from the start heading (lean-invariant)."""
        fwd_xy = self._xmat()[:2, 0]
        n = float(np.linalg.norm(fwd_xy))
        if n < 1e-3:
            return 0.0
        f = fwd_xy / n
        s = self._start_fwd_xy[0] * f[1] - self._start_fwd_xy[1] * f[0]
        c = float(np.dot(self._start_fwd_xy, f))
        return float(np.arctan2(s, c))

    def _get_reward(self, action):
        forward_vel = float(self.data.qvel[0])
        lateral_vel = float(self.data.qvel[1])

        action_diff = action - self.prev_action
        smoothness_cost = SMOOTHNESS_COST * float(np.sum(np.square(action_diff)))
        lateral_cost    = LATERAL_COST    * (lateral_vel ** 2)

        drift = self._heading_drift()
        heading_cost = HEADING_COST * max(abs(drift) - HEADING_FREE, 0.0)

        self._reward_components = {
            "alive":           HEALTHY_REWARD,
            "forward_vel":     forward_vel,
            "smoothness_cost": -smoothness_cost,
            "lateral_cost":    -lateral_cost,
            "heading_cost":    -heading_cost,
            "_diag_drift_deg": float(np.degrees(drift)),
        }
        return (forward_vel + HEALTHY_REWARD
                - smoothness_cost - lateral_cost - heading_cost)

    def _common_motion_costs(self, limited_action):
        """No anti-jitter for walk (see module docstring)."""
        action_diff = limited_action - self.prev_action
        self.prev_action_diff = action_diff.copy()
        return 0.0, {}
