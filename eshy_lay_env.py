"""
Lay env: body flat and as low as possible, with a gentle landing.

  + lay reward: torso level AND low
  − feet off the floor, side roll, sudden commands, vertical impact,
    joint velocity, scooting
"""
import numpy as np

from eshy_env import EshyEnv


TORSO_STANDING_Z = 0.045
LAY_TARGET_Z     = 0.020    # torso centre ~2 cm above the floor
UPRIGHT_MIN      = 0.3
MIN_HEIGHT       = 0.005


class EshyLayEnv(EshyEnv):
    JOINT_RANGE_DEG = 90    # full splay to flatten the body

    def _up_axis(self):
        return self._xmat()[:, 2]

    def _foot_heights(self):
        return np.array([
            self.data.geom("FL_foot").xpos[2],
            self.data.geom("FR_foot").xpos[2],
            self.data.geom("BL_foot").xpos[2],
            self.data.geom("BR_foot").xpos[2],
        ], dtype=np.float64)

    def _get_reward(self, action):
        up = self._up_axis()

        flatness = max(up[2], 0.0) ** 2

        # Drop from standing toward LAY_TARGET_Z, normalised to 0..1.
        drop = max(TORSO_STANDING_Z - self.data.qpos[2], 0.0)
        drop_norm = min(drop / (TORSO_STANDING_Z - LAY_TARGET_Z), 1.0)

        lay_reward = 25.0 * flatness * drop_norm

        contact_cost = 8.0 * float(np.sum(np.maximum(self._foot_heights(), 0.0)))
        balance_cost = 3.0 * abs(up[1])

        action_diff = action - self.prev_action
        smooth_cost = 8.0 * float(np.sum(np.square(action_diff)))

        impact_cost = 5.0 * float(self.data.qvel[2] ** 2)
        jitter_cost = 3.0 * float(np.sum(np.square(self.data.qvel[6:10])))

        vx, vy = self.data.qvel[0], self.data.qvel[1]
        wander_cost = 0.5 * (abs(vx) + abs(vy)) + 0.3 * abs(self.data.qvel[5])

        alive = 0.3 * max(up[2], 0.0)

        self._reward_components = {
            "alive":        alive,
            "lay_reward":   lay_reward,
            "contact_cost": -contact_cost,
            "balance_cost": -balance_cost,
            "smooth_cost":  -smooth_cost,
            "impact_cost":  -impact_cost,
            "jitter_cost":  -jitter_cost,
            "wander_cost":  -wander_cost,
        }

        return (alive + lay_reward
                - contact_cost - balance_cost
                - smooth_cost - impact_cost - jitter_cost - wander_cost)

    def _is_terminated(self):
        if self.data.qpos[2] < MIN_HEIGHT:     return True
        if self._up_axis()[2] < UPRIGHT_MIN:   return True
        return False
