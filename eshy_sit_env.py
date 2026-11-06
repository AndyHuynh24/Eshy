"""
Sit env.

  + nose up (capped at NOSE_UP_TARGET)
  + rear of torso on the ground
  + hold still once sitting
  − feet off the floor, legs colliding
  − body translating, tipping, or jittering
"""
import numpy as np
import mujoco
from eshy_env import EshyEnv, CONTROL_HZ


NOSE_UP_TARGET     = 0.3          # fwd[2] = sin(pitch); 0.30 ≈ 17°

NOSE_UP_REWARD     = 25.0
REAR_TOUCH_BASE    = 15.0
REAR_TOUCH_BONUS   = 25.0
SIT_HOLD_REWARD    = 30.0
ALIVE              = 0.3

# Torso half-extents (must match eshy.xml)
HALF_X = 0.0445
HALF_Z = 0.0186

# Penalises hovering at half-height instead of committing to the sit.
REAR_SIT_Z         = 0.015
HEIGHT_LOITER_RATE = 8.0

FEET_COST          = 100.0
LEG_COLLISION_COST = 5.0
BACK_BACKWARD_COST = 0.5
BODY_ROT_COST      = 1.0
SMOOTH_COST        = 8.0
SMOOTH_L1_COST     = 4.0
JITTER_COST        = 8.0
JITTER_L1_COST     = 8.0
TRANSLATE_COST     = 6.0
SYMMETRY_COST      = 0.05

MIN_HEIGHT         = 0.001
UPRIGHT_MIN        = 0.3


class EshySitEnv(EshyEnv):

    JOINT_RANGE_DEG = 70    # rear legs fold under the body

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._loiter_time = 0.0
        self._dt = 1.0 / CONTROL_HZ
        # 30 deg/s slew cap forces a slow, deliberate sit.
        self.max_action_delta = (30.0 / CONTROL_HZ) / self.joint_range_deg

        self._leg_geom_ids = set()
        for leg in ("FL", "FR", "BL", "BR"):
            for part in ("geom", "foot"):
                gid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_{part}"
                )
                if gid >= 0:
                    self._leg_geom_ids.add(gid)

    def reset(self, *args, **kwargs):
        self._loiter_time = 0.0
        return super().reset(*args, **kwargs)

    def _rear_bottom_z(self):
        torso = self.data.body("torso")
        offset = np.array([-HALF_X, 0.0, -HALF_Z])
        return float(torso.xpos[2] + (self._xmat() @ offset)[2])

    def _foot_heights(self):
        return np.array([
            self.data.geom("FL_foot").xpos[2],
            self.data.geom("FR_foot").xpos[2],
            self.data.geom("BL_foot").xpos[2],
            self.data.geom("BR_foot").xpos[2],
        ], dtype=np.float64)

    def _count_leg_leg_contacts(self):
        n = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self._leg_geom_ids and c.geom2 in self._leg_geom_ids:
                n += 1
        return n

    def _get_reward(self, action):
        xmat = self._xmat()
        fwd = xmat[:, 0]
        upright_gate = max(xmat[2, 2], 0.0)

        # Per-foot contact: 1 on the floor, 0 at 8 mm up; averaged.
        foot_z = self._foot_heights()
        per_foot = np.clip(1.0 - np.maximum(foot_z, 0.0) / 0.008, 0.0, 1.0)
        foot_contact_gate = float(np.mean(per_foot))

        nose_up_factor = min(max(fwd[2], 0.0), NOSE_UP_TARGET) / NOSE_UP_TARGET
        rear_z = max(self._rear_bottom_z(), 0.0)
        rear_low_factor = max(1.0 - rear_z / 0.03, 0.0)

        nose_up_reward = (NOSE_UP_REWARD * nose_up_factor * NOSE_UP_TARGET
                          * upright_gate * foot_contact_gate)

        # Base term gives gradient before nose-up is solved; the bonus pays
        # only for the full pose, so it dominates.
        rear_touch_reward = (
            REAR_TOUCH_BASE * rear_low_factor * upright_gate
            + REAR_TOUCH_BONUS * rear_low_factor * nose_up_factor
              * upright_gate * foot_contact_gate
        )

        # 1/(1+k·motion): sharp gradient toward freezing once seated.
        leg_qvel = self.data.qvel[6:10]
        leg_speed = float(np.sum(np.abs(leg_qvel)))
        sit_quality = (rear_low_factor * nose_up_factor
                       * foot_contact_gate * upright_gate)
        sit_hold_reward = SIT_HOLD_REWARD * sit_quality / (1.0 + 20.0 * leg_speed)

        # Grows with time spent above REAR_SIT_Z × height above it.
        if rear_z > REAR_SIT_Z:
            self._loiter_time += self._dt
        else:
            self._loiter_time = 0.0
        height_loiter_cost = (HEIGHT_LOITER_RATE * self._loiter_time
                              * max(rear_z - REAR_SIT_Z, 0.0))

        feet_cost = FEET_COST * float(np.sum(np.maximum(foot_z, 0.0)))
        leg_collision_cost = LEG_COLLISION_COST * self._count_leg_leg_contacts()

        hip_deg = np.degrees(self.data.qpos[7:11])
        symmetry_cost = SYMMETRY_COST * (
            abs(hip_deg[0] - hip_deg[1]) + abs(hip_deg[2] - hip_deg[3])
        )
        back_backward_cost = BACK_BACKWARD_COST * (
            max(-hip_deg[2], 0.0) + max(-hip_deg[3], 0.0)
        )

        # L2 + L1: the L1 term keeps a constant gradient near zero, so
        # micro-motion is driven to zero rather than decaying quadratically.
        action_diff = action - self.prev_action
        smooth_cost = (SMOOTH_COST * float(np.sum(np.square(action_diff)))
                       + SMOOTH_L1_COST * float(np.sum(np.abs(action_diff))))
        jitter_cost = (JITTER_COST * float(np.sum(np.square(leg_qvel)))
                       + JITTER_L1_COST * leg_speed)
        translate_cost = TRANSLATE_COST * (
            abs(self.data.qvel[0]) + abs(self.data.qvel[1])
        )
        body_rot_cost = BODY_ROT_COST * float(
            np.sum(np.square(self.data.qvel[3:6]))
        )

        alive = ALIVE * upright_gate

        self._reward_components = {
            "alive": alive,
            "nose_up": nose_up_reward,
            "rear_touch": rear_touch_reward,
            "sit_hold": sit_hold_reward,
            "feet_cost": -feet_cost,
            "leg_collision_cost": -leg_collision_cost,
            "back_backward_cost": -back_backward_cost,
            "body_rot_cost": -body_rot_cost,
            "symmetry_cost": -symmetry_cost,
            "smooth_cost": -smooth_cost,
            "jitter_cost": -jitter_cost,
            "translate_cost": -translate_cost,
            "height_loiter_cost": -height_loiter_cost,
            "_diag_leg_speed": leg_speed,
            "_diag_rear_z": rear_z,
            "_diag_nose_up_factor": nose_up_factor,
            "_diag_foot_contact": foot_contact_gate,
        }

        return (alive
                + nose_up_reward
                + rear_touch_reward
                + sit_hold_reward
                - feet_cost
                - leg_collision_cost
                - back_backward_cost
                - body_rot_cost
                - symmetry_cost
                - smooth_cost
                - jitter_cost
                - translate_cost
                - height_loiter_cost)

    def _is_terminated(self):
        if self.data.qpos[2] < MIN_HEIGHT:
            return True
        if self._xmat()[2, 2] < UPRIGHT_MIN:
            return True
        return False