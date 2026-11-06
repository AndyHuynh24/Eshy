"""
Train a PPO policy for one task, with periodic 3-view videos and per-term
reward logging.

Each reward component is logged to TensorBoard as `reward/<name>`: the
per-episode total, averaged over episodes finished in the rollout. Costs are
stored negated; `_diag_*` entries are diagnostics and not part of the reward.

Usage:
    python train_with_video.py --task walk
    python train_with_video.py --task sit --timesteps 3000000
    python train_with_video.py --task walk --resume models/walk/final_model.zip
    tensorboard --logdir logs/
"""
import argparse
import os
import numpy as np
import imageio
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from eshy_walk_env import EshyWalkEnv
from eshy_sit_env import EshySitEnv
from eshy_lay_env import EshyLayEnv
from eshy_turn_env import EshyTurnEnv
from eshy_arc_env import EshyArcEnv


TASKS = {
    "walk": {"cls": EshyWalkEnv,
             "desc": "Walk forward (forward vel, smoothness, lateral penalty)"},
    "sit":  {"cls": EshySitEnv,
             "desc": "Sit — nose up, body lowered, holds still after sit-down"},
    "lay":  {"cls": EshyLayEnv,
             "desc": "Lay down — body flat & low, all 4 feet on floor"},
    "turn": {"cls": EshyTurnEnv,
             "desc": "Turn in place to TARGET_DEG, ending in the standing pose"},
    "arc":  {"cls": EshyArcEnv,
             "desc": "Continuous arc-walk (forward vel + yaw in chosen direction). "
                     "Bidirectional — direction randomized per episode."},
}


class VideoRecorderCallback(BaseCallback):
    """Records a 3-view composite video every video_freq steps."""

    def __init__(self, eval_env, video_dir, video_freq=50000,
                 video_length=300, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_dir = video_dir
        self.video_freq = video_freq
        self.video_length = video_length
        self.last_video_step = 0
        os.makedirs(video_dir, exist_ok=True)

        self.r_main = mujoco.Renderer(self.eval_env.model, height=480, width=320)
        self.r_side = mujoco.Renderer(self.eval_env.model, height=240, width=320)
        self.r_top  = mujoco.Renderer(self.eval_env.model, height=240, width=320)

    def _on_step(self):
        if self.num_timesteps - self.last_video_step >= self.video_freq:
            self.last_video_step = self.num_timesteps
            self._record_video()
        return True

    def _record_video(self):
        frames = []
        obs, _ = self.eval_env.reset()
        episode_reward = 0
        distance = 0

        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = self.eval_env.step(action)
            episode_reward += reward
            distance = info.get("x_pos", 0)

            self.r_main.update_scene(self.eval_env.data, camera="track")
            left = self.r_main.render()
            self.r_side.update_scene(self.eval_env.data, camera="side")
            tr = self.r_side.render()
            self.r_top.update_scene(self.eval_env.data, camera="topdown")
            br = self.r_top.render()

            right = np.concatenate([tr, br], axis=0)
            frame = np.concatenate([left, right], axis=1)
            frames.append(frame)

            if term or trunc:
                obs, _ = self.eval_env.reset()

        filename = os.path.join(
            self.video_dir,
            f"step_{self.num_timesteps:08d}_dist_{distance:.2f}m.mp4",
        )
        try:
            imageio.mimsave(filename, frames, fps=50)
        except ValueError:
            filename = filename.replace(".mp4", ".gif")
            gif_frames = frames[::3]
            imageio.mimsave(filename, gif_frames, fps=20, loop=0)

        if self.verbose > 0:
            print(f"\n[Video] Saved: {filename}")
            print(f"  Step: {self.num_timesteps:,}, "
                  f"Reward: {episode_reward:.1f}, "
                  f"Distance: {distance:.3f}m\n")


class RewardComponentLogger(BaseCallback):
    """Logs each reward component's per-episode total, averaged per rollout."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_running_sum = None
        self._completed_totals = {}
        self._completed_episodes = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if self._ep_running_sum is None:
            self._ep_running_sum = [{} for _ in range(len(infos))]

        for i, info in enumerate(infos):
            comps = info.get("reward_components")
            if comps:
                for k, v in comps.items():
                    self._ep_running_sum[i][k] = (
                        self._ep_running_sum[i].get(k, 0.0) + float(v)
                    )

            if dones[i] and self._ep_running_sum[i]:
                for k, total in self._ep_running_sum[i].items():
                    self._completed_totals[k] = (
                        self._completed_totals.get(k, 0.0) + total
                    )
                self._completed_episodes += 1
                self._ep_running_sum[i] = {}
        return True

    def _on_rollout_end(self):
        if self._completed_episodes == 0:
            return
        for k, total in self._completed_totals.items():
            self.logger.record(f"reward/{k}",
                               total / self._completed_episodes)
        self.logger.record("reward/_episodes_in_rollout",
                           self._completed_episodes)
        self._completed_totals.clear()
        self._completed_episodes = 0


def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    if args.task not in TASKS:
        raise SystemExit(
            f"Unknown --task '{args.task}'. Choose from: {list(TASKS.keys())}"
        )
    task = TASKS[args.task]
    EnvCls = task["cls"]
    print(f"Task: {args.task} — {task['desc']}")

    print(f"Creating {args.n_envs} parallel environments...")
    env = make_vec_env(lambda: Monitor(EnvCls()), n_envs=args.n_envs)

    eval_env = EnvCls()

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, tensorboard_log=args.log_dir)
    else:
        print("Initializing PPO from scratch...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=args.log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        )

    callbacks = CallbackList([
        VideoRecorderCallback(
            eval_env=eval_env,
            video_dir=args.video_dir,
            video_freq=args.video_freq,
        ),
        CheckpointCallback(
            save_freq=max(50000 // args.n_envs, 1),
            save_path=args.save_dir,
            name_prefix="Eshy_ppo",
        ),
        RewardComponentLogger(),
    ])

    print(f"Training for {args.timesteps:,} timesteps")
    print(f"Videos: {args.video_dir}/ (every {args.video_freq:,} steps)")
    print(f"Tensorboard: {args.log_dir}/ "
          f"(run `tensorboard --logdir {args.log_dir}/`)")
    print(f"Models: {args.save_dir}/")
    print("-" * 60)

    model.learn(total_timesteps=args.timesteps,
                callback=callbacks, progress_bar=False)

    final_path = os.path.join(args.save_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining complete! Final model: {final_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=6)
    parser.add_argument("--video_freq", type=int, default=100_000)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--task", type=str, default="walk")
    args = parser.parse_args()
    if args.log_dir   is None: args.log_dir   = f"./logs/{args.task}/"
    if args.save_dir  is None: args.save_dir  = f"./models/{args.task}/"
    if args.video_dir is None: args.video_dir = f"./videos/{args.task}/"
    main(args)