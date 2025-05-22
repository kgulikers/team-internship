
import os
import wandb
import numpy as np
import imageio
from stable_baselines3.common.callbacks import BaseCallback

class WandbVideoCallback(BaseCallback):
    def __init__(self, env, video_freq=5000, video_length=100, name="agent", verbose=0):
        super().__init__(verbose)
        self.env = env
        self.video_freq = video_freq
        self.video_length = video_length
        self.name = name
        self.frames = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.video_freq < self.video_length:
            frame = self.env.render(mode="rgb_array") if hasattr(self.env, 'render') else None
            if frame is not None:
                self.frames.append(frame)

        if self.num_timesteps % self.video_freq == self.video_length - 1 and self.frames:
            video_path = f"/tmp/{self.name}_step{self.num_timesteps}.mp4"
            imageio.mimsave(video_path, self.frames, fps=20)
            wandb.log({f"{self.name}_video": wandb.Video(video_path, fps=20, format="mp4")})
            self.frames = []

        return True

    def _on_training_end(self) -> None:
        if self.frames:
            video_path = f"/tmp/{self.name}_final.mp4"
            imageio.mimsave(video_path, self.frames, fps=20)
            wandb.log({f"{self.name}_video_final": wandb.Video(video_path, fps=20, format="mp4")})
            self.frames = []
