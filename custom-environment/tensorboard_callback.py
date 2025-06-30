from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = 0
        self.current_length = 0

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]

        self.current_rewards += rewards[0]
        self.current_length += 1

        if dones[0]:
            self.episode_rewards.append(self.current_rewards)
            self.episode_lengths.append(self.current_length)

            # Log to TensorBoard
            self.logger.record("custom/ep_rew_mean", self.current_rewards)
            self.logger.record("custom/ep_len_mean", self.current_length)

            self.current_rewards = 0
            self.current_length = 0

        return True
