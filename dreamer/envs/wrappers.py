import gymnasium as gym
import numpy as np


class ChannelFirstEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space
        obs_shape = obs_space.shape[-1:] + obs_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def _permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        return observation

    def observation(self, observation):
        observation = self._permute_orientation(observation)
        return observation


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class PixelNormalization(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _pixel_normalization(self, obs):
        return obs / 255.0 - 0.5

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pixel_normalization(obs), reward, terminated, truncated, info

    def reset(self):
        obs, info = self.env.reset()
        return self._pixel_normalization(obs), info

# gym.wrappers.AddRenderObservation also does the same thing
class GymPixelEnv(gym.Wrapper):
    def __init__(self, env):
        self._env = env
        obs,_ = self.reset()
        self.obs_shape = obs.shape

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        
    @property
    def action_space(self):
        return self._env.action_space
    
    def render(self):
        return self._env.render()

    def reset(self, seed=None, options=None):
        obs, info = self._env.reset()
        obs = self.render()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs = self.render()
        return obs, reward, terminated, truncated, info