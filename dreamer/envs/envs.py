# import gym
import gymnasium as gym
import cv2
import dmc2gym

from dreamer.envs.wrappers import *


def make_dmc_env(
    domain_name,
    task_name,
    seed,
    visualize_reward,
    from_pixels,
    height,
    width,
    frame_skip,
    pixel_norm=True,
):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=visualize_reward,
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
    )
    if pixel_norm:
        env = PixelNormalization(env)
    return env


def make_atari_env(task_name, skip_frame, width, height, seed, pixel_norm=True):
    env = gym.make(task_name)
    env = gym.wrappers.ResizeObservation(env, (height, width))
    env = ChannelFirstEnv(env)
    env = SkipFrame(env, skip_frame)
    if pixel_norm:
        env = PixelNormalization(env)
    env.seed(seed)
    return env

def make_gymnasium_env(task_name, width, height):
    env = gymnasium_env(task_name, width, height)
    env = NormalizeActions(env)
    env = TimeLimit(env, 500)
    return env

class gymnasium_env:
    def __init__(self, name, width, height):
        self._env = gym.make(name, render_mode='rgb_array')
        self._action_repeat = 1
        self.width = width
        self.height = height
        self.reset()
  
    @property
    def observation_space(self):
        # return self._env.observation_space
        return gym.spaces.Box(low=0, high=255, shape=(3, self.height, self.width), dtype=np.uint8)
  
    @property
    def action_space(self):
        return self._env.action_space
  
    def reset(self):
        self._env.reset()
        obs = self.render()

        return (obs / 255 - 0.5).astype(np.float32)
  
    def step(self, action):
        reward = 0
        for _ in range(self._action_repeat):
            _, rew, term, trunc, info = self._env.step(action)
            done = term or trunc
            reward += rew
            obs = self.render()
            obs = (obs / 255 - 0.5).astype(np.float32)
            if done:
                break
        return obs, reward, done, info
  
    def render(self):
        img = self._env.render()
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return img.transpose(2, 0, 1)

class NormalizeActions:
    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)

class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if 'discount' not in info:
                info['discount'] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()
  
def get_env_infos(env):
    obs_shape = env.observation_space.shape
    if isinstance(env.action_space, gym.spaces.Discrete):
        discrete_action_bool = True
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        discrete_action_bool = False
        action_size = env.action_space.shape[0]
    else:
        raise Exception
    return obs_shape, discrete_action_bool, action_size