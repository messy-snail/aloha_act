# import gym
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np
import mujoco
import mujoco_viewer
from enum import Enum
from gymnasium.envs.registration import register
from gymnasium import spaces

class RenderMode(Enum):
    WINDOW = 'window'
    OFFSCREEN = 'offscreen'
    OFF = 'off'
    
class BaseEnv(gym.Env):
    def __init__(self, args, render_mode: RenderMode):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(args.env_name)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.np_random, _ = seeding.np_random(None)
        # Time
        self.current_time = 0.0  # Initialize current time
        self.time_step = 0.002  # mjcf와 동일하게 설정
        self.max_episode_time = args.time_step
            
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.model.nq + self.model.nv,), dtype=np.float32)
        self.initialize_view(self.render_mode)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def initialize_view(self, render_mode, azimuth =90.0, elevation = -15.0, distance = 7.0, lookat=np.array([0.0, 0.0, 1.0]) ):
        if(render_mode!=RenderMode.OFF):
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, width=1400, height=900, mode=render_mode.value)
            # self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, mode=render_mode.value)
            # self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, mode='offscreen', width=1400, height=900)
            # 카메라 설정
            self.viewer.cam.azimuth = azimuth
            self.viewer.cam.elevation = elevation
            self.viewer.cam.distance = distance
            self.viewer.cam.lookat = lookat
    
    def step(self, action):
        NotImplemented

    def reset(self):
        NotImplemented
    

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _get_reward(self, action):
        NotImplemented
    
    def _check_done(self):
        NotImplemented

    def render(self, mode: RenderMode = RenderMode.WINDOW):
        if self.viewer is None:
            self.initialize_view(mode)

        self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

