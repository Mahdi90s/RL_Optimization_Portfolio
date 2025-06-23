import gym 
from gym import spaces 
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(self, df, window_size=5, initial_balance=1000):
        super(PortfolioEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(
            low = -np.inf, high=np.inf, shape=(window_size,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.positions = 0  # 1=long, -1=short, 0=flat
        self.total_value = self.initial_balance
        self.history = []

        return self._get_observation()

    def _get_observation(self):
        obs = self.df['LogReturn_Z'].iloc[self.current_step - self.window_size:self.current_step].values
        return obs.astype(np.float32)

    def step(self, action):
        done = False
        log_return_today = self.df['LogReturn'].iloc[self.current_step]

        # ACTION LOGIC
        if action == 1:
            self.positions = 1
        elif action == 2:
            self.positions = -1
        else:
            self.positions = 0

        # REWARD = profit/loss based on today's return
        reward = self.positions * log_return_today
        self.total_value *= (1 + reward)

        self.history.append(self.total_value)

        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step {self.current_step}, Total Value: {self.total_value:.2f}")
    
        
    