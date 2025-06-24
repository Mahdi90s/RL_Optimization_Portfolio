import gym 
from gym import spaces 
import numpy as np
import pandas as pd

class Mult_Asset_portEnv(gym.Env):
    def __init__(self, df,num_assets,features_list, window_size=5, initial_balance=1000, transaction_cost_rate = 0.001):
        super(Mult_Asset_portEnv, self).__init__()
        
        self.df = df.copy()
        self.num_assets = num_assets
        self.features_list = features_list
        self.num_features_per_asset = len(features_list)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
        
        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(self.num_assets + 1), 
                                       dtype=np.float32)
        
        num_portfolio_features = self.num_assets + 1
        
        observation_row_size = self.num_assets * self.num_features_per_asset + num_portfolio_features
        
        
        self.observation_space = spaces.Box(
            low = -np.inf, high=np.inf, shape=(window_size, observation_row_size), 
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.cash = self.initial_balance
        self.portfolio_weights = np.array([0.0] * self.num_assets + [1.0], dtype=np.float32)
        self.assets_shares = np.array([0.0] * self.num_assets, dtype=np.float32)
        self.history = []
        self.peak_portfolio_value = self.initial_balance

        return self._get_observation()

    def _get_observation(self):
        
            start_idx = self.current_step - self.window_size
            end_idx = self.current_step
            
            feature_cols = [f"{ticker}_{feature}" 
                            for ticker in self.df['Ticker'].unique() 
                            for feature in self.features_list]
            
            current_features_flat = np.random.rand(self.num_assets * self.num_features_per_asset) # Replace with actual features
            current_portfolio_state_flat = self.portfolio_weights # num_assets + 1
            obs_features = self.df.loc[self.df.index[start_idx:end_idx], (slice(None), self.df['Ticker'].unique())]
            obs_shape_flat = self.num_assets * self.num_features_per_asset + (self.num_assets + 1)
            observation_data_per_day = np.random.rand(self.num_assets * self.num_features_per_asset + (self.num_assets + 1))
            obs = np.array([observation_data_per_day for _ in range(self.window_size)]) # Stack for window
            return obs.astype(np.float32) # This is a placeholder for your actual data extraction


    def step(self, action):
        
        action = action / np.sum(action) 
        current_day_returns = self.df.loc[self.df.index[self.current_step], 'LogReturn'] # Example multi-index access
        
        capital_to_reallocate = (action[:-1] - self.portfolio_weights[:-1]) * self.balance
        transaction_costs = np.sum(np.abs(capital_to_reallocate)) * self.transaction_cost_rate

        self.portfolio_weights = action # The agent's action becomes the new target weights

        portfolio_daily_return = np.sum(action[:-1] * current_day_returns) # Sum of (weight * return) for assets
        portfolio_daily_return = np.sum(self.portfolio_weights[:-1] * current_day_returns.values) # Assuming `current_day_returns` is a Series
        self.balance = self.balance * (1 + portfolio_daily_return) - transaction_costs
        
        risk_penalty = 0.0001 # A small constant penalty per step
        reward = portfolio_daily_return - risk_penalty # Placeholder for actual risk-adjusted return
        
        self.history.append(self.balance)
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.balance) # Update peak for drawdown

        self.current_step += 1
        done = self.current_step >= len(self.df) or self.balance <= 0 # Episode ends if out of data or bankrupt

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step {self.current_step}, Total Portfolio Value: {self.balance:.2f}, "
                f"Weights: {[f'{w:.2f}' for w in self.portfolio_weights]}")
        # Add more sophisticated rendering for analysis/visuals later