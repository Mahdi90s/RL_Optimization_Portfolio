import pandas as pd 
import numpy as np

import gym
from gym import spaces

class Mult_asset_env(gym.Env):
    
    def __init__(
        self,
        df, 
        num_assets, 
        features_list, 
        window_size=5, 
        initial_balance=1000, 
        transaction_cost_rate=0.001
        ):
        
        super(Mult_asset_env, self).__init__()
        
        if 'Date' not in df.columns:
            df = df.reset_index().rename(columns = {df.index.name or 'index': 'Date'})
        
        self.df = df.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        self.tickers = sorted(self.df['Ticker'].unique())
        self.num_assets = num_assets
        self.features_list = features_list
        self.num_features_per_asset = len(features_list)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
              
        self.action_space = spaces.Box(low = 0.0,
                                       high = 1.0,
                                       shape= (self.num_assets + 1,),
                                       dtype=np.float32)
        
        num_portfolio_features = self.num_assets + 1
        observation_row_size = self.num_assets * self.num_features_per_asset + num_portfolio_features
        
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(window_size, 
                                            observation_row_size),
                                            dtype=np.float32)
        
        self.dates = self.df['Date'].unique()
        self.current_step = self.window_size
        
        self.reset()
    
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.cash = self.initial_balance
        self.portfolio_weights = np.array([0.0] * self.num_assets + [1.0],dtype=np.float32)
        self.assets_shares = np.array([0.0] * self.num_assets, dtype=np.float32)
        self.history = []
        self.peak_portfolio_value = self.initial_balance
        return self.get_obs()
        
    def get_obs(self):
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
                
        start_date = self.dates[start_idx]
        end_date = self.dates[end_idx]
        
        obs_df = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] < end_date)]
        
        expected_rows = self.window_size * self.num_assets
        obs_values = obs_df[self.features_list].values.astype(np.float32)
        
        if obs_values.shape[0] < expected_rows:
            pad_rows = expected_rows - obs_values.shape[0]
            obs_values = np.vstack([obs_values, np.zeros((pad_rows, len(self.features_list)), dtype=np.float32)])
                
        feature_window_data = obs_values.reshape(
            self.window_size, self.num_assets * self.num_features_per_asset
        )
        
        current_portfolio_state_repeated = np.tile(self.portfolio_weights, (self.window_size, 1))
        obs = np.concatenate([feature_window_data, current_portfolio_state_repeated], axis=1)
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return obs.astype(np.float32)
    
    def step(self, action):
        
        if self.current_step >= len(self.dates):
            return self.get_obs(), 0.0, True, {'reason': 'end_of_data'}
        
        sum_action = np.sum(action)
        if sum_action == 0:
            action = np.ones_like(action) / (self.num_assets + 1)
        else:
            action = action / (sum_action + 1e-9)
        
        self.portfolio_weights = action

        current_date = self.dates[self.current_step]
        current_data_row = self.df[self.df['Date'] == current_date]
        current_close_prices = current_data_row['Close'].values.astype(np.float32) 
        
        current_close_prices = np.maximum(current_close_prices, 1e-6)
        
        current_assets_market_value = np.sum(self.assets_shares * current_close_prices)
        total_portfolio_value_before_trades = self.cash + current_assets_market_value
        
        if total_portfolio_value_before_trades <= 0:
            obs = self.get_obs()
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
            return obs, -100, True, {'reason': 'bankrupt'}
        
        target_assets_value = action[:-1] * total_portfolio_value_before_trades
        target_assets_shares = target_assets_value / current_close_prices
        
        shares_to_buy_sell = target_assets_shares - self.assets_shares
        transaction_costs = np.sum (np.abs(shares_to_buy_sell * current_close_prices)) * self.transaction_cost_rate
        
        cash_flow_from_trades = np.sum(shares_to_buy_sell * current_close_prices)
        new_cash = self.cash - cash_flow_from_trades - transaction_costs
        
        
        if new_cash <= 0:
            return self.get_obs(), -100, True, {'reason': 'low_cash'}
        
        
        self.cash = new_cash
        self.assets_shares += shares_to_buy_sell 
        
        new_assets_market_value = np.sum(self.assets_shares * current_close_prices)
        self.balance = self.cash + new_assets_market_value
        
        epsilon_balance = 1e-9
        portfolio_daily_return = (self.balance - total_portfolio_value_before_trades) / (total_portfolio_value_before_trades + epsilon_balance)
        
        max_return_clip = 0.1
        min_return_clip = -0.1
        portfolio_daily_return = np.clip(portfolio_daily_return, min_return_clip, max_return_clip)
        reward = portfolio_daily_return
        
        self.history.append(self.balance)
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.balance)
        
        
        done = (self.current_step + 1) >= len(self.dates) or self.balance < (self.initial_balance * 0.1)
        if not done:
            self.current_step += 1
            
        info = {}
        if done:
            if self.balance <= 0: info['reason'] = 'bankrupt'
            elif self.balance < (self.initial_balance * 0.1): info['reason'] = 'low_balance'
            elif self.current_step >= len(self.dates): info['reason'] = 'end_of_data'
        
        obs = self.get_obs()
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)    
           
        return obs, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f'Step {self.current_step}, Total portfolio Value: {self.balance:.2f}, cash: {self.cash:.2f}, Weights: {self.portfolio_weights}')
        
    