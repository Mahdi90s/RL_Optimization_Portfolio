import pandas as pd
import numpy as np
import gym
from gym import spaces

# --- START: Mult_Asset_portEnv CLASS DEFINITION (PASTE YOUR LATEST VERSION HERE) ---
class Mult_Asset_portEnv(gym.Env):
    def __init__(self, df, num_assets, features_list, window_size=5, initial_balance=1000, transaction_cost_rate=0.001):
        super(Mult_Asset_portEnv, self).__init__()
        
        self.df = df.copy()
        self.num_assets = num_assets
        self.features_list = features_list
        self.num_features_per_asset = len(features_list)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
        
        self.action_space = spaces.Box(low=0.0, high=1.0,
                                       shape=(self.num_assets + 1,), # <--- ADD A COMMA HERE
                                       dtype=np.float32)
        
        num_portfolio_features = self.num_assets + 1
        observation_row_size = self.num_assets * self.num_features_per_asset + num_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, observation_row_size), 
            dtype=np.float32
        )
        
        if len(self.df) < self.window_size + 1:
            raise ValueError(f"DataFrame too short for window_size {self.window_size}. "
                             f"Requires at least {self.window_size + 1} rows, but has {len(self.df)}.")
        
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

        all_feature_columns = [
            f"{ticker}_{feature}"
            for ticker in sorted(self.df.columns.str.split('_').str[0].unique())
            for feature in self.features_list
        ]
        
        missing_cols = [col for col in all_feature_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected feature columns in df_wide: {missing_cols}. "
                             f"Please check features_list and df_wide structure.")

        features_window_data = self.df.loc[self.df.index[start_idx:end_idx], all_feature_columns].values

        current_portfolio_state_repeated = np.tile(self.portfolio_weights, (self.window_size, 1))

        observation = np.concatenate([features_window_data, current_portfolio_state_repeated], axis=1)
        
        return observation.astype(np.float32)

    def step(self, action):
            action = action / np.sum(action) 
            self.portfolio_weights = action

            current_data_row = self.df.iloc[self.current_step]
            
            asset_tickers_in_order = sorted(self.df.columns.str.split('_').str[0].unique())
            
            current_close_prices = np.array([current_data_row[f"{ticker}_Close"] for ticker in asset_tickers_in_order], dtype=np.float32)
            current_log_returns = np.array([current_data_row[f"{ticker}_LogReturn"] for ticker in asset_tickers_in_order], dtype=np.float32)

            current_close_prices = np.maximum(current_close_prices, 1e-6)

            current_assets_market_value = np.sum(self.assets_shares * current_close_prices)
            total_portfolio_value_before_trades = self.cash + current_assets_market_value
            
            # --- DEBUG: Check for NaNs/Infs before crucial calculation ---
            # print(f"DEBUG: Step {self.current_step}, cash={self.cash:.2f}, assets_value={current_assets_market_value:.2f}, total_value_before_trades={total_portfolio_value_before_trades:.2f}")
            # if np.isnan(total_portfolio_value_before_trades) or np.isinf(total_portfolio_value_before_trades):
            #     print("DEBUG: total_portfolio_value_before_trades is NaN/Inf!")
            #     reward = -100 # Heavy penalty to indicate failure
            #     done = True
            #     return self._get_observation(), reward, done, {}


            if total_portfolio_value_before_trades <= 0:
                reward = -100
                done = True
                return self._get_observation(), reward, done, {}

            target_assets_value = action[:-1] * total_portfolio_value_before_trades
            target_cash_value = action[-1] * total_portfolio_value_before_trades

            target_assets_shares = target_assets_value / current_close_prices
            
            shares_to_buy_sell = target_assets_shares - self.assets_shares

            cash_flow_from_trades = np.sum(shares_to_buy_sell * current_close_prices)
            transaction_costs = np.sum(np.abs(shares_to_buy_sell * current_close_prices)) * self.transaction_cost_rate

            new_cash = self.cash - cash_flow_from_trades - transaction_costs
            new_assets_shares = self.assets_shares + shares_to_buy_sell

            if new_cash < 0:
                reward = -100
                done = True
                return self._get_observation(), reward, done, {}
            
            self.cash = new_cash
            self.assets_shares = new_assets_shares

            new_assets_market_value = np.sum(self.assets_shares * current_close_prices)
            self.balance = self.cash + new_assets_market_value

            # --- FIX 1: Add a small epsilon to prevent division by zero/near-zero ---
            # This prevents `portfolio_daily_return` from becoming NaN or Inf if total_portfolio_value_before_trades is 0 or extremely small.
            epsilon = 1e-9 
            portfolio_daily_return = (self.balance - total_portfolio_value_before_trades) / (total_portfolio_value_before_trades + epsilon)
            
            # --- FIX 2: Clip reward to prevent extreme values ---
            # Sometimes, returns can be extremely large or small, which can destabilize training.
            # Clip the reward to a reasonable range (e.g., -1.0 to 1.0 or -0.1 to 0.1 depending on expected daily returns)
            max_return_clip = 0.1 # Max daily return to consider for reward (e.g., 10%)
            min_return_clip = -0.1 # Min daily return to consider for reward
            portfolio_daily_return = np.clip(portfolio_daily_return, min_return_clip, max_return_clip)


            risk_penalty = 0.0001 
            reward = portfolio_daily_return - risk_penalty
            
            # --- DEBUG: Check final reward and observation before returning ---
            # if np.isnan(reward) or np.isinf(reward):
            #     print(f"DEBUG: Reward is NaN/Inf at step {self.current_step}! Value: {reward}")
            # obs_to_return = self._get_observation()
            # if np.isnan(obs_to_return).any() or np.isinf(obs_to_return).any():
            #     print(f"DEBUG: Observation is NaN/Inf at step {self.current_step}!")
            #     print(obs_to_return)
            
            self.history.append(self.balance)
            self.peak_portfolio_value = max(self.peak_portfolio_value, self.balance) 

            self.current_step += 1
            done = self.current_step >= len(self.df) or self.balance <= 0 or self.balance < (self.initial_balance * 0.1) 
            
            info = {}
            if done:
                if self.balance <= 0: info['reason'] = 'bankrupt'
                elif self.balance < (self.initial_balance * 0.1): info['reason'] = 'low_balance'
                elif self.current_step >= len(self.df): info['reason'] = 'end_of_data'

            return self._get_observation(), reward, done, info
        
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step {self.current_step}, Total Portfolio Value: {self.balance:.2f}, "
                f"Cash: {self.cash:.2f}, "
                f"Weights: {[f'{w:.2f}' for w in self.portfolio_weights]}")