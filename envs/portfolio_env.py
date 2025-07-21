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

# --- Inside Mult_Asset_portEnv class, _get_observation method ---
# --- Inside Mult_Asset_portEnv class, _get_observation method ---
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

        # Ensure features_window_data is explicitly float32 here
        features_window_data = self.df.loc[self.df.index[start_idx:end_idx], all_feature_columns].values.astype(np.float32)

        current_portfolio_state_repeated = np.tile(self.portfolio_weights, (self.window_size, 1))

        observation = np.concatenate([features_window_data, current_portfolio_state_repeated], axis=1)
        
        # --- DEBUG LINE FOR OBSERVATION DTYPE ---
        print(f"DEBUG: Observation dtype at step {self.current_step}: {observation.dtype}")
        # --- END DEBUG LINE ---

        # --- DEBUG LINE FOR OBSERVATION (already there, but uncomment if not) ---
        if np.isnan(observation).any() or np.isinf(observation).any():
            print(f"DEBUG: NaN/Inf detected in observation at step {self.current_step}!")
            # print("Observation data:\n", observation) 
            # raise ValueError("Observation contains NaN/Inf values!") 
        # --- END DEBUG LINE ---

        return observation.astype(np.float32) # Final explicit cast for safety

# --- Inside Mult_Asset_portEnv class, step method ---
# --- Inside Mult_Asset_portEnv class, step method ---
    def step(self, action):
        # --- FIX 1: Robust Action Normalization ---
        # Add a small epsilon to the sum to prevent division by zero
        sum_action = np.sum(action)
        if sum_action == 0: # If all actions are zero, divide by num_assets to prevent NaN
            action = np.ones_like(action) / (self.num_assets + 1) # Distribute equally, or handle as no action
            # print(f"DEBUG: Action sum was zero. Normalized to equal distribution. Action: {action}")
        else:
            action = action / (sum_action + 1e-9) # Add epsilon to denominator
        
        # --- DEBUG: Check action after normalization ---
        # print(f"DEBUG: Step {self.current_step}, Normalized Action: {action}")
        # if np.isnan(action).any() or np.isinf(action).any():
        #     print(f"DEBUG: Action contains NaN/Inf after normalization at step {self.current_step}!")
        #     raise ValueError("Action is NaN/Inf after normalization!") # Terminate for immediate debugging
        # --- END DEBUG ---

        self.portfolio_weights = action 

        # --- DEBUG: Check portfolio_weights after update ---
        # if np.isnan(self.portfolio_weights).any() or np.isinf(self.portfolio_weights).any():
        #     print(f"DEBUG: portfolio_weights contain NaN/Inf at step {self.current_step} after action update!")
        #     raise ValueError("portfolio_weights is NaN/Inf!") # Terminate for immediate debugging
        # --- END DEBUG ---

        current_data_row = self.df.iloc[self.current_step]
        
        asset_tickers_in_order = sorted(self.df.columns.str.split('_').str[0].unique())
        
        current_close_prices = np.array([current_data_row[f"{ticker}_Close"] for ticker in asset_tickers_in_order], dtype=np.float32)
        current_log_returns = np.array([current_data_row[f"{ticker}_LogReturn"] for ticker in asset_tickers_in_order], dtype=np.float32)

        current_close_prices = np.maximum(current_close_prices, 1e-6)

        current_assets_market_value = np.sum(self.assets_shares * current_close_prices)
        total_portfolio_value_before_trades = self.cash + current_assets_market_value
        
        # DEBUG: Check total_portfolio_value_before_trades (uncommented if needed)
        # print(f"DEBUG: Step {self.current_step}, Total val before trades: {total_portfolio_value_before_trades:.2f}")
        # if total_portfolio_value_before_trades <= 0:
        #    print(f"DEBUG: Total portfolio value before trades is <= 0 at step {self.current_step}!")
        # if np.isnan(total_portfolio_value_before_trades) or np.isinf(total_portfolio_value_before_trades):
        #    print(f"DEBUG: total_portfolio_value_before_trades is NaN/Inf at step {self.current_step}!")

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

        # DEBUG: Check cash after trades (uncommented if needed)
        # print(f"DEBUG: Step {self.current_step}, new_cash={new_cash:.2f}")
        # if new_cash < 0:
        #     print(f"DEBUG: New cash is negative at step {self.current_step}!")

        if new_cash < 0:
            reward = -100
            done = True
            return self._get_observation(), reward, done, {}
        
        self.cash = new_cash
        self.assets_shares = new_assets_shares

        new_assets_market_value = np.sum(self.assets_shares * current_close_prices)
        self.balance = self.cash + new_assets_market_value

        epsilon_balance = 1e-9 # Use a distinct epsilon for balance to avoid confusion
        portfolio_daily_return = (self.balance - total_portfolio_value_before_trades) / (total_portfolio_value_before_trades + epsilon_balance)
        
        # DEBUG: Check portfolio_daily_return (uncommented if needed)
        # print(f"DEBUG: Step {self.current_step}, Daily return: {portfolio_daily_return:.4f}")
        # if np.isnan(portfolio_daily_return) or np.isinf(portfolio_daily_return):
        #     print(f"DEBUG: portfolio_daily_return is NaN/Inf at step {self.current_step}!")

        max_return_clip = 0.1 
        min_return_clip = -0.1 
        portfolio_daily_return = np.clip(portfolio_daily_return, min_return_clip, max_return_clip)

        risk_penalty = 0.0001 
        reward = portfolio_daily_return - risk_penalty
        
        # DEBUG: Check final reward (uncommented if needed)
        # print(f"DEBUG: Step {self.current_step}, Final reward: {reward:.4f}")
        # if np.isnan(reward) or np.isinf(reward):
        #     print(f"DEBUG: Final reward is NaN/Inf at step {self.current_step}!")
        
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
