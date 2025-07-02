import gym 
from gym import spaces 
import numpy as np
import pandas as pd

class Mult_Asset_portEnv_DQN(gym.Env):
    def __init__(self, df, num_assets, features_list, window_size=5, initial_balance=1000, transaction_cost_rate=0.001): # num_discrete_levels removed
        super(Mult_Asset_portEnv_DQN, self).__init__()
        
        self.df = df.copy()
        self.num_assets = num_assets
        self.features_list = features_list
        self.num_features_per_asset = len(features_list)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
        
        # --- MODIFICATION FOR DISCRETE ACTION SPACE (Simplified to Discrete) ---
        # Define a set of predefined portfolio allocation strategies (weights for assets + cash)
        # Assuming num_assets is 2 (AAPL, MSFT), so 3 dimensions for (AAPL, MSFT, Cash)
        if self.num_assets != 2:
            raise ValueError("This Discrete action space is set up for 2 assets only. Please adjust 'action_map'.")
        
        self.action_map = [
            np.array([0.0, 0.0, 1.0], dtype=np.float32),  # Action 0: 100% Cash
            np.array([0.5, 0.5, 0.0], dtype=np.float32),  # Action 1: 50% Asset1, 50% Asset2, 0% Cash (Equal-weight stocks)
            np.array([0.8, 0.2, 0.0], dtype=np.float32),  # Action 2: 80% Asset1, 20% Asset2, 0% Cash
            np.array([0.2, 0.8, 0.0], dtype=np.float32),  # Action 3: 20% Asset1, 80% Asset2, 0% Cash
            np.array([0.25, 0.25, 0.5], dtype=np.float32),# Action 4: 25% Asset1, 25% Asset2, 50% Cash
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Action 5: 100% Asset1, 0% Asset2, 0% Cash
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Action 6: 0% Asset1, 100% Asset2, 0% Cash
        ]
        self.num_actions = len(self.action_map)
        self.action_space = spaces.Discrete(self.num_actions) # Agent chooses an integer from 0 to num_actions-1
        # --- END MODIFICATION ---
        
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
        self.portfolio_weights = np.array([0.0] * self.num_assets + [1.0], dtype=np.float32) # Start 100% cash
        self.assets_shares = np.array([0.0] * self.num_assets, dtype=np.float32) # Shares owned per asset
        
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

        features_window_data = self.df.loc[self.df.index[start_idx:end_idx], all_feature_columns].values.astype(np.float32)

        current_portfolio_state_repeated = np.tile(self.portfolio_weights, (self.window_size, 1))

        observation = np.concatenate([features_window_data, current_portfolio_state_repeated], axis=1)
        
        # DEBUG LINE FOR OBSERVATION (uncomment if needed)
        # if np.isnan(observation).any() or np.isinf(observation).any():
        #     print(f"DEBUG: NaN/Inf detected in observation at step {self.current_step}!")
        #     # print("Observation data:\n", observation) 
        #     # raise ValueError("Observation contains NaN/Inf values!") 

        return observation.astype(np.float32)

    def step(self, action):
        # --- MODIFICATION FOR DISCRETE ACTION SPACE: Interpret action ---
        # Action is now a single integer chosen by the agent (e.g., 0, 1, 2, 3, 4, 5, 6)
        # Map this integer to the predefined portfolio weights from action_map
        if not (0 <= action < self.num_actions):
            raise ValueError(f"Action {action} is out of bounds for action_space with {self.num_actions} actions.")
            
        normalized_action = self.action_map[action]
        # (No need for np.sum(action) / (sum_action + 1e-9) here as weights are predefined and normalized)
        
        self.portfolio_weights = normalized_action # Update internal state with target weights
        # --- END MODIFICATION ---

        current_data_row = self.df.iloc[self.current_step]
        
        asset_tickers_in_order = sorted(self.df.columns.str.split('_').str[0].unique())
        
        current_close_prices = np.array([current_data_row[f"{ticker}_Close"] for ticker in asset_tickers_in_order], dtype=np.float32)
        current_log_returns = np.array([current_data_row[f"{ticker}_LogReturn"] for ticker in asset_tickers_in_order], dtype=np.float32)

        current_close_prices = np.maximum(current_close_prices, 1e-6)

        current_assets_market_value = np.sum(self.assets_shares * current_close_prices)
        total_portfolio_value_before_trades = self.cash + current_assets_market_value
        
        # DEBUG: Check total_portfolio_value_before_trades (uncomment if needed)
        # if total_portfolio_value_before_trades <= 0:
        #    print(f"DEBUG: Total portfolio value before trades is <= 0 at step {self.current_step}!")
        # if np.isnan(total_portfolio_value_before_trades) or np.isinf(total_portfolio_value_before_trades):
        #    print(f"DEBUG: total_portfolio_value_before_trades is NaN/Inf at step {self.current_step}!")

        if total_portfolio_value_before_trades <= 0:
            reward = -100
            done = True
            return self._get_observation(), reward, done, {}

        target_assets_value = normalized_action[:-1] * total_portfolio_value_before_trades
        target_cash_value = normalized_action[-1] * total_portfolio_value_before_trades

        target_assets_shares = target_assets_value / current_close_prices
        
        shares_to_buy_sell = target_assets_shares - self.assets_shares

        cash_flow_from_trades = np.sum(shares_to_buy_sell * current_close_prices)
        transaction_costs = np.sum(np.abs(shares_to_buy_sell * current_close_prices)) * self.transaction_cost_rate

        new_cash = self.cash - cash_flow_from_trades - transaction_costs
        new_assets_shares = self.assets_shares + shares_to_buy_sell

        # DEBUG: Check cash after trades (uncomment if needed)
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

        epsilon_balance = 1e-9 
        portfolio_daily_return = (self.balance - total_portfolio_value_before_trades) / (total_portfolio_value_before_trades + epsilon_balance)
        
        # DEBUG: Check portfolio_daily_return (uncomment if needed)
        # if np.isnan(portfolio_daily_return) or np.isinf(portfolio_daily_return):
        #     print(f"DEBUG: portfolio_daily_return is NaN/Inf at step {self.current_step}!")

        max_return_clip = 0.1 
        min_return_clip = -0.1 
        portfolio_daily_return = np.clip(portfolio_daily_return, min_return_clip, max_return_clip)

        risk_penalty = 0.0001 
        reward = portfolio_daily_return - risk_penalty
        
        # DEBUG: Check final reward (uncomment if needed)
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