import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO # To load the models and use their policies

# Import the continuous control environment that A2C/PPO were trained on
from envs.portfolio_env import Mult_Asset_portEnv

class MetaPortfolioEnv(gym.Env):
    def __init__(self, df_with_perf_data, sub_agent_models, num_assets, features_list, window_size=5, initial_balance=1000, transaction_cost_rate=0.001, perf_window_size=20):
        super(MetaPortfolioEnv, self).__init__()

        self.df = df_with_perf_data.copy()
        self.sub_agent_models = sub_agent_models # Dictionary of {'A2C': model_a2c, 'PPO': model_ppo}
        self.num_assets = num_assets
        self.features_list = features_list # Features used by sub-agents for their observation
        self.window_size = window_size # Lookback for sub-agent observations
        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
        self.perf_window_size = perf_window_size # Lookback for sub-agent performance in meta-obs

        # --- Meta-Agent's Action Space ---
        # Action 0: Choose A2C
        # Action 1: Choose PPO
        self.num_sub_agents = len(self.sub_agent_models)
        self.action_space = spaces.Discrete(self.num_sub_agents)

        # --- Meta-Agent's Observation Space ---
        # This will be a flattened vector of:
        # 1. Overall Market Features (e.g., average LogReturn_Z across assets for current window_size)
        # 2. Overall Portfolio State (balance, cash, drawdown, etc.)
        # 3. Recent Performance of each sub-agent (e.g., average daily return over perf_window_size)

        # Calculate base feature size per day (num_assets * num_features_per_asset) from sub-env
        # And current portfolio state (num_assets + 1 for weights) from sub-env
        # This is what a sub-agent receives from Mult_Asset_portEnv
        sub_env_obs_row_size = self.num_assets * len(self.features_list) + (self.num_assets + 1)
        
        # Meta-observation features:
        # 1. Rolling average of LogReturn_Z for all assets (over current window_size, e.g., 5 * 2 = 10 features)
        # 2. Overall Portfolio state (balance, cash, current_weights) = 1 + 1 + (num_assets + 1) = 3 + num_assets features
        # 3. Rolling average daily return for EACH sub-agent (over perf_window_size, e.g., 20 * num_sub_agents = 40 features)
        # Let's simplify and use the last day's average return for the period, and a rolling Sharpe for each sub-agent.

        # Simplified meta-observation features:
        # - Mean LogReturn_Z across all assets for the current window (1 feature)
        # - Std Dev of LogReturn_Z across all assets for the current window (1 feature)
        # - Current portfolio balance (normalized) (1 feature)
        # - Current cash percentage (1 feature)
        # - Current overall portfolio weights (num_assets + 1 features)
        # - Average performance (e.g., mean daily return) of each sub-agent over perf_window_size (num_sub_agents features)
        # - Standard deviation of performance of each sub-agent over perf_window_size (num_sub_agents features)

        num_meta_features = 2 + 1 + 1 + (self.num_assets + 1) + (self.num_sub_agents * 2) # (mean, std) for market, (balance, cash, weights), (mean, std) for each sub-agent
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_meta_features,), dtype=np.float32 # Flattened meta-observation
        )

        # Ensure enough data for both sub-agent window and meta-agent performance window
        # The meta-agent needs current_step + perf_window_size data points to calculate historical performance
        # (initial window_size for sub-agent obs) + (perf_window_size for meta-agent's performance history)
        if len(self.df) < self.window_size + self.perf_window_size + 1:
            raise ValueError(f"DataFrame too short for meta-agent environment. Requires at least "
                             f"{self.window_size + self.perf_window_size + 1} rows, but has {len(self.df)}.")

        self.reset()

    def reset(self):
        # The meta-agent's 'current_step' starts after enough history for sub-agent obs AND meta-agent perf history
        self.current_step = self.window_size + self.perf_window_size
        self.balance = self.initial_balance
        self.cash = self.initial_balance
        self.portfolio_weights = np.array([0.0] * self.num_assets + [1.0], dtype=np.float32) # Start 100% cash
        self.assets_shares = np.array([0.0] * self.num_assets, dtype=np.float32)

        self.history = []
        self.peak_portfolio_value = self.initial_balance
        
        return self._get_observation()

    def _get_observation(self):
        # Ensure we have enough data for the observation window and performance window
        if self.current_step < self.window_size + self.perf_window_size:
            # This should ideally not happen if reset() sets current_step correctly
            # But as a fallback, return zeros or raise an error if this state is invalid
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # --- Extract Market Features for Meta-Observation ---
        # Get LogReturn_Z for all assets for the current window_size
        market_obs_start_idx = self.current_step - self.window_size
        market_obs_end_idx = self.current_step
        
        log_return_z_cols = [f"{ticker}_LogReturn_Z" for ticker in sorted(self.df.columns.str.split('_').str[0].unique())]
        
        market_window_data = self.df.loc[self.df.index[market_obs_start_idx:market_obs_end_idx], log_return_z_cols].values
        
        mean_log_return_z = np.mean(market_window_data)
        std_log_return_z = np.std(market_window_data)

        # --- Extract Portfolio State for Meta-Observation ---
        normalized_balance = self.balance / self.initial_balance
        cash_percentage = self.cash / self.balance if self.balance > 0 else 0.0

        # --- Extract Sub-Agent Performance Metrics for Meta-Observation ---
        perf_data_start_idx = self.current_step - self.perf_window_size
        perf_data_end_idx = self.current_step

        a2c_perf = self.df.loc[self.df.index[perf_data_start_idx:perf_data_end_idx], 'A2C_DailyReturn'].values
        ppo_perf = self.df.loc[self.df.index[perf_data_start_idx:perf_data_end_idx], 'PPO_DailyReturn'].values
        
        # Handle potential NaNs in performance data (e.g., from initial window_size steps)
        # Replace NaNs with 0 or a small number, or average over non-NaNs
        a2c_perf = np.nan_to_num(a2c_perf, nan=0.0)
        ppo_perf = np.nan_to_num(ppo_perf, nan=0.0)

        # Calculate mean and std dev of daily returns for each sub-agent over the performance window
        a2c_mean_perf = np.mean(a2c_perf)
        a2c_std_perf = np.std(a2c_perf)
        ppo_mean_perf = np.mean(ppo_perf)
        ppo_std_perf = np.std(ppo_perf)

        # --- Construct the final meta-observation vector ---
        meta_observation = np.concatenate([
            np.array([mean_log_return_z, std_log_return_z]), # Market features
            np.array([normalized_balance, cash_percentage]), # Portfolio state
            self.portfolio_weights,                          # Current detailed portfolio weights
            np.array([a2c_mean_perf, a2c_std_perf, ppo_mean_perf, ppo_std_perf]) # Sub-agent performance
        ])

        # Ensure the observation matches the defined space shape
        if meta_observation.shape != self.observation_space.shape:
            raise RuntimeError(f"Meta-observation shape mismatch: Expected {self.observation_space.shape}, got {meta_observation.shape}")

        return meta_observation.astype(np.float32)

    def step(self, action):
        done = False
        info = {}

        # 1. Meta-Agent chooses a sub-agent based on its action
        chosen_agent_name = None
        chosen_sub_model = None
        if action == 0:
            chosen_agent_name = 'A2C'
            chosen_sub_model = self.sub_agent_models['A2C']
            # Re-create environment for sub-agent prediction as its observation depends on it
            temp_env = Mult_Asset_portEnv(df=self.df.copy(), num_assets=self.num_assets, features_list=self.features_list, window_size=self.window_size, initial_balance=self.initial_balance, transaction_cost_rate=self.transaction_cost_rate)
            temp_env.current_step = self.current_step # Align temp_env's current step
            temp_obs = temp_env._get_observation() # Get observation for sub-agent

        elif action == 1:
            chosen_agent_name = 'PPO'
            chosen_sub_model = self.sub_agent_models['PPO']
            temp_env = Mult_Asset_portEnv(df=self.df.copy(), num_assets=self.num_assets, features_list=self.features_list, window_size=self.window_size, initial_balance=self.initial_balance, transaction_cost_rate=self.transaction_cost_rate)
            temp_env.current_step = self.current_step # Align temp_env's current step
            temp_obs = temp_env._get_observation() # Get observation for sub-agent
        else:
            raise ValueError(f"Invalid meta-agent action: {action}. Must be 0 (A2C) or 1 (PPO).")

        # 2. Get the action (portfolio weights) from the chosen sub-agent
        sub_agent_action, _states = chosen_sub_model.predict(temp_obs, deterministic=True)
        
        # 3. Apply the sub-agent's action to the main portfolio simulation
        # The following logic is adapted from Mult_Asset_portEnv's step method
        
        # Normalize action to ensure weights sum to 1.0 (sub-agent's action is continuous)
        sub_agent_action = sub_agent_action / np.sum(sub_agent_action) 
        self.portfolio_weights = sub_agent_action # Meta-env updates its portfolio_weights based on sub-agent's decision

        current_data_row = self.df.iloc[self.current_step]
        asset_tickers_in_order = sorted(self.df.columns.str.split('_').str[0].unique())
        
        current_close_prices = np.array([current_data_row[f"{ticker}_Close"] for ticker in asset_tickers_in_order], dtype=np.float32)
        current_log_returns = np.array([current_data_row[f"{ticker}_LogReturn"] for ticker in asset_tickers_in_order], dtype=np.float32)

        current_close_prices = np.maximum(current_close_prices, 1e-6)

        current_assets_market_value = np.sum(self.assets_shares * current_close_prices)
        total_portfolio_value_before_trades = self.cash + current_assets_market_value
        
        if total_portfolio_value_before_trades <= 0:
            reward = -100
            done = True
            info['reason'] = 'bankrupt'
            return self._get_observation(), reward, done, info

        target_assets_value = sub_agent_action[:-1] * total_portfolio_value_before_trades
        target_cash_value = sub_agent_action[-1] * total_portfolio_value_before_trades

        target_assets_shares = target_assets_value / current_close_prices
        shares_to_buy_sell = target_assets_shares - self.assets_shares

        cash_flow_from_trades = np.sum(shares_to_buy_sell * current_close_prices)
        transaction_costs = np.sum(np.abs(shares_to_buy_sell * current_close_prices)) * self.transaction_cost_rate

        new_cash = self.cash - cash_flow_from_trades - transaction_costs
        new_assets_shares = self.assets_shares + shares_to_buy_sell

        if new_cash < 0:
            reward = -100
            done = True
            info['reason'] = 'meta_cash_negative'
            return self._get_observation(), reward, done, info
        
        self.cash = new_cash
        self.assets_shares = new_assets_shares

        new_assets_market_value = np.sum(self.assets_shares * current_close_prices)
        self.balance = self.cash + new_assets_market_value

        epsilon_balance = 1e-9 
        portfolio_daily_return = (self.balance - total_portfolio_value_before_trades) / (total_portfolio_value_before_trades + epsilon_balance)
        
        max_return_clip = 0.1 
        min_return_clip = -0.1 
        portfolio_daily_return = np.clip(portfolio_daily_return, min_return_clip, max_return_clip)

        risk_penalty = 0.0001 
        reward = portfolio_daily_return - risk_penalty # Meta-agent gets this reward

        self.history.append(self.balance)
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.balance) 

        self.current_step += 1
        done = self.current_step >= len(self.df) or self.balance <= 0 or self.balance < (self.initial_balance * 0.1) 
        
        if done:
            if self.balance <= 0: info['reason'] = 'bankrupt'
            elif self.balance < (self.initial_balance * 0.1): info['reason'] = 'low_balance'
            elif self.current_step >= len(self.df): info['reason'] = 'end_of_data'

        return self._get_observation(), reward, done, info
        
    def render(self, mode='human'):
        if mode == 'human':
            # This render includes sub-agent name for clarity
            print(f"Step {self.current_step}, Total Port Value: {self.balance:.2f}, Cash: {self.cash:.2f}, "
                  f"Weights: {[f'{w:.2f}' for w in self.portfolio_weights]}, Chosen Agent: {self.chosen_agent_name if hasattr(self, 'chosen_agent_name') else 'N/A'}")