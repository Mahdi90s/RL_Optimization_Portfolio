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

        # 1. Dynamically select all relevant feature columns from df_wide
        # This creates a list of column names like ['AAPL_Close', 'AAPL_LogReturn', ..., 'MSFT_LogReturn_Z', ...]
        all_feature_columns = [
            f"{ticker}_{feature}"
            for ticker in sorted(self.df.columns.str.split('_').str[0].unique()) # Ensure consistent ticker order
            for feature in self.features_list
        ]
        
        # Ensure all required feature columns exist in df_wide
        missing_cols = [col for col in all_feature_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected feature columns in df_wide: {missing_cols}. "
                             f"Please check features_list and df_wide structure.")

        # 2. Extract the windowed data for these features
        # This will give a 2D NumPy array of shape (window_size, num_assets * num_features_per_asset)
        features_window_data = self.df.loc[self.df.index[start_idx:end_idx], all_feature_columns].values

        # 3. Include current portfolio state (weights) in the observation
        # The portfolio weights are the same for all timesteps in the observation window.
        # So, we repeat the current portfolio_weights for each day in the window.
        # This results in a 2D array of shape (window_size, num_assets + 1)
        current_portfolio_state_repeated = np.tile(self.portfolio_weights, (self.window_size, 1))

        # 4. Concatenate the features and portfolio state horizontally for each day in the window
        # The final observation shape will be (window_size, (num_assets * num_features_per_asset) + (num_assets + 1))
        observation = np.concatenate([features_window_data, current_portfolio_state_repeated], axis=1)
        
        return observation.astype(np.float32)

    def step(self, action):
            # 1. Normalize action to ensure weights sum to 1.0
            # The agent's action is target portfolio weights for assets and cash
            action = action / np.sum(action) 
            self.portfolio_weights = action # Update internal state with target weights

            # Get today's data for calculations
            current_data_row = self.df.iloc[self.current_step]
            
            # Dynamically get current close prices and log returns for all assets
            # Ensure order matches the expected asset order for weights
            asset_tickers_in_order = sorted(self.df.columns.str.split('_').str[0].unique())
            
            # Extract current day's close prices for all assets
            current_close_prices = np.array([current_data_row[f"{ticker}_Close"] for ticker in asset_tickers_in_order], dtype=np.float32)
            
            # Extract current day's log returns for all assets
            current_log_returns = np.array([current_data_row[f"{ticker}_LogReturn"] for ticker in asset_tickers_in_order], dtype=np.float32)

            # Handle potential zero or negative prices to avoid division by zero or invalid calculations
            # If any price is zero or negative, set it to a very small positive number or handle as an invalid state
            current_close_prices = np.maximum(current_close_prices, 1e-6) # Ensure prices are positive

            # 2. Calculate current market value of assets and total portfolio value BEFORE trades
            current_assets_market_value = np.sum(self.assets_shares * current_close_prices)
            total_portfolio_value_before_trades = self.cash + current_assets_market_value
            
            # Handle potential bankruptcy (can't trade if balance is too low)
            if total_portfolio_value_before_trades <= 0:
                reward = -100 # Heavy penalty for bankruptcy
                done = True
                return self._get_observation(), reward, done, {}

            # 3. Determine target asset values and target cash based on new weights
            # Agent's action[:-1] represents weights for assets, action[-1] for cash
            target_assets_value = action[:-1] * total_portfolio_value_before_trades
            target_cash_value = action[-1] * total_portfolio_value_before_trades

            # 4. Calculate target shares for each asset
            target_assets_shares = target_assets_value / current_close_prices
            
            # 5. Calculate shares to buy/sell for each asset
            shares_to_buy_sell = target_assets_shares - self.assets_shares

            # 6. Calculate cash flow from trades and transaction costs
            cash_flow_from_trades = np.sum(shares_to_buy_sell * current_close_prices)
            transaction_costs = np.sum(np.abs(shares_to_buy_sell * current_close_prices)) * self.transaction_cost_rate

            # 7. Attempt to update cash and asset shares
            new_cash = self.cash - cash_flow_from_trades - transaction_costs
            new_assets_shares = self.assets_shares + shares_to_buy_sell

            # 8. Constraint check: Ensure enough cash for purchases and prevent short selling if not allowed
            # For simplicity, we enforce no shorting (shares_to_buy_sell < 0 implies selling)
            # and ensure cash doesn't go negative after trades.
            # This is a critical area for more advanced environment design (e.g., slippage, partial fills)

            # If a trade would result in negative cash or shares (and shorting is not allowed), penalize heavily
            if new_cash < 0: # Agent tried to spend more cash than available
                # Option 1: Clip trades to cash available (more complex, but realistic)
                # Option 2: Penalize and revert (simpler for initial RL)
                # For now, let's heavily penalize if cash goes negative
                reward = -100 # Heavy penalty
                done = True
                # Revert state for consistency if episode ends immediately due to invalid trade
                # self.cash = old_cash_before_trades_attempt
                # self.assets_shares = old_assets_shares_before_trades_attempt
                # self.balance = old_balance_before_trades_attempt
                return self._get_observation(), reward, done, {}
            
            # Update actual holdings
            self.cash = new_cash
            self.assets_shares = new_assets_shares

            # 9. Recalculate total portfolio value after trades
            new_assets_market_value = np.sum(self.assets_shares * current_close_prices)
            self.balance = self.cash + new_assets_market_value

            # 10. Calculate realistic portfolio daily return (for reward)
            # This is the actual percentage change in portfolio value
            portfolio_daily_return = (self.balance - total_portfolio_value_before_trades) / total_portfolio_value_before_trades \
                                    if total_portfolio_value_before_trades != 0 else 0

            # 11. Apply risk penalty (placeholder for now)
            risk_penalty = 0.0001 
            
            # 12. Final reward calculation
            reward = portfolio_daily_return - risk_penalty
            
            self.history.append(self.balance)
            self.peak_portfolio_value = max(self.peak_portfolio_value, self.balance) 

            # 13. Update step and check done conditions
            self.current_step += 1
            done = self.current_step >= len(self.df) or self.balance <= 0 or self.balance < (self.initial_balance * 0.1) 
            
            # Optional: Add terminal info for debugging
            info = {}
            if done:
                if self.balance <= 0: info['reason'] = 'bankrupt'
                elif self.balance < (self.initial_balance * 0.1): info['reason'] = 'low_balance'
                elif self.current_step >= len(self.df): info['reason'] = 'end_of_data'

            return self._get_observation(), reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step {self.current_step}, Total Portfolio Value: {self.balance:.2f}, "
                f"Weights: {[f'{w:.2f}' for w in self.portfolio_weights]}")
        # Add more sophisticated rendering for analysis/visuals later