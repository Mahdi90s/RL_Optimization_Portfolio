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
        transaction_cost_rate=0.001,
        risk_aversion=0.10,
        cash_penalty=0.02,
        invest_bouns=0.001,
        vol_window=20,
        clip_return=0.10
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
        
        self.risk_aversion = float(risk_aversion)
        self.cash_penalty_coef = float (cash_penalty)
        self.invest_bouns_coef = float (invest_bouns)
        self.vol_window = int(vol_window)
        self.clip_return = float(clip_return)
        
        
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
        
        self._daily_idx = {}
        for d in self.dates:
            block = self.df[self.df['Date'] == d].sort_values('Ticker')
            if len(block) != self.num_assets:
                block = block.set_index('Ticker').reindex(self.tickers).reset_index()
            self._daily_idx[d] = block
        
        self.reset()
    
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.cash = self.initial_balance
        self.portfolio_weights = np.array([0.0] * self.num_assets + [1.0],dtype=np.float32)
        self.assets_shares = np.array([0.0] * self.num_assets, dtype=np.float32)
        self.history = []
        self.return_history = []
        self.peak_portfolio_value = self.initial_balance
        return self.get_obs()
    
    def _window_df(self, start_idx, end_idx):
        start_date = self.dates[start_idx]
        end_date = self.dates[end_idx]
        obs_df = self.df[(self.df['Date'] >= start_date) & (self.df['Date'] < end_date)]
        
        return obs_df
            
    def get_obs(self):
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step      
        
        obs_df = self._window_df(start_idx, end_idx)
        
        expected_rows = self.window_size * self.num_assets
        obs_values = obs_df[self.features_list].values.astype(np.float32)
        
        if obs_values.shape[0] < expected_rows:
            pad_rows = expected_rows - obs_values.shape[0]
            obs_values = np.vstack([obs_values, np.zeros((pad_rows, len(self.features_list)), dtype=np.float32)])

        mean = np.mean(obs_values, axis=0, keepdims=True)
        std = np.std(obs_values, axis=0, keepdims=True) + 1e-8
        obs_values = (obs_values - mean) / std

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
        
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        s = float(np.sum(action))
        if s <= 1e-12:
            action = np.ones_like(action, dtype=np.float32) / (self.num_assets + 1)
        else:
            action = action / s
        
        self.portfolio_weights = action

        current_date = self.dates[self.current_step]
        block = self._daily_idx[current_date]
        current_close_prices = block['Close'].values.astype(np.float32) 
        current_close_prices = np.nan_to_num(current_close_prices, nan=0.0)
        current_close_prices = np.maximum(current_close_prices, 1e-6)
        
        current_assets_mv = np.sum(self.assets_shares * current_close_prices)
        pv_before = float(self.cash + current_assets_mv)
        
        if pv_before <= 0:
            obs = self.get_obs()
            return obs, -100, True, {'reason': 'bankrupt'}
        
        target_assets_value = action[:-1] * pv_before
        target_assets_shares = target_assets_value / current_close_prices
        
        shares_to_buy_sell = target_assets_shares - self.assets_shares
        trade_cash = float(np.sum(shares_to_buy_sell * current_close_prices))
        transaction_costs = np.sum (np.abs(shares_to_buy_sell * current_close_prices)) * self.transaction_cost_rate
        new_cash = self.cash - trade_cash - transaction_costs
        
        
        # if new_cash <= -1e-6:
        #     obs = self.get_obs()
        #     return obs, -1.0, Fales, {'reason': 'infeasible_trade'}
        
        
        # self.cash = new_cash
        # self.assets_shares += shares_to_buy_sell 
        
        # new_assets_market_value = np.sum(self.assets_shares * current_close_prices)
        # self.balance = self.cash + new_assets_market_value
        
        # epsilon_balance = 1e-9
        # portfolio_daily_return = (self.balance - total_portfolio_value_before_trades) / (total_portfolio_value_before_trades + epsilon_balance)
        
        # max_return_clip = 0.1
        # min_return_clip = -0.1
        # portfolio_daily_return = np.clip(portfolio_daily_return, min_return_clip, max_return_clip)
        # self.return_history.append(portfolio_daily_return)
        
        # if len(self.return_history) < 2:
        #     reward = portfolio_daily_return + 0.01 * (self.balance - self.initial_balance)
        # else:
        #     window = min(len(self.return_history), 30)
        #     mean_return = np.mean(self.return_history[-window:])
        #     std_return = np.std(self.return_history[-window:]) + 1e-9
            
        #     if std_return < 1e-6:
        #         reward = mean_return
        #     else:
        #         reward = mean_return / std_return
            
        #     reward += portfolio_daily_return * 0.1

        # cash_penalty = -0.02 * (self.cash / self.balance)
        # reward += cash_penalty
        if new_cash < -1e-6:
            obs = self.get_obs()
            return obs, -0.1, False, {'reason': 'infeasible_trade'}
        
        self.last_transaction_costs = transaction_costs
        self.cash = max(0.0, new_cash)
        self.assets_shares = self.assets_shares + shares_to_buy_sell
        
        new_assets_mv = float(np.sum(self.assets_shares * current_close_prices))
        self.balance = float(self.cash + new_assets_mv)
        
        daily_ret = (self.balance - pv_before) / (pv_before + 1e-9)
        daily_ret = float(np.clip(daily_ret, -self.clip_return, self.clip_return))
        self.return_history.append(daily_ret)
        
        w = min(len(self.return_history), self.vol_window)
        vol = float(np.std(self.return_history[-w:])) if w >= 2 else 0.0
        
        w_cash = float(self.portfolio_weights[-1])
        invested_weight = 1.0 - w_cash
        reward = (
            daily_ret
            - self.risk_aversion * vol
            - self.cash_penalty_coef * w_cash
            + self.invest_bouns_coef * invested_weight
        )

        
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
        # obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)    
           
        return obs, float(reward), bool(done), info

    def render(self, mode='human'):
        if mode == 'human':
            holdings_str = ', '.join([
                f'{ticker}: {shares:.2} shares'
                for ticker, shares in zip(self.tickers, self.assets_shares)
            ])
            print(
                f'Step {self.current_step} | '
                f'PV: {self.balance:.2f} | '
                f'Cash: {self.cash:.2f} | '
                f'TxCost: {self.last_transaction_costs:.4f} | '
                f'Weights(asset...,cash): {np.round(self.portfolio_weights, 3).tolist()} | '
                f'Holdings: {holdings_str}'
            )
    