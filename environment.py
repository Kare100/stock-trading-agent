"""
Stock Trading Environment

Actions:
    0 - Hold
    1 - Buy
    2 - Sell

State: [normalized price, MA10, MA50, returns, holding position]
"""

import numpy as np
import pandas as pd


class StockTradingEnv:
    def __init__(self, csv_path="data/stock_data.csv", initial_balance=10_000.0):
        self.df = pd.read_csv(csv_path)
        self.initial_balance = initial_balance
        self.n_steps = len(self.df)
        self.action_space = 3  # hold, buy, sell
        self.state_size = 5
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.current_step]
        max_price = self.df["Close"].max()

        return np.array([
            row["Close"] / max_price,
            row["MA10"] / max_price,
            row["MA50"] / max_price,
            row["Returns"],
            float(self.shares_held > 0),  # are we holding?
        ], dtype=np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]
        self.prev_net_worth = self.net_worth

        if action == 1:  # buy
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price

        elif action == 2:  # sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - self.prev_net_worth

        return self._get_state(), reward, done

    def portfolio_value(self):
        current_price = self.df.iloc[self.current_step]["Close"]
        return self.balance + self.shares_held * current_price
