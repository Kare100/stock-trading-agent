"""
Generate sample stock price data for training.
Run this once to create data/stock_data.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_stock_data(n_days=1000, start_price=100.0):
    dates = pd.date_range(start="2021-01-01", periods=n_days, freq="B")
    
    returns = np.random.normal(loc=0.0003, scale=0.015, size=n_days)
    prices = [start_price]
    for r in returns[1:]:
        prices.append(round(prices[-1] * (1 + r), 2))

    volume = np.random.randint(1_000_000, 10_000_000, size=n_days)

    df = pd.DataFrame({
        "Date": dates,
        "Close": prices,
        "Volume": volume,
    })

    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Returns"] = df["Close"].pct_change()
    df = df.dropna().reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/stock_data.csv", index=False)
    print(f"Stock data saved to data/stock_data.csv ({len(df)} rows)")
    return df


if __name__ == "__main__":
    generate_stock_data()
