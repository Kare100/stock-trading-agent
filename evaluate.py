"""
Evaluate the trained agent and plot portfolio performance.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from src.environment import StockTradingEnv
from src.agent import QLearningAgent


def evaluate(data_path="data/stock_data.csv", model_path="models/q_agent.pkl"):
    env = StockTradingEnv(csv_path=data_path)
    agent = QLearningAgent()
    agent.load(model_path)

    state = env.reset()
    done = False

    portfolio_values = [env.initial_balance]
    actions_taken = []
    action_labels = {0: "Hold", 1: "Buy", 2: "Sell"}

    while not done:
        action = agent.get_action(state)
        state, reward, done = env.step(action)
        portfolio_values.append(env.portfolio_value())
        actions_taken.append(action)

    final_value = portfolio_values[-1]
    profit = final_value - env.initial_balance
    profit_pct = (profit / env.initial_balance) * 100

    print(f"\nEvaluation Results")
    print(f"------------------")
    print(f"Starting balance : ${env.initial_balance:,.2f}")
    print(f"Final value      : ${final_value:,.2f}")
    print(f"Profit/Loss      : ${profit:+,.2f} ({profit_pct:+.2f}%)")
    print(f"\nAction breakdown:")
    for i, label in action_labels.items():
        count = actions_taken.count(i)
        print(f"  {label}: {count} times")

    plot_evaluation(env.df["Close"].tolist(), portfolio_values)


def plot_evaluation(prices, portfolio_values):
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    ax1.plot(prices, color="steelblue", linewidth=1)
    ax1.set_title("Stock Price Over Time")
    ax1.set_ylabel("Price ($)")

    ax2.plot(portfolio_values, color="seagreen", linewidth=1)
    ax2.axhline(y=10000, color="red", linestyle="--", label="Starting balance")
    ax2.set_title("Portfolio Value Over Time")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/evaluation_results.png")
    print("\nEvaluation plot saved to results/evaluation_results.png")


if __name__ == "__main__":
    evaluate()
