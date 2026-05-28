"""
Train the Q-learning stock trading agent.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from src.environment import StockTradingEnv
from src.agent import QLearningAgent


def train(episodes=100, data_path="data/stock_data.csv"):
    env = StockTradingEnv(csv_path=data_path)
    agent = QLearningAgent()

    rewards_history = []
    portfolio_history = []

    print(f"Training for {episodes} episodes...\n")

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        final_value = env.portfolio_value()
        rewards_history.append(total_reward)
        portfolio_history.append(final_value)

        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_portfolio = np.mean(portfolio_history[-10:])
            print(
                f"Episode {episode:4d} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Portfolio: ${avg_portfolio:,.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    agent.save()
    plot_training(rewards_history, portfolio_history)
    print("\nTraining complete.")


def plot_training(rewards, portfolios):
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.plot(rewards, alpha=0.6, color="steelblue")
    ax1.set_title("Total Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    ax2.plot(portfolios, alpha=0.6, color="seagreen")
    ax2.axhline(y=10000, color="red", linestyle="--", label="Starting balance ($10,000)")
    ax2.set_title("Final Portfolio Value per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/training_results.png")
    print("Training plot saved to results/training_results.png")


if __name__ == "__main__":
    # generate data if it doesn't exist
    if not os.path.exists("data/stock_data.csv"):
        from src.generate_data import generate_stock_data
        generate_stock_data()

    train(episodes=100)
