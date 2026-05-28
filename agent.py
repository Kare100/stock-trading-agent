"""
Q-Learning Agent for stock trading.

Uses a discretized Q-table to map states to actions.
"""

import numpy as np
import pickle
import os


class QLearningAgent:
    def __init__(
        self,
        state_size=5,
        action_size=3,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        bins=10,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.bins = bins

        # discretize state space into bins for Q-table
        self.q_table = {}

    def _discretize(self, state):
        """Convert continuous state to a discrete tuple for Q-table lookup."""
        return tuple(np.round(state, 2))

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        key = self._discretize(state)
        if key not in self.q_table:
            return np.random.randint(self.action_size)
        
        return np.argmax(self.q_table[key])

    def update(self, state, action, reward, next_state, done):
        """Q-table update rule."""
        key = self._discretize(state)
        next_key = self._discretize(next_state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.action_size)

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_key])

        self.q_table[key][action] += self.lr * (target - self.q_table[key][action])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="models/q_agent.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Agent saved to {path}")

    def load(self, path="models/q_agent.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        self.epsilon = self.epsilon_min
        print(f"Agent loaded from {path}")
