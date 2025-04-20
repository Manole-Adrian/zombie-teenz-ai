import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
from IQN.IQNNetwork import IQNNetwork
import torch

class IQNAgent:
    def __init__(self, observation_space, action_space, num_quantiles=32, hidden_size=64, lr=1e-4, gamma=0.99):
        self.model = IQNNetwork(observation_space, action_space, num_quantiles, hidden_size)
        self.target_model = IQNNetwork(observation_space, action_space, num_quantiles, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.num_quantiles = num_quantiles

        # Copy the weights of the model to the target network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.quantile_tau = 0.5  # Quantile tau, typically in the range [0, 1]

    def select_action(self, state, epsilon=0.1):
        # Epsilon-greedy policy
        if random.random() < epsilon:
            return random.choice(range(self.model.action_space))  # Random action
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            quantile_values = self.model(state_tensor)
            # Select action with maximum expected quantile value
            action = quantile_values.mean(dim=-1).max(dim=-1)[1].item()
            return action

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        return random.sample(self.replay_buffer, self.batch_size)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = self.sample_batch()
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(states, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)

        # Compute the target Q-values
        next_quantile_values = self.target_model(next_states_tensor)
        next_quantile_values = next_quantile_values.mean(dim=-1)  # Average over quantiles to get a scalar
        next_q_values = next_quantile_values.max(dim=-1)[0]  # Max over actions to get the next Q-value

        # Target Q-value for each transition
        target_q_values = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q_values

        # Get the predicted quantile values for the selected actions
        quantile_values = self.model(states_tensor)
        selected_quantiles = quantile_values.gather(1, actions_tensor.unsqueeze(-1).repeat(1, 1, self.num_quantiles))

        # Since target_q_values is a scalar, we need to expand it to match the shape of selected_quantiles
        target_q_values_expanded = target_q_values.unsqueeze(-1).repeat(1, self.num_quantiles)

        # Compute the loss
        loss = quantile_huber_loss(selected_quantiles.squeeze(-1), target_q_values_expanded, self.quantile_tau)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def quantile_huber_loss(quantile_values, target, quantile_tau, kappa=1.0):
    """Compute the quantile Huber loss."""
    # Compute the difference between quantile values and target
    delta = target - quantile_values
    abs_delta = torch.abs(delta)

    # Use Huber loss for the difference
    huber_loss = torch.where(abs_delta < kappa,
                             0.5 * delta ** 2,
                             kappa * (abs_delta - 0.5 * kappa))

    # Weight by the quantile
    loss = huber_loss * (quantile_tau - (delta < 0).float())
    return loss.mean()

