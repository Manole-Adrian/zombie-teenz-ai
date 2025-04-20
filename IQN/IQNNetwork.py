import torch
import torch.nn as nn
import torch.nn.functional as F


class IQNNetwork(nn.Module):
    def __init__(self, observation_space, action_space, num_quantiles=32, hidden_size=64):
        super(IQNNetwork, self).__init__()

        # Define a simple MLP model
        self.fc1 = nn.Linear(observation_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space * num_quantiles)

        self.num_quantiles = num_quantiles
        self.action_space = action_space

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        quantile_values = self.fc3(x)

        # Reshape the output to (batch_size, action_space, num_quantiles)
        return quantile_values.view(-1, self.action_space, self.num_quantiles)


