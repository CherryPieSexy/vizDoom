import torch
from torch import nn
import torch.nn.functional as fun
import numpy as np


class DQN(nn.Module):
    def __init__(self, n_actions, epsilon=1.0):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x_screens):
        """Forward

        :param x_screens: screen batch of shape [batch, channels, height, width]
        :return: estimated q-values of shape [batch, n_actions],
        """
        x = fun.relu(self.conv1(x_screens))
        x = fun.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        q_values = self.fc2(fun.relu(self.fc1(x)))
        return q_values

    def sample_actions(self, screens):
        # noinspection PyCallingNonCallable, PyUnresolvedReferences
        screens = torch.tensor(screens, dtype=torch.float32)
        if len(screens.size()) == 3:
            screens.unsqueeze_(0)
        q_values = self.forward(screens).detach().numpy()

        eps = self.epsilon
        batch_size, n_actions = q_values.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = q_values.argmax(axis=-1)

        should_explore = np.random.choice([0, 1], batch_size, p=[1-eps, eps])
        return np.where(should_explore, random_actions, best_actions)
