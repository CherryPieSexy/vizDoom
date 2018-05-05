import torch
from torch import nn
import torch.nn.functional as fun
import numpy as np


class DRQN(nn.Module):
    def __init__(self, n_actions, epsilon=1.0):
        super(DRQN, self).__init__()
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)

        self.lstm = nn.LSTM(192, 128, batch_first=True)
        self.fc1 = nn.Linear(128, n_actions)

    def forward(self, x_screens, hidden):
        """Forward

        :param x_screens: screen batch of shape [batch, time, channels, height, width]
        :param hidden: lstm's hidden state
        :return: estimated q-values of shape [batch, time, n_actions],
        """
        batch, time = x_screens.shape[:2]
        chw = x_screens.shape[2:]
        x_screens = x_screens.view(batch*time, *chw)

        x = fun.relu(self.conv1(x_screens))
        x = fun.relu(self.conv2(x))
        x = x.view(batch, time, -1)

        x, hidden = self.lstm(x, hidden)

        q_values = self.fc1(x)  # have shape [batch, time, n_actions] now
        return q_values, hidden

    # TODO: fix
    def sample_actions(self, screens):
        # noinspection PyCallingNonCallable, PyUnresolvedReferences
        screens = torch.tensor(screens, dtype=torch.float32)
        if len(screens.size()) == 3:
            screens.unsqueeze_(0)  # add batch
            screens.unsqueeze_(0)  # add time
        q_values, _ = self.forward(screens, None)
        q_values = q_values.detach().numpy()
        print(q_values)

        eps = self.epsilon
        batch_size, n_actions = q_values.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = q_values.argmax(axis=-1)

        should_explore = np.random.choice([0, 1], batch_size, p=[1-eps, eps])
        return np.where(should_explore, random_actions, best_actions)
