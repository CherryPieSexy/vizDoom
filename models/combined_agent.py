import torch
from torch import nn
import torch.nn.functional as fun
import numpy as np


class CombinedAgent(nn.Module):
    def __init__(self, scenario, n_actions, epsilon=1.0):
        super(CombinedAgent, self).__init__()
        self.scenario = scenario
        self.n_actions = n_actions
        self.epsilon = epsilon

        if self.scenario == 'basic':
            self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
            self.dropout = nn.Dropout(0.5)

            self.lstm = nn.LSTM(192, 128, batch_first=True)
            self.advantage_layer = nn.Linear(128, n_actions)
            self.value_layer = nn.Linear(128, 1)
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
            self.dropout = nn.Dropout(0.5)

            self.lstm = nn.LSTM(2560, 512, batch_first=True)
            self.advantage_layer = nn.Linear(512, n_actions)
            self.value_layer = nn.Linear(512, 1)

    def forward(self, x_screens, lstm_state):
        batch, time = x_screens.shape[:2]
        chw = x_screens.shape[2:]
        x_screens = x_screens.contiguous().view(batch*time, *chw)
        x = fun.relu(self.conv1(x_screens))
        x = fun.relu(self.conv2(x))
        if self.scenario != 'basic':
            x = fun.relu(self.conv3(x))
        x = x.view(batch, time, -1)
        x = self.dropout(x)
        x, lstm_state = self.lstm(x, lstm_state)
        advantage = self.advantage_layer(x)
        value = self.value_layer(x)
        q_values = value + advantage - advantage.mean(-1, keepdim=True)
        return q_values, lstm_state

    # noinspection PyCallingNonCallable, PyUnresolvedReferences
    def sample_actions(self, device, screen, lstm_state):
        screen = torch.tensor(screen, dtype=torch.float32, device=device)
        q_values, lstm_state = self.forward(screen, lstm_state)
        q_values = q_values.detach().cpu().numpy().reshape(-1)

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions, 1)[0]
        else:
            action = q_values.argmax(axis=-1)
        return [action], lstm_state


if __name__ == '__main__':
    net = CombinedAgent('basic', 8, 1.0)
    a, _ = net.sample_actions(
        'cpu',
        np.random.normal(size=[1, 1, 1, 30, 45]),
        None
    )
    print(a)
