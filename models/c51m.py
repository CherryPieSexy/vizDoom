import torch
from torch import nn
import torch.nn.functional as fun
import numpy as np


class C51M(nn.Module):
    def __init__(self, scenario, n_actions, epsilon=1.0, v_min=10, v_max=150):
        super(C51M, self).__init__()
        self.scenario = scenario
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.v_min = v_min
        self.v_max = v_max

        if self.scenario == 'basic':
            # 21 atom for basic scenario
            self.n_atoms = 21
            self.delta_z = (v_max - v_min) / (self.n_atoms - 1)
            self.atom_support = np.array([v_min + i * self.delta_z for i in range(self.n_atoms)], dtype=np.float32)
            self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
            self.dropout = nn.Dropout(0.5)

            self.lstm = nn.LSTM(192, 128, batch_first=True)
            self.value_layer = nn.Linear(128, self.n_atoms)
            self.advantages = nn.ModuleList()
            for _ in range(n_actions):
                self.advantages.append(nn.Linear(128, self.n_atoms))
        else:
            # 51 atom for basic scenario
            self.n_atoms = 51
            self.delta_z = (v_max - v_min) / (self.n_atoms - 1)
            self.atom_support = np.array([v_min + i * self.delta_z for i in range(self.n_atoms)], dtype=np.float32)
            self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
            self.dropout = nn.Dropout(0.5)

            self.lstm = nn.LSTM(2560, 512, batch_first=True)
            self.value_layer = nn.Linear(512, self.n_atoms)
            self.advantages = nn.ModuleList()
            for _ in range(n_actions):
                self.advantages.append(nn.Linear(512, self.n_atoms))

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
        value = self.value_layer(x)
        # noinspection PyUnresolvedReferences
        advantages = torch.stack([adv(x) for adv in self.advantages], dim=-1)

        atom_logits = value.unsqueeze(-1) + advantages - advantages.mean(-1, keepdim=True)
        atom_probabilities = fun.softmax(atom_logits, dim=-2)  # soft max along all atoms

        return atom_probabilities, lstm_state

    # noinspection PyCallingNonCallable, PyUnresolvedReferences
    def sample_actions(self, device, screen, lstm_state):
        screen = torch.tensor(screen, dtype=torch.float32, device=device)
        z_probabilities, lstm_state = self.forward(screen, lstm_state)
        q_values = torch.matmul(torch.from_numpy(self.atom_support).to(device), z_probabilities)
        q_values = q_values.view(-1).detach().cpu().numpy()

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions, 1)[0]
        else:
            action = q_values.argmax(axis=-1)
        return [action], lstm_state


if __name__ == '__main__':
    net = C51M('basic', 8, 1.0)
    a, _ = net.sample_actions(
        'cpu',
        np.random.normal(size=[1, 1, 1, 30, 45]),
        None
    )
    print(a)
