import numpy as np


# TODO: add in-game features here!
class ReplayMemory:
    def __init__(self, capacity, screen_shape):
        self._capacity = capacity
        self._cursor = 0
        self._full = False

        self.screens = np.zeros((capacity,) + screen_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.is_done = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self._capacity if self._full else self._cursor

    def add(self, screen, action, reward, is_done):
        self.screens[self._cursor] = screen
        self.actions[self._cursor] = action
        self.rewards[self._cursor] = reward
        self.is_done[self._cursor] = is_done
        self._cursor += 1
        if self._cursor >= self._capacity:
            self._cursor = 0
            self._full = True

    def sample(self, batch_size, hist_size):
        """sample

        :param batch_size:
        :param hist_size:
        :return:
        """
        idx = np.zeros(batch_size, dtype=np.int32)
        count = 0

        while count < batch_size:
            # index of the last element
            index = np.random.randint(hist_size - 1, self.__len__() - 1)

            if self._cursor <= index + 1 < self._cursor + hist_size:
                continue

            if np.any(self.is_done[index - (hist_size - 1):index]):
                continue

            idx[count] = index
            count += 1

        all_indices = idx.reshape((-1, 1)) + np.arange(-(hist_size - 1), 2)
        screens = self.screens[all_indices]
        actions = self.actions[all_indices[:, :-1]]
        rewards = self.rewards[all_indices[:, :-1]]
        is_done = self.is_done[all_indices[:, :-1]]

        return screens, actions, rewards, is_done
