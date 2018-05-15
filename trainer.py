import torch
from torch.nn.functional import mse_loss
from tensorboardX import SummaryWriter
from utils import reward_shaping, screen_transform
from tqdm import trange, TqdmSynchronisationWarning
import warnings
import numpy as np


class Trainer:
    def __init__(self, cuda, scenario,
                 environment, test_environment, experience_replay,
                 policy_net, target_net, optimizer, not_update,
                 log_folder, gamma=0.99):
        self.scenario = scenario
        # noinspection PyUnresolvedReferences
        self.device = torch.device("cuda" if cuda else "cpu")

        self._environment = environment
        self._test_environment = test_environment
        self._experience_replay = experience_replay

        self._policy_net = policy_net.to(self.device)
        self._target_net = target_net.to(self.device)
        self._target_net.eval()  # should always be in 'eval' mode
        self._optimizer = optimizer

        self._not_update = not_update

        self._episodes_done = 0
        self._episode_reward = 0.0
        self._policy_state = None
        self._log_folder = log_folder
        self._writer = SummaryWriter(log_folder)
        self._gamma = gamma

    def train(self, n_epoch,
              steps_per_epoch, play_steps, batch_size, time_size,
              tests_per_epoch, start_epsilon, end_epsilon):
        """General training function"""
        self._play_and_record(1000)
        n = len(str(n_epoch - 1))
        self._policy_net.epsilon = start_epsilon
        epsilon_decay = (start_epsilon - end_epsilon) / (n_epoch - 1)
        for epoch in range(n_epoch):
            self._target_net.load_state_dict(self._policy_net.state_dict())
            self._epoch(steps_per_epoch, play_steps, epoch, n, batch_size, time_size)
            epsilon = self._policy_net.epsilon
            self._policy_net.epsilon = end_epsilon
            test_shaped, test_rewards = self._test_policy(tests_per_epoch)
            self._writer.add_scalar(self.scenario + '/test mean shaped', test_shaped, epoch)
            self._writer.add_scalar(self.scenario + '/test mean reward', test_rewards, epoch)

            self._policy_net.epsilon = epsilon - epsilon_decay

        self._policy_net.epsilon += epsilon_decay
        self.save_policy()

    def _play_and_record(self, n_steps):
        """Plays the game for n_steps and stores every
        <screen, features, action, reward, done> tuple in replay memory.

        Returns mean reward
        """
        self._policy_net.eval()
        mean_reward = 0.0
        for step in range(n_steps):
            screen, features = self._environment.observe()
            screen = screen_transform(self.scenario, screen)
            action, self._policy_state = self._policy_net.sample_actions(
                self.device,
                screen[None, None],  # add [batch, time] dimensions
                self._policy_state)
            action = action[0]
            reward, done = self._environment.step(action)
            if not done:
                _, new_features = self._environment.observe()
                # if episode is not ended yet, evaluate shaped reward
                self._episode_reward += reward_shaping[self.scenario](reward, features, new_features)
            else:
                # if episode is just ended, reward needn't to be shaped
                self._episode_reward += reward
                self._writer.add_scalar(self.scenario + '/episode shaped reward',
                                        self._episode_reward, self._episodes_done)
                self._writer.add_scalar(self.scenario + '/episode reward',
                                        self._environment.get_episode_reward(), self._episodes_done)
                self._episodes_done += 1
                self._environment.reset()
                self._episode_reward = 0.0

                self._policy_state = None
            self._experience_replay.add(screen.numpy(), action, reward, done)
            mean_reward += reward
        return mean_reward / n_steps

    def _epoch(self, n_steps, play_steps, epoch, n_epoch, batch_size, time_size):
        """Plays the game for a given number of steps with current policy,
        stores transitions history in the Experience Replay,
        samples transitions from ER and trains policy_net on them,
        calls log writer for every iteration

        :param n_steps: number of training steps
        :param play_steps: number of steps to play and record
        :param epoch: current epoch
        :param n_epoch: length of number of epoch, needs to form good looking progress bar
        :param batch_size:
        :param time_size:
        """
        steps = trange(n_steps)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', TqdmSynchronisationWarning)
            steps.set_description('Epoch {:={n}d}'.format(epoch, n=n_epoch), refresh=False)
            for step in steps:
                mean_reward = self._play_and_record(play_steps)

                sample = self._experience_replay.sample(batch_size, time_size)
                td_loss = self._train_on_batch(sample, batch_size, time_size)
                self._writer.add_scalar(self.scenario + '/td loss', td_loss, step + epoch * n_steps)
                self._writer.add_scalar(self.scenario + '/train batch mean shaped reward',
                                        mean_reward, step + epoch * n_steps)

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    def _train_on_batch(self, sample, batch, time):
        """Calculates TD loss for a single batch

        sample = (
            (np.array) screen batch of shape [batch, (time+1), screen_shape]
            (np.array) action batch of shape [batch, time]
            (np.array) reward batch of shape [batch, time]
            (np.array) finish batch of shape [batch, time]
        )
        :return: mse loss, torch.tensor
        """
        self._policy_net.train()
        self._optimizer.zero_grad()

        # -----------------------------------forward---------------------------------
        screens, actions, rewards, is_done = sample
        screens = torch.tensor(screens, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        is_done = torch.tensor(is_done, dtype=torch.float32, device=self.device)

        curr_state_q_values, _ = self._policy_net(screens, None)
        next_state_q_values, _ = self._target_net(screens[:, :], None)

        q_values_for_actions = curr_state_q_values[:, :-1].contiguous().\
            view(batch * time, -1)[np.arange(batch * time), actions.ravel()].view(batch, time)

        # ------------------------------------simple---------------------------------
        # target_q_values = rewards + self._gamma * (1.0 - is_done) * next_state_q_values[:, 1:].max(-1)[0]

        # ------------------------------------double---------------------------------
        # a_online = curr_state_q_values[:, 1:].max(-1)[1].view(-1)
        # next_state_q_values = next_state_q_values[:, 1:].contiguous().\
        #     view(batch * time, -1)[np.arange(batch * time), a_online].view(batch, time)
        # target_q_values = rewards + self._gamma * (1.0 - is_done) * next_state_q_values

        # ----------------------------multi-step + double----------------------------
        a_online = curr_state_q_values[:, -1].max(-1)[1]
        target_q = next_state_q_values[np.arange(batch), -1, a_online]
        target_q_values = torch.zeros(batch, time, dtype=torch.float32, device=self.device)
        for i in reversed(range(time)):
            target_q = rewards[:, i] + self._gamma * (1.0 - is_done[:, i]) * target_q
            target_q_values[:, i] = target_q
        target_q_values.to(self.device)

        # -------------------------------------loss----------------------------------
        loss = mse_loss(q_values_for_actions[:, self._not_update:], target_q_values[:, self._not_update:].detach())
        # -----------------------------------backward--------------------------------
        loss.backward()
        for p in self._policy_net.parameters():
            p.grad.data.clamp_(-5, 5)
        # -----------------------------------optimize--------------------------------
        self._optimizer.step()
        return loss

    def _test_policy(self, n_tests):
        self._policy_net.eval()
        shaped_rewards, rewards = [], []
        with torch.no_grad():
            for _ in range(n_tests):
                episode_reward = 0.0
                policy_state = None
                self._test_environment.reset()
                while True:
                    screen, features = self._test_environment.observe()
                    screen = screen_transform(self.scenario, screen)
                    action, policy_state = self._policy_net.sample_actions(
                        self.device,
                        screen[None, None],
                        policy_state)
                    action = action[0]
                    reward, done = self._test_environment.step(action)
                    if not done:
                        _, new_features = self._test_environment.observe()
                        episode_reward += reward_shaping[self.scenario](reward, features, new_features)
                    else:
                        episode_reward += reward
                        shaped_rewards += [episode_reward]
                        rewards += [self._test_environment.get_episode_reward()]
                        break
            mean_shaped = sum(shaped_rewards) / len(shaped_rewards)
            mean_rewards = sum(rewards) / len(rewards)
        return mean_shaped, mean_rewards

    def save_policy(self):
        torch.save({
            'policy_net_state': self._policy_net.state_dict(),
        }, self._log_folder + '/model.pth')
        print('Model saved in: {}'.format(self._log_folder + '/model.pth'))
