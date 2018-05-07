import torch
import numpy as np
from torch.nn.functional import mse_loss
from tensorboardX import SummaryWriter
from utils import reward_shaping, screen_transform
from tqdm import trange, TqdmSynchronisationWarning
import warnings


class Trainer:
    def __init__(self, cuda, scenario,
                 environment, test_environment, experience_replay,
                 policy_net, target_net, optimizer,
                 log_folder, gamma=0.99):
        self.scenario = scenario
        # noinspection PyUnresolvedReferences
        self.device = torch.device("cuda" if cuda else "cpu")

        self._environment = environment
        self._test_environment = test_environment
        self._experience_replay = experience_replay

        self._policy_net = policy_net.to(self.device)
        self._target_net = target_net.to(self.device)
        self._optimizer = optimizer

        self._episodes_done = 0
        self._episode_reward = 0.0
        self._log_folder = log_folder
        self._writer = SummaryWriter(log_folder)
        self._gamma = gamma

    def train(self, n_epoch,
              steps_per_epoch, play_steps, batch_size, time_size,
              tests_per_epoch, start_epsilon, end_epsilon):
        """General training function"""
        self._play_and_record(time_size + 1)
        n = len(str(n_epoch-1))
        self._policy_net.epsilon = start_epsilon
        epsilon_decay = (start_epsilon - end_epsilon) / (n_epoch - 1)
        for epoch in range(n_epoch):
            self._epoch(steps_per_epoch, play_steps, epoch, n, batch_size, time_size)

            test_shaped, test_rewards = self._test_policy(tests_per_epoch)
            self._writer.add_scalar(self.scenario + '/test mean shaped', test_shaped, epoch)
            self._writer.add_scalar(self.scenario + '/test mean reward', test_rewards, epoch)

            self._target_net.load_state_dict(self._policy_net.state_dict())
            self._policy_net.epsilon -= epsilon_decay
            self.save(epoch)

        self._policy_net.epsilon += epsilon_decay

    def _play_and_record(self, n_steps):
        """Plays the game for n_steps and stores every
        <screen, features, action, reward, done> tuple in replay memory.

        Returns mean reward
        """
        mean_reward = 0.0
        for step in range(n_steps):
            screen, features = self._environment.observe()
            screen = screen_transform(self.scenario, screen)
            action = self._policy_net.sample_actions(self.device, screen)[0]
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
                                        self._episode_reward, self._episodes_done)
                self._episodes_done += 1
                self._environment.reset()
                self._episode_reward = 0.0
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
                td_loss = self._batch_loss(sample)
                self._train_step(td_loss)
                self._writer.add_scalar(self.scenario + '/td loss', td_loss, step + epoch * n_steps)
                self._writer.add_scalar(self.scenario + '/train batch mean shaped reward',
                                        mean_reward, step + epoch * n_steps)

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    def _batch_loss(self, sample):
        """Calculates TD loss for a single batch

        sample = {
            'screens': (np.array) screen batch of shape [batch, (time+1), screen_shape]
            'actions': (np.array) action batch of shape [batch, time]
            'rewards': [batch, time]
            'is_done': [batch, time]
        }
        :return: mse loss, torch.tensor
        """
        screens, actions, rewards, is_done = [sample[key] for key in sample]

        batch, time = actions.shape
        chw = screens.shape[2:]
        curr_state_q_values = self._policy_net(
            torch.tensor(screens[:, :-1], dtype=torch.float32, device=self.device).view(batch*time, *chw)
        )
        next_state_q_values = self._target_net(
            torch.tensor(screens[:,  1:], dtype=torch.float32, device=self.device).view(batch*time, *chw)
        )
        actions = actions.reshape(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        is_done = torch.tensor(is_done, dtype=torch.float32, device=self.device).view(-1)

        q_values_for_actions = curr_state_q_values[np.arange(batch*time), actions]
        target_q_values = rewards + self._gamma * (1.0 - is_done) * next_state_q_values.max(1)[0]
        loss = mse_loss(q_values_for_actions, target_q_values.detach())
        return loss

    def _test_policy(self, n_tests):
        shaped_rewards, rewards = [], []
        for _ in range(n_tests):
            episode_reward = 0.0
            while True:
                screen, features = self._test_environment.observe()
                action = self._policy_net.sample_actions(self.device, screen_transform(self.scenario, screen))[0]
                reward, done = self._test_environment.step(action)
                if not done:
                    _, new_features = self._test_environment.observe()
                    episode_reward += reward_shaping[self.scenario](reward, features, new_features)
                else:
                    episode_reward += reward
                    shaped_rewards += [episode_reward]
                    rewards += [self._test_environment.get_episode_reward()]
                    self._test_environment.reset()
                    break
        mean_shaped = sum(shaped_rewards) / len(shaped_rewards)
        mean_rewards = sum(rewards) / len(rewards)
        return mean_shaped, mean_rewards

    def _train_step(self, loss):
        self._optimizer.zero_grad()
        # TODO: add gradient clipping in future
        loss.backward()
        self._optimizer.step()

    def save(self, epoch):
        torch.save({
            'policy_net_state': self._policy_net.state_dict(),
            'target_net_state': self._target_net.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'epsilon': self._policy_net.epsilon
        }, self._log_folder + 'checkpoints/epoch_' + str(epoch) + '.pth')
