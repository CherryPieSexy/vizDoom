import torch
from doom_environment import DoomEnvironment
from experience_replay import ReplayMemory
from dqn import DQN
from trainer import Trainer
# from hyperparameters import hp

cfg = 'scenarios/basic.cfg'
train_env = DoomEnvironment(cfg, False, 12)
test_env = DoomEnvironment(cfg, False, 12)

train_env._game.get_screen

# er = ReplayMemory(10**5, (3, 60, 108))
# agent = DQN(8)
# target = DQN(8)
# target.load_state_dict(agent.state_dict())

# optim = torch.optim.RMSprop(agent.parameters(), 0.0025)

# trainer = Trainer(train_env, test_env, er, agent, target, optim, 'logs/Basic/DQN/')
# trainer.train(n_epoch=20, steps_per_epoch=2000,
#               play_steps=1, batch_size=64, time_size=1,
#               tests_per_epoch=100)
