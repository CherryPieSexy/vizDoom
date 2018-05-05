from collections import namedtuple

HyperParameters = namedtuple('HyperParameters', [
    'config_file',
    'train',
    'watch_n_episodes',

    'train_skiprate',
    'test_skiprate',
    'replay_size',
    'screen_size',
    'log_folder',

    'cuda',
    'learning_rate',
    'n_epoch',
    'steps_per_epoch',
    'play_steps',
    'batch_size',
    'time_size',
    'tests_per_epoch',
    'start_epsilon',
    'end_epsilon',

    'load_model',
])

# Basic
hp_basic = HyperParameters(config_file='scenarios/basic.cfg',
                           train=False, watch_n_episodes=10,
                           train_skiprate=10, test_skiprate=4,
                           replay_size=10 ** 5, screen_size=(1, 30, 45),
                           log_folder='logs/Basic/DQN/', cuda=False,
                           learning_rate=0.0025,
                           n_epoch=20, steps_per_epoch=2000, play_steps=1,
                           batch_size=64, time_size=1, tests_per_epoch=100,
                           start_epsilon=1.0, end_epsilon=0.1,
                           load_model='epoch_19.pth')

# Center defence
# TODO
# hp = HyperParameters(config_file='scenarios/basic.cfg', skiprate=12,
#                      replay_size=10**5, screen_size=(1, 30, 45),
#                      log_folder='logs/Basic/DQN/', cuda=False,
#                      learning_rate=0.0025,
#                      n_epoch=20, steps_per_epoch=2000, play_steps=1,
#                      batch_size=64, time_size=1, tests_per_epoch=100,
#                      start_epsilon=1.0, end_epsilon=0.1,
#                      load_model=None, watch_model=True)

# TODO: Deadly corridor, Health gathering
