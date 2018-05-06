from collections import namedtuple

HyperParameters = namedtuple('HyperParameters', [
    'scenario',

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
hp_basic = HyperParameters(scenario='basic', config_file='scenarios/basic.cfg',
                           train=True, watch_n_episodes=10,
                           train_skiprate=10, test_skiprate=4,
                           replay_size=10 ** 5, screen_size=(1, 30, 45),
                           log_folder='logs/Basic/DQN/', cuda=False,
                           learning_rate=0.0025,
                           n_epoch=20, steps_per_epoch=2000, play_steps=1,
                           batch_size=64, time_size=1, tests_per_epoch=100,
                           start_epsilon=1.0, end_epsilon=0.0,
                           load_model=None)

# Defend_The_Center
# hp_defend_the_center = HyperParameters(scenario='defend_the_center', config_file='scenarios/defend_the_center.cfg',
#                                        train=False, watch_n_episodes=10,
#                                        train_skiprate=10, test_skiprate=4,
#                                        replay_size=10 ** 5, screen_size=(3, 30, 45),
#                                        log_folder='logs/Defend_The_Center/DQN/', cuda=False,
#                                        learning_rate=0.0025,
#                                        n_epoch=20, steps_per_epoch=2000, play_steps=1,
#                                        batch_size=64, time_size=1, tests_per_epoch=100,
#                                        start_epsilon=1.0, end_epsilon=0.1,
#                                        load_model=None)

# TODO: Deadly corridor, Health gathering
