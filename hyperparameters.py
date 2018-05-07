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
                           train=False, watch_n_episodes=1,
                           train_skiprate=10, test_skiprate=4,
                           replay_size=10 ** 5, screen_size=(1, 30, 45),
                           log_folder='logs/Basic/DQN/', cuda=False,
                           learning_rate=0.0025,
                           n_epoch=20, steps_per_epoch=2000, play_steps=1,
                           batch_size=64, time_size=1, tests_per_epoch=100,
                           start_epsilon=1.0, end_epsilon=0.0,
                           load_model=None)

# Defend_The_Center
hp_def_c = HyperParameters(scenario='defend_the_center', config_file='scenarios/defend_the_center.cfg',
                           train=True, watch_n_episodes=0,
                           train_skiprate=4, test_skiprate=1,
                           replay_size=10 ** 5, screen_size=(3, 60, 108),
                           log_folder='logs/Defend_The_Center/DQN/', cuda=True,
                           learning_rate=0.0025,
                           n_epoch=30, steps_per_epoch=4000, play_steps=1,
                           batch_size=64, time_size=4, tests_per_epoch=100,
                           start_epsilon=1.0, end_epsilon=0.0,
                           load_model=None)

# Deadly corridor
hp_d_cor = HyperParameters(scenario='deadly_corridor', config_file='scenarios/deadly_corridor.cfg',
                           train=False, watch_n_episodes=0,
                           train_skiprate=4, test_skiprate=1,
                           replay_size=10 ** 5, screen_size=(3, 60, 108),
                           log_folder='logs/Deadly_Corridor/DQN/', cuda=True,
                           learning_rate=0.0025,
                           n_epoch=30, steps_per_epoch=4000, play_steps=1,
                           batch_size=64, time_size=4, tests_per_epoch=100,
                           start_epsilon=1.0, end_epsilon=1.0,
                           load_model=None)
# TODO: Health gathering
# hp_h_gth
