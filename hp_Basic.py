from collections import namedtuple

HyperParameters = namedtuple('HyperParameters', [
    'config_file'
    'log_folder'

    'cuda',
    'n_epoch',
    'steps_per_epoch'
    'play_steps'
    'batch_size'
    'time_size'
    'tests_per_epoch'
    'start_epsilon'
    'end_epsilon'

    'load_model'
])

hp = HyperParameters(config_file='scenarios/basic.cfg', log_folder='logs/Basic/DQN',
                     cuda=False, 
                     n_epoch=20, steps_per_epoch=2000, play_steps=1,
                     batch_size=64, time_size=1, tests_per_epoch=100, 
                     start_epsilon=1.0, end_epsilon=0.1,
                     load_model=None)
