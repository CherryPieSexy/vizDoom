from collections import namedtuple

HyperParametersTrain = namedtuple('HyperParametersTrain', [
    'scenario', 'agent', 'cuda',

    'train_skiprate',
    'test_skiprate',
    'replay_size',
    'screen_size',

    'learning_rate',
    'n_epoch',
    'steps_per_epoch',
    'play_steps',
    'batch_size',
    'time_size',
    'not_update',  # first 'not_update' observations needed to estimate policy state
    'tests_per_epoch',
    'start_epsilon',
    'end_epsilon',
])

HyperParametersTest = namedtuple('HyperParametersTest', [
    'scenario',
    'agent',

    'test_skiprate',
    'epsilon',
    'n_episodes'
])

# basic
hp_basic_train = HyperParametersTrain(scenario='basic', agent='Distributional', cuda=False,
                                      train_skiprate=10, test_skiprate=4,
                                      replay_size=10 ** 4, screen_size=(1, 30, 45),
                                      learning_rate=0.0025,
                                      n_epoch=21, steps_per_epoch=2000, play_steps=1,
                                      batch_size=32, time_size=3, not_update=1, tests_per_epoch=100,
                                      start_epsilon=1.0, end_epsilon=0.01
                                      )
hp_basic_test = HyperParametersTest(scenario='basic', agent='Distributional',
                                    test_skiprate=4, epsilon=0.01, n_episodes=10)

# Deadly corridor
hp_d_cor_train = HyperParametersTrain(scenario='deadly_corridor', agent='Distributional', cuda=True,
                                      train_skiprate=4, test_skiprate=1,
                                      replay_size=10 ** 5, screen_size=(3, 60, 108),
                                      learning_rate=0.0002,
                                      n_epoch=21, steps_per_epoch=8000, play_steps=15,
                                      batch_size=128, time_size=10, not_update=4, tests_per_epoch=100,
                                      start_epsilon=1.0, end_epsilon=0.01
                                      )
hp_d_cor_test = HyperParametersTest(scenario='deadly_corridor', agent='Distributional',
                                    test_skiprate=1, epsilon=0.01, n_episodes=10)

# Defend The Center
hp_def_c_train = HyperParametersTrain(scenario='defend_the_center', agent='Distributional', cuda=True,
                                      train_skiprate=4, test_skiprate=1,
                                      replay_size=10 ** 5, screen_size=(3, 60, 108),
                                      learning_rate=0.0002,
                                      n_epoch=21, steps_per_epoch=8000, play_steps=15,
                                      batch_size=128, time_size=10, not_update=4, tests_per_epoch=100,
                                      start_epsilon=1.0, end_epsilon=0.01
                                      )
hp_def_c_test = HyperParametersTest(scenario='defend_the_center', agent='Distributional',
                                    test_skiprate=1, epsilon=0.01, n_episodes=5)

# Health Gathering
hp_h_gth_train = HyperParametersTrain(scenario='health_gathering', agent='Distributional', cuda=False,
                                      train_skiprate=4, test_skiprate=1,
                                      replay_size=10 ** 5, screen_size=(3, 60, 108),
                                      learning_rate=0.0002,
                                      n_epoch=21, steps_per_epoch=8000, play_steps=15,
                                      batch_size=128, time_size=10, not_update=4, tests_per_epoch=100,
                                      start_epsilon=1.0, end_epsilon=0.01
                                      )
hp_h_gth_test = HyperParametersTest(scenario='health_gathering', agent='Distributional',
                                    test_skiprate=1, epsilon=0.01, n_episodes=5)
