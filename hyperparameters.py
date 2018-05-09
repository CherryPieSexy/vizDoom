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

# Basic
# config_file = 'scenarios/' + hp.scenario + '.cfg'
# log_folder = 'logs/' + hp.scenario + '/' + hp.agent
# load_model = 'logs/' + hp.scenario + '/' + hp.agent + '/model.pth'
hp_basic_train = HyperParametersTrain(scenario='basic', agent='DQN', cuda=False,
                                      train_skiprate=10, test_skiprate=4,
                                      replay_size=10 ** 4, screen_size=(1, 30, 45),
                                      learning_rate=0.0025,
                                      n_epoch=20, steps_per_epoch=2000, play_steps=1,
                                      batch_size=64, time_size=1, not_update=0, tests_per_epoch=100,
                                      start_epsilon=1.0, end_epsilon=0.0
                                      )
hp_basic_test = HyperParametersTest(scenario='basic', agent='DQN',
                                    test_skiprate=4, epsilon=0.1, n_episodes=10)

# # Deadly corridor
# hp_d_cor_train = HyperParametersTrain(scenario='deadly_corridor', config_file='scenarios/deadly_corridor.cfg',
#                                       train_skiprate=4, test_skiprate=1,
#                                       replay_size=10 ** 5, screen_size=(3, 60, 108),
#                                       log_folder='logs/Deadly_Corridor/DQN/', cuda=True,
#                                       learning_rate=0.0025,
#                                       n_epoch=30, steps_per_epoch=5000, play_steps=10,
#                                       batch_size=64, time_size=7, not_update=0, tests_per_epoch=30,
#                                       start_epsilon=1.0, end_epsilon=0.0
#                                       )
# hp_d_cor_test = HyperParametersTest(scenario='deadly_corridor', config_file='scenarios/deadly_corridor.cfg',
#                                     test_skiprate=1, load_model='logs/Deadly_Corridor/DQN/model.pth',
#                                     epsilon=0.1, n_episodes=10)
#
# # Defend The Center
# hp_def_c_train = HyperParametersTrain(scenario='defend_the_center', config_file='scenarios/defend_the_center.cfg',
#                                       train_skiprate=4, test_skiprate=1,
#                                       replay_size=10 ** 5, screen_size=(3, 60, 108),
#                                       log_folder='logs/Defend_The_Center/DQN/', cuda=True,
#                                       learning_rate=0.0025,
#                                       n_epoch=40, steps_per_epoch=1000, play_steps=10,
#                                       batch_size=64, time_size=7, not_update=0, tests_per_epoch=30,
#                                       start_epsilon=1.0, end_epsilon=0.0
#                                       )
# hp_def_c_test = HyperParametersTest(scenario='defend_the_center', config_file='scenarios/defend_the_center.cfg',
#                                     test_skiprate=1, load_model='logs/Defend_The_Center/DQN/model.pth',
#                                     epsilon=0.1, n_episodes=5)
# #
# # # Health Gathering
# hp_h_gth_train = HyperParametersTrain(scenario='health_gathering', config_file='scenarios/health_gathering.cfg',
#                                       train_skiprate=4, test_skiprate=1,
#                                       replay_size=10 ** 5, screen_size=(3, 60, 108),
#                                       log_folder='logs/Health_Gathering/DQN/', cuda=False,
#                                       learning_rate=0.0025,
#                                       n_epoch=30, steps_per_epoch=5000, play_steps=10,
#                                       batch_size=64, time_size=7, not_update=0, tests_per_epoch=30,
#                                       start_epsilon=1.0, end_epsilon=0.0
#                                       )
# hp_h_gth_test = HyperParametersTest(scenario='health_gathering', config_file='scenarios/health_gathering.cfg',
#                                     test_skiprate=1, load_model='logs/Health_Gathering/DQN/model.pth',
#                                     epsilon=0.1, n_episodes=5)
