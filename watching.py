from torch import load
from doom_environment import DoomEnvironment
from utils import watch_agent
from models.dqn import DQN
from time import sleep
from hyperparameters import hp_basic_test as hp
# from hyperparameters import hp_def_c_test as hp
# from hyperparameters import hp_d_cor_test as hp

if __name__ == '__main__':
    print('---------------------------- vizDoom watching script ---------------------------')

    test_env = DoomEnvironment(hp.config_file, False, hp.test_skiprate)
    test_env.make_visible()
    policy_net = DQN(hp.scenario, 2 ** test_env.get_n_buttons())
    policy_net.load_state_dict(load(hp.load_model)['policy_net_state'])
    policy_net.epsilon = hp.epsilon

    print('scenario: {}'.format(hp.scenario))
    print('loaded model: {}'.format(hp.load_model))
    print('agent\'s epsilon: {}'.format(hp.epsilon))

    print('------------------------------- watch the model --------------------------------')
    print('n_episodes: {}'.format(hp.n_episodes))
    for _ in range(hp.n_episodes):
        reward, shaped = watch_agent(hp.scenario, policy_net, test_env)
        print('Episode {} done, reward: {}, shaped: {}'.format(_, reward, shaped))
        sleep(1.0)
        if _ != hp.n_episodes - 1:
            test_env.reset()
    print('Exit')
