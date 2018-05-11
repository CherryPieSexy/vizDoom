from torch import load
from doom_environment import DoomEnvironment
from utils import watch_agent
from models import agent
from time import sleep
# from hyperparameters import hp_basic_test as hp
from hyperparameters import hp_d_cor_test as hp
# from hyperparameters import hp_def_c_test as hp
# from hyperparameters import hp_h_gth_test as hp

if __name__ == '__main__':
    print('---------------------------- vizDoom watching script ---------------------------')

    test_env = DoomEnvironment('scenarios/' + hp.scenario + '.cfg', False, hp.test_skiprate)
    test_env.make_visible()
    policy_net = agent[hp.agent](hp.scenario, 2 ** test_env.get_n_buttons(), hp.epsilon)
    policy_net.load_state_dict(load(
        'logs/' + hp.scenario + '/' + hp.agent + '/model.pth',
        map_location=lambda storage, loc: storage)['policy_net_state'])

    print('scenario: {}, agent: {}'.format(hp.scenario, hp.agent))
    print('loaded model: {}'.format('logs/' + hp.scenario + '/' + hp.agent + '/model.pth'))
    print('agent\'s epsilon: {}'.format(hp.epsilon))

    print('------------------------------- watch the model --------------------------------')
    print('n_episodes: {}'.format(hp.n_episodes))
    for _ in range(hp.n_episodes):
        reward, shaped = watch_agent(hp.scenario, policy_net, test_env)
        print('Episode {} done, reward: {}, shaped: {}'.format(_, reward, shaped))
        sleep(1.0)
        # if _ != hp.n_episodes - 1:
        #     test_env.reset()
    print('Exit')
