import torch
from doom_environment import DoomEnvironment
from experience_replay import ReplayMemory
from models.dqn import DQN
from trainer import Trainer
from utils import watch_agent
# from hyperparameters import hp_basic as hp
# from hyperparameters import hp_def_c as hp
from hyperparameters import hp_d_cor as hp
from time import sleep


if __name__ == '__main__':
    print('------------------------------ vizDoom main script -----------------------------')
    print('scenario: {}'.format(hp.scenario))
    train_env = DoomEnvironment(hp.config_file, False, hp.train_skiprate)
    test_env = DoomEnvironment(hp.config_file, False, hp.test_skiprate)

    policy_net = DQN(hp.scenario, 2 ** train_env.get_n_buttons())
    target_net = DQN(hp.scenario, 2 ** train_env.get_n_buttons())
    optimizer = torch.optim.RMSprop(policy_net.parameters(), hp.learning_rate)

    if hp.load_model is not None:
        print('loaded model: {}'.format(hp.load_model))
        loaded_state = torch.load(hp.log_folder + 'checkpoints/' + hp.load_model)
        policy_net.load_state_dict(loaded_state['policy_net_state'])
        target_net.load_state_dict(loaded_state['policy_net_state'])
        optimizer.load_state_dict(loaded_state['optimizer'])
        policy_net.epsilon = loaded_state['epsilon']

    if hp.train:
        print('----------------------------------- Training -----------------------------------')
        print('training parameters:')
        print('n_epoch: {}, steps_per_epoch: {}, play_steps: {}'.format(hp.n_epoch, hp.steps_per_epoch, hp.play_steps))
        print('batch_size: {}, time_size: {}, tests_per_epoch: {}'.format(hp.batch_size, hp.time_size,
                                                                          hp.tests_per_epoch))
        er = ReplayMemory(hp.replay_size, hp.screen_size)
        trainer = Trainer(scenario=hp.scenario, cuda=hp.cuda,
                          environment=train_env, test_environment=test_env,
                          experience_replay=er,
                          policy_net=policy_net, target_net=target_net, optimizer=optimizer,
                          log_folder=hp.log_folder)
        trainer.train(n_epoch=hp.n_epoch, steps_per_epoch=hp.steps_per_epoch,
                      play_steps=hp.play_steps, batch_size=hp.batch_size, time_size=hp.time_size,
                      tests_per_epoch=hp.tests_per_epoch,
                      start_epsilon=hp.start_epsilon, end_epsilon=hp.end_epsilon)
        print('-------------------------------- Training done ---------------------------------')

    if hp.watch_n_episodes != 0:
        print('------------------------------- watch the model --------------------------------')
        test_env.make_visible()
        policy_net.epsilon = hp.end_epsilon
        for _ in range(hp.watch_n_episodes):
            test_env.reset()
            reward = watch_agent(hp.scenario, policy_net, test_env)
            print('Episode {} done, reward: {}'.format(_, reward))
            sleep(1.0)

    print('------------------------------------ Done! -------------------------------------')
