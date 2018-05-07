import torch
from doom_environment import DoomEnvironment
from experience_replay import ReplayMemory
from models.dqn import DQN
from trainer import Trainer
from hyperparameters import hp_basic_train as hp
# from hyperparameters import hp_def_c_train as hp
# from hyperparameters import hp_d_cor_train as hp


if __name__ == '__main__':
    print('---------------------------- vizDoom training script ---------------------------')
    print('scenario: {}'.format(hp.scenario))

    print('training parameters:')
    print('n_epoch: {}, steps_per_epoch: {}, play_steps: {}'.format(hp.n_epoch, hp.steps_per_epoch, hp.play_steps))
    print('batch_size: {}, time_size: {}, tests_per_epoch: {}'.format(hp.batch_size, hp.time_size,
                                                                      hp.tests_per_epoch))

    train_env = DoomEnvironment(hp.config_file, False, hp.train_skiprate)
    test_env = DoomEnvironment(hp.config_file, False, hp.test_skiprate)
    er = ReplayMemory(hp.replay_size, hp.screen_size)

    policy_net = DQN(hp.scenario, 2 ** train_env.get_n_buttons())
    target_net = DQN(hp.scenario, 2 ** train_env.get_n_buttons())
    optimizer = torch.optim.RMSprop(policy_net.parameters(), hp.learning_rate)

    trainer = Trainer(scenario=hp.scenario, cuda=hp.cuda,
                      environment=train_env, test_environment=test_env,
                      experience_replay=er,
                      policy_net=policy_net, target_net=target_net, optimizer=optimizer,
                      log_folder=hp.log_folder)

    print('----------------------------------- Training -----------------------------------')
    trainer.train(n_epoch=hp.n_epoch, steps_per_epoch=hp.steps_per_epoch,
                  play_steps=hp.play_steps, batch_size=hp.batch_size, time_size=hp.time_size,
                  tests_per_epoch=hp.tests_per_epoch,
                  start_epsilon=hp.start_epsilon, end_epsilon=hp.end_epsilon)
    print('-------------------------------- Training done ---------------------------------')
    print('Exit')
