import torchvision.transforms as transforms
import numpy as np


def reward_shaping_basic(reward, prev_obs, next_obs):
    # no observations needed here
    return reward


def reward_shaping_dtc(reward, prev_obs, next_obs):
    # observation = [ammo, health]
    heals_ammo_decrease = next_obs - prev_obs < 0
    penalty = np.dot(heals_ammo_decrease.astype(int), [-0.1, -0.1])
    return reward + penalty


def reward_shaping_dcr(reward, prev_obs, next_obs):
    # observation = [ammo, health, kill count]
    heals_ammo_decrease = next_obs[:2] - prev_obs[:2] < 0
    kill_reward = int(next_obs[2] - prev_obs[2] > 0) * 1.0
    penalty = np.dot(heals_ammo_decrease.astype(int), [-0.1, -0.1])
    print(penalty, kill_reward)
    return reward - penalty + kill_reward


def reward_shaping_hg(reward, prev_obs, next_obs):
    pass  # TODO


reward_shaping = {
    'basic': reward_shaping_basic,
    'defend_the_center': reward_shaping_dtc,
    'deadly_corridor': reward_shaping_dcr,
    'health_gathering': reward_shaping_hg
}

basic_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((30, 45)),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((60, 108)),
    transforms.ToTensor(),
])


def screen_transform(scenario, screen):
    if scenario == 'basic':
        return basic_transform(screen)
    return transform(screen)


def watch_agent(scenario, agent, env):
    reward = 0.0
    while True:
        screen, features = env.observe()
        action = agent.sample_actions(screen_transform(scenario, screen))[0]
        r, done = env.advance_action_step(action)
        if not done:
            _, new_features = env.observe()
            reward += reward_shaping[scenario](r, features, new_features)
        else:
            return reward + r
