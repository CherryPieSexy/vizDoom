from .dqn import DQN
from .drqn import DRQN
from .combined_agent import CombinedAgent

agent = {
    'DQN': DQN,
    'DRQN': DRQN,
    'combined': CombinedAgent
}
