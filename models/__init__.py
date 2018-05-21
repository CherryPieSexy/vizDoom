from .dqn import DQN
from .drqn import DRQN
from .ddrqn import DDRQN
from .distributional import Distributional

agent = {
    'Random': DQN,
    'DQN': DQN,
    'DRQN': DRQN,
    'DDRQN': DDRQN,
    'Distributional': Distributional
}
