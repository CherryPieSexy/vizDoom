from .dqn import DQN
from .drqn import DRQN
from .d4rqn import D4RQN
from .c51m import C51M

agent = {
    'Random': DQN,
    'DQN': DQN,
    'DRQN': DRQN,
    'D4RQN': D4RQN,
    'C51M': C51M
}
