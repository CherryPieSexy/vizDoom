from models.dqn import DQN
from models.drqn import DRQN
from doom_environment import DoomEnvironment
import numpy as np
# TODO: agent should to be trained with weaker version of bots, i.e. with 20hp


# TODO: develop a proper name to the agent
class Ogent:
    def __init__(self, n_actions, epsilon=1.0):
        # TODO: n_actions is NOT too big: only 11, but dqn and drqn has different values
        # TODO: attack, move left, move right, aml, amr, tl, tr, f, b, af, ab
        # TODO: hard-code weapon change strategy. Fuck!
        self._dqn = DQN('deathmatch', n_actions, epsilon)  # TODO: shouldn't be 2 ** n_actions
        self._drqn = DRQN('deathmatch', n_actions, epsilon)

    def forward(self, x_screens, hidden):
        # TODO: add _detection_ layer to the DRQN model
        q_values, hidden = self._drqn(x_screens, hidden)
        detection = np.random.rand()
        # TODO: Arnold paper propose to use dqn ____during evaluation___
        # TODO: if there are no detected enemies
        # TODO: or agent does not have any ammo left
        if detection > 0.0:  # without sigmoid
            q_values = self._dqn(x_screens)
        return detection, hidden, q_values

    def sample_actions(self):
        # pretty same as in drqn
        # may be I can even call .sample_actions()
        pass


def agent():
    return 0


if __name__ == '__main__':
    cfg = 'scenarios/deathmatch_shotgun.cfg'
    doom = DoomEnvironment(cfg, True, 4)
    while True:
        doom.step(agent())
