from vizdoom import DoomGame, Mode
import itertools as it


class DoomEnvironment:
    def __init__(self, config, visible, skiprate):
        self._game = DoomGame()
        self._game.load_config(config)
        self._game.set_window_visible(visible)
        self._game.set_mode(Mode.PLAYER)
        self._game.init()

        n_actions = self._game.get_available_buttons_size()
        self._actions = [list(a) for a in it.product([0, 1], repeat=n_actions)]
        self._skiprate = skiprate

    def observe(self):
        observation = self._game.get_state()
        screen = observation.screen_buffer
        game_variables = observation.game_variables
        return screen, game_variables

    def step(self, action_id):
        """Takes single action, performs it and observe the result

        if session is just ended will return zeros as screen and in-game features
        :param action_id: index of action
        :return: screen, game_variables, reward, is_done
        """
        reward = self._game.make_action(self._actions[action_id], self._skiprate)
        is_done = self._game.is_episode_finished()
        return reward, is_done

    def reset(self):
        self._game.new_episode()

    def get_episode_reward(self):
        return self._game.get_total_reward()
