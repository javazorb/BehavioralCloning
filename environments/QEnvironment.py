import numpy as np
import config
import dataset.dataset as dataset


class QEnvironment:
    def __init__(self, size=config.ENV_SIZE, environment=None):
        if environment is not None:
            environment = np.squeeze(environment)
        self.environment = environment
        self.size = size
        self.goal_position = (dataset.get_env_floor_height(self.environment) + 1, config.ENV_SIZE - 1)
        self.start_position = (dataset.get_env_floor_height(self.environment) + 1, 0)
        self.current_position = self.start_position
        self.state = np.zeros((1, self.size, self.size), dtype=np.float32)
        self.done = False

    def reset(self):
        self.state.fill(0)
        self.start_position = (dataset.get_env_floor_height(self.environment) + 1, 0)
        self.current_position = self.start_position
        self.done = False
        self.goal_position = (dataset.get_env_floor_height(self.environment), self.size - 1)
        self.state[0, dataset.get_env_floor_height(self.environment), :] = self.environment[
                                                                           dataset.get_env_floor_height(
                                                                            self.environment), :]
        self.state[0, self.goal_position[0], self.goal_position[1]] = 2  # Mark goal
        self.state[0, self.start_position[0], self.start_position[1]] = 1  # Mark start
        return self.state

    def step(self, action):
        if self.current_position == self.goal_position:
            reward = 100
            self.done = True
        else:
            reward = 0
        x, y = self.current_position
        # TODO perform action and change current pos with changed coordinates and check if collision with obstacle appears and set reward accordingly
        self.state.fill(0)
        self.state[0, x, y] = 1
        next_state = self.state
        return next_state, reward, self.done
