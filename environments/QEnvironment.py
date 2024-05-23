import numpy as np
import config
import dataset.dataset as dataset


class QEnvironment:
    def __init__(self, size=config.ENV_SIZE, environment=None):
        self.environment = environment
        self.size = size
        self.goal_position = self.environment[dataset.get_env_floor_height(self.environment), config.ENV_SIZE - 1]
        self.start_position = self.environment[dataset.get_env_floor_height(self.environment), 0]
        self.current_position = self.start_position
        self.state = np.zeros((1, self.size, self.size), dtype=np.float32)
        self.done = False

    def reset(self):
        self.state.fill(0)
        self.current_position = self.start_position
        self.done = False
        self.goal_position = self.environment[dataset.get_env_floor_height(self.environment), self.size - 1]

    def step(self, action):
        if self.current_position == self.goal_position:
            reward = 100
            self.done = True
        else:
            reward = 0
        x, y = self.current_position
        # TODO perform action and change current pos with changed coordinates and check if collision appears and set reward accordingly
        self.state.fill(0)
        self.state[0, x, y] = 1
        next_state = self.state
        return next_state, reward, self.done
