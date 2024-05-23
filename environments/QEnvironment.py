import numpy as np
import config
import dataset.dataset as dataset


class QEnvironment: # TODO finish implementation
    def __init__(self, size=config.ENV_SIZE, environment=None):
        self.environment = environment
        self.size = size
        self.goal_position = self.environment[dataset.get_env_floor_height(self.environment), config.ENV_SIZE - 1]
        self.start_position = self.environment[dataset.get_env_floor_height(self.environment), 0]
        self.done = False

    def reset(self):
        pass

    def step(self, action):
        pass
