import numpy as np
import config
import dataset.dataset as dataset


class QEnvironment:
    def __init__(self, size=config.ENV_SIZE, environment=None):
        if environment is not None:
            environment = np.squeeze(environment)
        self.environment = environment
        self.size = size
        floor_height = dataset.get_env_floor_height(self.environment)
        self.goal_position = (floor_height + 1, config.ENV_SIZE - 1)
        self.start_position = (floor_height + 1, 0)
        self.current_position = self.start_position
        self.state = np.zeros((1, self.size, self.size), dtype=np.float32)
        self.done = False

    def reset(self):
        self.state.fill(0)
        floor_height = dataset.get_env_floor_height(self.environment)
        self.start_position = (floor_height + 1, 0)
        self.current_position = self.start_position
        self.done = False
        self.goal_position = (floor_height + 1, self.size - 1)
        self.state[0, floor_height, :] = self.environment[floor_height, :]
        self.state[0, self.goal_position[0], self.goal_position[1]] = 2  # Mark goal
        self.state[0, self.start_position[0], self.start_position[1]] = 1  # Mark start
        return self.state

    def step(self, action):
        if self.current_position == self.goal_position:
            reward = 100
            self.done = True
            return self.current_position, reward, self.done
        x, y = self.current_position
        if action == 0:  # RUN_RIGHT
            new_position = (x, min(y + 1, self.size - 1))
        elif action == 1:  # RUN_LEFT
            new_position = (x, max(y - 1, 0))
        elif action == 2:  # JUMP
            new_position = (min(x + 1, self.size - 1), y)
        elif action == 3:  # JUMP_RIGHT
            new_position = (min(x + 1, self.size - 1), min(y + 1, self.size - 1))

        new_x, new_y = new_position
        # Check if the agent is above the floor and if the next action is not jumping
        if x > dataset.get_env_floor_height(self.environment) + 1 and action not in [2, 3]:
            new_x = min(x - 1, dataset.get_env_floor_height(self.environment) + 1)  # Move down to simulate gravity
            new_position = (new_x, new_y)

        if self.environment[new_x, new_y] == 255:
            reward = -1  # Hit obstacle
            next_position = self.current_position
        elif new_position == self.goal_position:
            reward = 100
            next_position = new_position
            self.done = True
        else:
            reward = 0
            next_position = new_position

        self.current_position = next_position
        self.state.fill(0)
        self.state[0, dataset.get_env_floor_height(self.environment), :] = self.environment[dataset.get_env_floor_height(self.environment), :]
        self.state[0, self.goal_position[0], self.goal_position[1]] = 2  # Mark goal
        self.state[0, self.start_position[0], self.start_position[1]] = 1  # Mark start
        next_state = self.state
        return next_state, reward, self.done
