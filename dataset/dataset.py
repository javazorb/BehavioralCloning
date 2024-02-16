import numpy as np
import constants
import math


def train_test_val_split(environments):
    """ Splits the environments into i.i.d training, validation and test sets.
    Utilizes the generate_synth_state_actions before splitting

    :param environments:
    :return: tuple (train, validation, test)
    """
    pass  # TODO


def generate_synth_state_actions(environments):
    """Generates synthetic state actions pairs for each environment based on a optimal calculated path,
    for each indivual environment

    :param environments: array of np.uint8 arrays
    :rtype: list of state action pairs
    :returns: a list of list with state action pairs e.g. environment with agent on pos 5 as state and move-right as action
    """
    envs_state_action = []
    for env in environments:
        env_pairs = []
        for pos in range(len(env)):
            state_action_pair = (env[pos], "ACTION")
            env_pairs.append(state_action_pair)
        envs_state_action.append(env_pairs)

    pass  # TODO


def get_env_floor_height(environment):
    for index, row in enumerate(environment):
        if constants.WHITE in row:
            if len(set(row)) == 1:
                return index


def get_obst_positions(environment, floor_height):
    obst_positions = np.ravel(np.where(environment[floor_height + 1] == constants.WHITE)) # ravel converts the 1,d arr to 1d
    return obst_positions[0], obst_positions[-1]


def calculate_optimal_trajectory(environment):
    """
    Calculates the optimal trajectory to be used for the given environment
    :param environment:
    :return: either a list of the actions or an array with the agent at every optimal position
    """
    obst_middle = math.ceil(constants.OBSTACLE_WIDTH / 2)
    floor_height = get_env_floor_height(environment)
    obst_start_pos, obst_end_pos = get_obst_positions(environment, floor_height)
    for index, row in enumerate(environment):
        pass  # TODO
