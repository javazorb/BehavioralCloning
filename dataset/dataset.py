import numpy as np
import constants
import math
import matplotlib.pyplot as plt
import os


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


def get_obstacle_height(environment, obst_start_pos):
    height = 0
    for index, row in enumerate(environment):
        if constants.WHITE == row[obst_start_pos]:
            height += 1
    return height


def calculate_optimal_trajectory(environment, env_index):
    """
    Calculates the optimal trajectory to be used for the given environment
    :param environment:
    :return: either a list of the actions or an array with the agent at every optimal position
    """
    obst_middle = math.ceil(constants.OBSTACLE_WIDTH / 2)
    floor_height = get_env_floor_height(environment)
    obst_start_pos, obst_end_pos = get_obst_positions(environment, floor_height)
    current_agent_height = floor_height # important for jumping
    traverse = 0
    previous_pos = 0
    jump_start = obst_start_pos - get_obstacle_height(environment, obst_start_pos)
    """for row in environment[floor_height + 1:]:
        for i in range(row):
            if i + get_obstacle_height(environment, obst_start_pos) < jump_start:
                row[i] = constants.AGENT
                previous_pos = i"""
    agent_positions = []  # To store agent positions for visualization or further processing
    reached_floor = False
    # Traverse environment rows
    for row_index, row in enumerate(environment[floor_height + 1:], start=floor_height + 1):
        if row_index == floor_height + 1:
            if previous_pos == 0:
                for i in range(len(row)):
                    if i < jump_start:
                        agent_positions.append((row_index, i))
                        row[i] = constants.AGENT
                    else:
                        previous_pos = i
                        break
                for i in range(obst_end_pos + get_obstacle_height(environment, obst_start_pos), len(row)):
                    agent_positions.append((row_index, i))  # TODO check if array is correct
                    row[i] = constants.AGENT
        if previous_pos != 0 and previous_pos <= obst_start_pos + obst_middle:
            # Move 1 step right and 1 step up until middle of the obstacle
            start_pos = previous_pos
            inital_height = row_index
            for col_index in range(start_pos, len(row)):
                if col_index < obst_start_pos + obst_middle - 1:
                    agent_positions.append((inital_height, col_index))
                    environment[inital_height, col_index] = constants.AGENT
                    previous_pos += 1
                    inital_height += 1
                else:
                    break  # Stop when reaching the middle of the obstacle
            # Decrease height and move 1 step right until back at floor height
            current_agent_height = inital_height
            start_pos = previous_pos
            for col_index in range(start_pos, len(row)):
                if current_agent_height > floor_height and not reached_floor:
                    agent_positions.append((current_agent_height, col_index))
                    environment[current_agent_height, col_index] = constants.AGENT
                    current_agent_height -= 1
                else:
                    reached_floor = True
                    break  # Stop when back at floor height

    plt.imshow(environment, cmap='gray', origin='lower', vmin=0, vmax=255)
    plt.axis('off')
    #plt.show()
    plt.savefig(os.path.join('dataset/images/optimal_paths', f'environment{env_index}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    return environment, agent_positions

