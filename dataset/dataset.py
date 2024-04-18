import numpy as np
import config
import math
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from config import Actions
import pickle


def save_dataset(dataset, name):
    with open('dataset' + os.sep + name + '.pkl', 'wb') as file:
        pickle.dump(dataset, file)


def load_dataset(name, folder):
    with open(folder + os.sep + name + '.pkl', 'rb') as file:
        return pickle.load(file)


def train_test_val_split(environments, optimal_paths):
    """ Splits the environments into i.i.d training, validation and test sets.
    Utilizes the generate_synth_state_actions before splitting

    :param environments:
    :param optimal_paths:
    :return: tuple (train, validation, test)
    """

    envs_actions_list = generate_synth_state_actions(environments, optimal_paths)
    X, y = map(list, zip(*envs_actions_list))
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.25, random_state=config.RANDOM_SEED)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def convert_path_to_actions(env, path):
    actions = []
    for index, coord in enumerate(path):
        if index + 1 < len(path) and path[index + 1][0] > coord[0]:
            actions.append(Actions.JUMP_RIGHT)
        else:
            actions.append(Actions.RUN_RIGHT)
    return actions


def generate_synth_state_actions(environments, optimal_paths):
    """Generates synthetic state actions pairs for each environment based on a optimal calculated path,
    for each indivual environment

    :param environments: array of np.uint8 arrays
    :param optimal_paths of the environment
    :rtype: list of state action pairs
    :returns: a list of list containing the environments and the action corresponding to the optimal path
    """
    envs_state_action = []
    for index, env in enumerate(environments):
        env_actions = convert_path_to_actions(env, optimal_paths[index])
        envs_state_action.append((env, env_actions))

    return envs_state_action


def get_env_floor_height(environment):
    for index, row in enumerate(environment):
        if config.WHITE in row:
            if len(set(row)) == 1:
                return index


def get_obst_positions(environment, floor_height):
    obst_positions = np.ravel(np.where(environment[floor_height + 1] == config.WHITE)) # ravel converts the 1,d arr to 1d
    return obst_positions[0], obst_positions[-1]


def get_obstacle_height(environment, obst_start_pos):
    height = 0
    for index, row in enumerate(environment):
        if config.WHITE == row[obst_start_pos]:
            height += 1
    return height


def calculate_optimal_trajectory(environment, env_index):
    """
    Calculates the optimal trajectory to be used for the given environment and its id
    :param env_index:
    :param environment:
    :return: either a list of the actions or an array with the agent at every optimal position
    """
    obst_middle = math.ceil(config.OBSTACLE_WIDTH / 2)
    floor_height = get_env_floor_height(environment)
    obst_start_pos, obst_end_pos = get_obst_positions(environment, floor_height)
    previous_pos = 0
    jump_start = obst_start_pos - get_obstacle_height(environment, obst_start_pos)

    agent_positions = []  # To store agent positions for visualization or further processing
    reached_floor = False
    # Traverse environment rows
    for row_index, row in enumerate(environment[floor_height + 1:], start=floor_height + 1):
        if row_index == floor_height + 1:
            if previous_pos == 0:
                for i in range(len(row)):
                    if i < jump_start:
                        agent_positions.append((row_index, i))
                        row[i] = config.AGENT
                    else:
                        previous_pos = i
                        break
                for i in range(obst_end_pos + get_obstacle_height(environment, obst_start_pos), len(row)):
                    agent_positions.append((row_index, i))
                    row[i] = config.AGENT
        if previous_pos != 0 and previous_pos <= obst_start_pos + obst_middle:
            # Move 1 step right and 1 step up until middle of the obstacle
            start_pos = previous_pos
            inital_height = row_index
            for col_index in range(start_pos, len(row)):
                if col_index < obst_start_pos + obst_middle - 1:
                    agent_positions.append((inital_height, col_index))
                    environment[inital_height, col_index] = config.AGENT
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
                    environment[current_agent_height, col_index] = config.AGENT
                    current_agent_height -= 1
                else:
                    reached_floor = True
                    break  # Stop when back at floor height

    plt.imshow(environment, cmap='gray', origin='lower', vmin=0, vmax=255)
    plt.axis('off')
    #plt.show()
    plt.savefig(os.path.join('dataset/images/optimal_paths', f'environment{env_index}.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    save_path = os.path.join('dataset/data/optimal_paths', f'environment{env_index}.npy')
    np.save(save_path, environment)

    return environment, agent_positions

