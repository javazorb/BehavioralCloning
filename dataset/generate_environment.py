import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def generate_environment(size=(60, 60), obstacle_range=(20, 45), floor_height_range=(1, 10), obstacle_width=5,
                         obstacle_height_range=(1, 10)):
    environment = np.zeros(size, dtype=np.uint8)
    floor_height = np.random.randint(*floor_height_range + (1,))
    environment[floor_height, :] = 255

    obstacle_top = floor_height + 1  # Assuming obstacle is one unit higher than the floor
    obstacle_bottom = obstacle_top + np.random.randint(*obstacle_height_range + (1,))
    obstacle_left = np.random.randint(*obstacle_range + (1,))
    obstacle_right = obstacle_left + obstacle_width

    i_top, i_bottom, i_left, i_right = map(int, [obstacle_top, obstacle_bottom, obstacle_left, obstacle_right])
    environment[i_top:i_bottom, i_left:i_right] = 255  # Use 255 for white

    return environment


def visualize_and_save_env(environment, save_path):
    plt.imshow(environment, cmap='gray', origin='lower', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_and_save_environments(num_environments=100, save_directory='dataset/data', visualize=True):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    for i in tqdm(range(num_environments), desc='Generating Environments', unit='Environment'):
        environment = generate_environment()
        save_path = os.path.join(save_directory, f'environment{i}.npy')
        np.save(save_path, environment)
        if visualize:
            visualize_and_save_env(environment, save_path=os.path.join('dataset/images', f'environment{i}.png'))


def load_environments(directory='dataset/data'):
    environments = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            environment = np.load(os.path.join(directory, filename))
            environments.append(environment)
    return environments
