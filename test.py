import numpy as np
import matplotlib.pyplot as plt
import config


def create_wide_environment(size=60, floor_height=50, obstacle_width=2):
    env = np.zeros((size, size), dtype=np.uint8)
    env[floor_height:, :] = 255  # Create floor
    for x in range(0, size, 10):
        env[floor_height - 1:floor_height + obstacle_width, x:x + obstacle_width] = 255  # Create obstacles
    return env


def create_narrow_environment(size=60, floor_height=50, obstacle_width=2):
    env = np.zeros((size, size), dtype=np.uint8)
    env[floor_height:, :] = 255  # Create floor
    for x in range(5, size, 6):
        env[floor_height - 1:floor_height + obstacle_width, x:x + obstacle_width] = 255  # Create obstacles
    return env


def create_random_environment(size=60, floor_height=50, num_obstacles=10):
    env = np.zeros((size, size), dtype=np.uint8)
    env[floor_height:, :] = 255  # Create floor
    np.random.seed(config.RANDOM_SEED)  # For reproducibility
    for _ in range(num_obstacles):
        x = np.random.randint(0, size)
        y = floor_height - 1
        env[y:y + 2, x:x + 2] = 255  # Create random obstacles
    return env


def visualize_environment(env, title):
    plt.imshow(env, cmap='gray')
    plt.title(title)
    plt.show()


# Create and visualize the environments
wide_env = create_wide_environment()
narrow_env = create_narrow_environment()
random_env = create_random_environment()

visualize_environment(wide_env, "Wide Environment")
visualize_environment(narrow_env, "Narrow Environment")
visualize_environment(random_env, "Random Environment")
