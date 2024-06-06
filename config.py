from enum import Enum

RANDOM_SEED = 42
ENV_SIZE = 60
OBSTACLE_WIDTH = 5
OBSTACLE_RANGE_START = 20
OBSTACLE_RANGE_END = 45
OBSTACLE_RANGE_HEIGHT_START = 1
OBSTACLE_RANGE_HEIGHT_END = 10
FLOOR_HEIGHT_RANGE_START = 1
FLOOR_HEIGHT_RANGE_END = 10
WHITE = 255
AGENT_START_POS = 0
AGENT_END_POS = 59
AGENT = 130
MAX_JUMP_HEIGHT = 20

BATCH_SIZE = 1
PARAMS = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 6}
MAX_EPOCHS = 100
MAX_STEPS = 100  # Maximum steps per episode
EPS_DECAY = 0.99  # Decay rate for epsilon in epsilon-greedy strategy
MIN_EPSILON = 0.01  # Minimum value for epsilon after decay
GAMMA = 0.99  # Discount factor for future rewards
# RUN_RIGHT = 1
# RUN_LEFT = 2
# JUMP = 3
# JUMP_RIGHT = 4


class Actions(Enum):
    RUN_RIGHT = 0
    RUN_LEFT = 1
    JUMP = 2
    JUMP_RIGHT = 3
