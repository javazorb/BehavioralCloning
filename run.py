import dataset.generate_environment as generate_data
import dataset.dataset as data
from tqdm import tqdm
import os
import json


def run():
    #generate_data.generate_and_save_environments(num_environments=1000)
    #training_data, validation_data, testing_data = data.train_test_val_split(generate_data.load_environments())
    envs = generate_data.load_environments()
    save_optimal_paths(envs)


def save_optimal_paths(envs):
    agent_positions_all_envs = []
    for index, env in tqdm(enumerate(envs), total=len(envs), desc="calculating optimal paths", unit="Environments"):
        _, agent_positions = data.calculate_optimal_trajectory(env, index)
        agent_positions_all_envs.append((index, sorted(list(set(agent_positions)), key=lambda x: x[1])))
    with open('dataset' + os.sep + 'optimal_paths.json', 'w') as file:
        json.dump(agent_positions_all_envs, file, indent=2)


if __name__ == '__main__':
    run()
