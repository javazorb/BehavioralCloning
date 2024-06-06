import config
import dataset.generate_environment as generate_data
import dataset.dataset as data
from tqdm import tqdm
import os
import json
import training.train as train
from model.model import BehavioralModelCNN, QLearningModel
from dataset.dataloader import EnvironmentDataset
import torch.nn as nn
import torch.optim as optim
import environments.QEnvironment as QEnvironment


def run():
    # generate_data.generate_and_save_environments(num_environments=1000)
    # training_data, validation_data, testing_data = data.train_test_val_split(generate_data.load_environments())
    envs = generate_data.load_environments()
    # save_optimal_paths(envs)
    optimal_paths = load_optimal_paths()
    train_data, test_data, val_data = data.train_test_val_split(envs, optimal_paths)
    #data.save_dataset(train_data, 'train_data')
    #data.save_dataset(test_data, 'test_data')
    #data.save_dataset(val_data, 'val_data')
    train_data = data.load_dataset('train_data', 'dataset')
    test_data = data.load_dataset('test_data', 'dataset')
    val_data = data.load_dataset('val_data', 'dataset')
    train_set = EnvironmentDataset(train_data)
    val_set = EnvironmentDataset(val_data)
    test_set = EnvironmentDataset(test_data)

    # model = BehavioralModelCNN()
    # train.train_model(model,  train_set, val_set, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001))
    q_model = QLearningModel()
    train.train_q_model(q_model, train_set, val_set, nn.MSELoss(), optim.Adam(q_model.parameters(), lr=0.001), epsilon=0.1)


def save_optimal_paths(envs):
    agent_positions_all_envs = []
    for index, env in tqdm(enumerate(envs), total=len(envs), desc="calculating optimal paths", unit="Environments"):
        _, agent_positions = data.calculate_optimal_trajectory(env, index)
        agent_positions_all_envs.append((index, sorted(list(set(agent_positions)), key=lambda x: x[1])))
    with open('dataset' + os.sep + 'optimal_paths.json', 'w') as file:
        json.dump(agent_positions_all_envs, file, indent=2)


def load_optimal_paths():
    with open('dataset' + os.sep + 'optimal_paths.json', 'r') as file:
        data = json.load(file)
    _, paths = map(list, zip(*data))
    return paths


if __name__ == '__main__':
    run()
