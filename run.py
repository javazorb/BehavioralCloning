import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
import glob
import numpy as np
import environments.QEnvironment as QEnvironment


# TODO add setup folder structure
def print_losses_pic(bc_losses_list, model_files, batchsize=10):
    cnt = 0
    plt.figure(figsize=(10, 6))
    print(len(bc_losses_list[0]))
    print(bc_losses_list[0])
    bc_losses_list = bc_losses_list[-2:]
    model_files = model_files[-2:]
    for model_file in model_files:
        values = range(1, len(bc_losses_list[cnt]) + 1)
        plt.plot(values, bc_losses_list[cnt], label=f'Test Loss for {os.path.basename(model_file)}', marker='.')
        cnt += 1

    plt.xlabel(f'Envs in batch_size {batchsize}')
    plt.ylabel('Loss')
    plt.title('Test loss')
    # plt.xlim(left=0.5)
    # plt.xlim(right=101)
    plt.legend()
    plt.grid(False)
    plt.show()


def run():
    # generate_data.generate_and_save_environments(num_environments=1000)
    # training_data, validation_data, testing_data = data.train_test_val_split(generate_data.load_environments())
    envs = generate_data.load_environments()
    # save_optimal_paths(envs)
    optimal_paths = load_optimal_paths()
    train_data, test_data, val_data = data.train_test_val_split(envs, optimal_paths)
    # data.save_dataset(train_data, 'train_data')
    # data.save_dataset(test_data, 'test_data')
    # data.save_dataset(val_data, 'val_data')
    train_data = data.load_dataset('train_data', 'dataset')
    test_data = data.load_dataset('test_data', 'dataset')
    val_data = data.load_dataset('val_data', 'dataset')
    train_set = EnvironmentDataset(train_data)
    val_set = EnvironmentDataset(val_data)
    test_set = EnvironmentDataset(test_data)

    # model = BehavioralModelCNN()
    # train.train_model(model,  train_set, val_set, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001))
    # q_model = QLearningModel()
    # train.train_q_model(q_model, train_set, val_set, nn.MSELoss(), optim.Adam(q_model.parameters(), lr=0.001), epsilon=0.1)

    get_test_loss(test_set)
    get_test_loss(test_set, True)


def get_test_loss(test_set, test_qmodel=False):
    bc_model_folder = 'model' + os.sep + 'CNN'
    qc_model_folder = 'model' + os.sep + 'QModel'
    if test_qmodel:
        model_files = glob.glob(os.path.join(qc_model_folder, '*.pt'))
    else:
        model_files = glob.glob(os.path.join(bc_model_folder, '*.pt'))
    test_loader = DataLoader(test_set, **config.PARAMS)
    device = train.get_device()
    cnt = 1
    min_loss_model = ''
    old_loss = 1
    bc_losses_list = []
    for model_file in model_files:
        model = torch.load(model_file)
        print(f'testing {cnt} model')
        if test_qmodel:
            config.BATCH_SIZE = 1
            test_loss, losses, rmse = train.loss_q(model, test_loader, device, nn.MSELoss(), True)
        else:
            config.BATCH_SIZE = 10
            test_loss, losses, accuracies = train.loss(model, test_loader, device, nn.CrossEntropyLoss(), True)
        bc_losses_list.append(losses)
        if old_loss > test_loss:
            min_loss_model = os.path.basename(model_file)
        print(f'accuracies for individual 10 env batch: {accuracies}')
        print(f'mean accuray: {np.mean(accuracies)}')
        print(f'Model: {os.path.basename(model_file)}, Test Loss: {test_loss}')
        cnt += 1
    print_losses_pic(bc_losses_list, model_files)
    print(f"Best Model is: {min_loss_model}")


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
