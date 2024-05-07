import numpy as np
import torch
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


# TODO 1: add train, test loss functions for behavioral cloning + behavioral cloning with rewards + Q-Learning
# TODO 2: add function where validation set is used for hyperparameter tuning
# TODO 3: add a reward system for policy tuning in the loss and training function for positive emphasis on correct
#   behaviour


def save_model(model, name, path=os.getcwd() + os.sep + 'model' + os.sep):
    path_model = os.path.join(path, f'{name}_model.pt')
    path_state_dict = os.path.join(path, f'{name}_state_dict.pt')
    torch.save(model.state_dict(), path_state_dict)
    torch.save(model, path_model)


def loss(model, val_loader, device, criterion):  # loss for normal NN Model use Cross Entropy loss if I remember correct
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for environments, actions in val_loader:
            environments = environments.to(device, dtype=torch.float32)
            actions = actions.to(device, dtype=torch.long)

            outputs = model(environments)
            outputs = outputs.view(-1, 4)
            actions = actions.view(-1)
            curr_loss = criterion(outputs, actions)
            val_loss += curr_loss.item() * environments.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss


def loss_reward():  # Loss for Behavioral Cloning model with rewards combination of Cross entropy and rewards
    pass


def loss_policy():  # Maybe interchangable with loss reward?
    pass


def loss_q():  # loss for Q-Learning model
    pass


def train_model(model, train_set, val_set, criterion, optimizer):
    use_cuda = torch.cuda.is_available()
    print(f'Using cuda: {use_cuda}')
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #device = torch.device("cpu")
    model.to(device)

    train_loader = DataLoader(train_set, **config.PARAMS)
    val_loader = DataLoader(val_set, **config.PARAMS)

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss = 0.0

        for environments, actions in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS}"):
            environments = environments.to(device, dtype=torch.float32)
            actions = actions.to(device, dtype=torch.long)  # actions are indices

            optimizer.zero_grad()
            outputs = model(environments)
            #print(f"Current cuda memory allocated: {torch.cuda.memory_allocated(device=device)}")
            outputs = outputs.view(-1, 4)
            actions = actions.view(-1)
            curr_loss = criterion(outputs, actions)
            curr_loss.backward()
            optimizer.step()
            train_loss += curr_loss.item() * environments.size(0)
            #print(f"Current cuda memory cached: {torch.cuda.memory_allocated(device=device)}")
        train_loss /= len(train_loader.dataset)
        val_loss = loss(model, val_loader, device, criterion)
        if epoch % 10 == 0 and epoch != 0:
            save_model(model, f"CNN_simple_CE_{epoch}")
        print(f"\nEpoch {epoch + 1}/{config.MAX_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    save_model(model, f"CNN_simple_CE")


def train_model_linear(model, train_set, val_set, criterion, optimizer):
    use_cuda = torch.cuda.is_available()
    print(f'Using cuda: {use_cuda}')
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)

    train_loader = DataLoader(train_set, **config.PARAMS)
    val_loader = DataLoader(val_set, **config.PARAMS)

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss = 0
        for environments, actions in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS}"):
            environments = environments.to(device, dtype=torch.float32)
            #actions = actions.view(-1).long().to(device, dtype=torch.float32)
            actions = actions.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(environments)
            loss = 0
            for i in range(config.BATCH_SIZE):
                loss += criterion(outputs[i], actions[i])
            loss /= config.BATCH_SIZE
            #loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            # env = environments[0]
            # path = actions[0]
            env_loss = 0
            """
            for environment, path in zip(environments, actions):
            
                path = path.to(device)
                for i, env_slice in enumerate(environment.T): # get one column of env with pos of floor and additionally with obstacle if in slice
                    env_slice = env_slice.type(torch.FloatTensor)
                    env_slice = env_slice.to(device)
                    optimizer.zero_grad()
                    outputs = model(env_slice)
                    temp = path[i] # TODO loss = index 1 is out of bonds
                    loss = criterion(outputs, path[i]) # TODO is it really right to that for each slice or other solution for train?
                    loss.backward()
                    optimizer.step()
                    env_loss += loss.item()
                env_loss *= len(path)
            train_loss += env_loss"""

            train_loss += loss.item() * environments.size(0)
        train_loss /= len(train_loader.dataset)
        val_loss = loss(model, val_set, device, criterion)
        print(f"Epoch {epoch + 1}/{config.MAX_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def test(model, test_loader):
    pass
