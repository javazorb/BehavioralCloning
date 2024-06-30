import numpy as np
import torch
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
from environments.QEnvironment import QEnvironment


# Future TODO: add function where validation set is used for hyperparameter tuning

def save_model(model, name, path=os.getcwd() + os.sep + 'model' + os.sep):
    path_model = os.path.join(path, f'{name}_model.pt')
    path_state_dict = os.path.join(path, f'{name}_state_dict.pt')
    torch.save(model.state_dict(), path_state_dict)
    torch.save(model, path_model)
    print(f'Model saved to {path_model}')


def loss(model, val_loader, device, criterion, return_list=False):  # loss for normal NN Model use Cross Entropy loss if I remember correct
    model.eval()
    val_loss = 0.0
    losses = []
    with torch.no_grad():
        for environments, actions in val_loader:
            environments = environments.to(device, dtype=torch.float32)
            actions = actions.to(device, dtype=torch.long)

            outputs = model(environments)
            outputs = outputs.view(-1, 4)
            actions = actions.view(-1)
            curr_loss = criterion(outputs, actions)
            val_loss += curr_loss.item() * environments.size(0)
            if return_list:
                losses.append(curr_loss.item())
    val_loss /= len(val_loader.dataset)
    if return_list:
        return val_loss, losses
    return val_loss


def loss_reward():  # Loss for Behavioral Cloning model with rewards combination of Cross entropy and rewards
    pass


def loss_policy():  # Maybe interchangable with loss reward?
    pass


def loss_q(model, val_loader, device, criterion, return_list=False):
    val_loss = 0.0
    model.eval()
    model.to(device)
    env = None
    epsilon = 0.1
    losses = []
    with torch.no_grad():
        for environment, actions in val_loader:
            environment = environment.to(device, dtype=torch.float32)
            env = QEnvironment(config.ENV_SIZE, environment.detach().cpu().numpy())
            state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
            env_reward = 0
            for step in range(config.MAX_STEPS):
                action = epsilon_greedy_action(model, state, epsilon)
                next_state, reward, done = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
                with torch.no_grad():
                    target_Q = reward + config.GAMMA * (torch.max(model(next_state)))
                Q_values = model(state)
                target_Q_expanded = target_Q.expand_as(Q_values)
                epoch_loss = criterion(Q_values, target_Q_expanded)
                #epoch_loss.backward()
                state = next_state
                env_reward += reward
                if done:
                    break
            if return_list:
                losses.append(epoch_loss.item())
            val_loss += epoch_loss.item() * state.size(0)
            epsilon = max(epsilon * config.EPS_DECAY, config.MIN_EPSILON)
    if return_list:
        return val_loss / len(val_loader.dataset), losses
    return val_loss / len(val_loader.dataset)


def train_model(model, train_set, val_set, criterion, optimizer):
    device = get_device()
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
            outputs = outputs.view(-1, 4)
            actions = actions.view(-1)
            curr_loss = criterion(outputs, actions)
            curr_loss.backward()
            optimizer.step()
            train_loss += curr_loss.item() * environments.size(0)
        train_loss /= len(train_loader.dataset)
        val_loss = loss(model, val_loader, device, criterion)
        if epoch % 10 == 0 and epoch != 0:
            save_model(model, f"CNN_simple_CE_{epoch}")
        print(f"\nEpoch {epoch + 1}/{config.MAX_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    save_model(model, f"CNN_simple_CE")


def epsilon_greedy_action(model, state, epsilon=0.1):
    if random.random() < epsilon:
        # Explore: choose a random action
        return torch.randint(0, len(config.Actions), (1,), dtype=torch.long)
    else:
        # Exploit: choose action with highest Q-value
        with torch.no_grad():
            Q_values = model(state)
            action = torch.argmax(Q_values, dim=1)
            return action


def train_q_model(model, train_set, val_set, criterion, optimizer, epsilon=0.1):
    device = get_device()
    model.to(device)
    env = None
    train_loader = DataLoader(train_set, **config.PARAMS)
    val_loader = DataLoader(val_set, **config.PARAMS)
    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for environment, actions in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS}"):
            environment = environment.to(device, dtype=torch.float32)
            env = QEnvironment(config.ENV_SIZE, environment.detach().cpu().numpy())
            actions = actions.to(device, dtype=torch.long)
            state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
            env_reward = 0
            for step in range(config.MAX_STEPS):
                action = epsilon_greedy_action(model, state, epsilon)
                next_state, reward, done = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
                with torch.no_grad():
                    next_max_Q = torch.max(model(next_state))
                    target_Q = reward + config.GAMMA * (torch.max(model(next_state)))
                Q_values = model(state)
                target_Q_expanded = target_Q.expand_as(Q_values)
                epoch_loss = criterion(Q_values, target_Q_expanded)
                optimizer.zero_grad()
                epoch_loss.backward()
                optimizer.step()
                state = next_state
                env_reward += reward
                if done:
                    break
            train_loss += epoch_loss.item() * state.size(0)
            epsilon = max(epsilon * config.EPS_DECAY, config.MIN_EPSILON)
        train_loss /= len(train_loader.dataset)
        #val_loss = loss_q(model, val_loader, device, criterion)

        if epoch % 10 == 0 and epoch != 0:
            save_model(model, f"CNN_Q_Model_{epoch}")
        print(f"\nEpoch {epoch + 1}/{config.MAX_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {train_loss:.4f}")
    save_model(model, f"CNN_Q_Model")


def get_device():
    use_cuda = torch.cuda.is_available()
    print(f'Using cuda: {use_cuda}')
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if torch.backends.mps.is_available():
        print(f'Using mps: {torch.backends.mps.is_available()}')
        device = torch.device("mps")
    return device


def train_model_linear(model, train_set, val_set, criterion, optimizer):
    device = get_device()
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
