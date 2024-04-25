import torch
import config
from torch.utils.data import DataLoader
from tqdm import tqdm


# TODO 1: add train, test loss functions for behavioral cloning + behavioral cloning with rewards + Q-Learning
# TODO 2: add function where validation set is used for hyperparameter tuning
# TODO 3: add a reward system for policy tuning in the loss and training function for positive emphasis on correct
#   behaviour


def loss(model, val_loader, device, criterion):  # loss for normal NN Model use Cross Entropy loss if I remember correct
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for environment, actions in val_loader:
            environment = environment.to(device)
            actions = actions.to(device)

            outputs = model(environment)
            loss = criterion(outputs, actions)
            val_loss += loss.item() * environment.size(0)

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
    model.to(device)

    train_loader = DataLoader(train_set, **config.PARAMS)
    val_loader = DataLoader(val_set, **config.PARAMS)

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        train_loss = 0
        for environment, actions in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS}"):
            environment = environment.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            outputs = model(environment)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * environment.size(0)
        train_loss /= len(train_loader.dataset)
        val_loss = loss(model, val_set, device, criterion)
        print(f"Epoch {epoch + 1}/{config.MAX_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def test(model, test_loader):
    pass
