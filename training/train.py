# TODO 1: add train, test loss functions for behavioral cloning + behavioral cloning with rewards + Q-Learning
# TODO 2: add function where validation set is used for hyperparameter tuning
# TODO 3: add a reward system for policy tuning in the loss and training function for positive emphasis on correct
#   behaviour

def train(model, train_loader, val_loader, criterion, optimizer):
    pass


def test(model, test_loader):
    pass


def loss():  # loss for normal NN Model use Cross Entropy loss if I remember correct
    pass


def loss_reward():  # Loss for Behavioral Cloning model with rewards combination of Cross entropy and rewards
    pass


def loss_policy():  # Maybe interchangable with loss reward?
    pass


def loss_q():  # loss for Q-Learning model
    pass
