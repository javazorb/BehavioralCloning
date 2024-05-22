import torch
from torch import nn

import config


class BehavioralModelCNN(nn.Module):
    def __init__(self):
        super(BehavioralModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=config.BATCH_SIZE, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=config.BATCH_SIZE, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=config.BATCH_SIZE * config.ENV_SIZE * config.ENV_SIZE, out_features=128)
        self.fc2 = nn.Linear(in_features=128,
                             out_features=240 * config.BATCH_SIZE)  # 60*4 = 240, 4 action probs per position

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, config.BATCH_SIZE * config.ENV_SIZE * config.ENV_SIZE)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 60, 4)
        return x


class BehavioralCloning(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(config.ENV_SIZE * config.ENV_SIZE, 256)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256, 128)
        self.activation2 = torch.nn.ReLU()  # TODO out feature 1 is it good or should i get percentage of which of all action most plausible
        self.linear3 = torch.nn.Linear(128,
                                       len(config.Actions))  # out_features: move_right percentage, move_left_percentage, jump percentage, move_right_jump percentage

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x


# TODO create Q-Learning model
class QLearningModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        pass

    def policy(self):
        pass
