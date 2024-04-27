import torch
import config


class BehavioralCloning(torch.nn.Module): # TODO going on with CNN

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(config.ENV_SIZE * config.ENV_SIZE, 256)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256, 128)
        self.activation2 = torch.nn.ReLU() # TODO out feature 1 is it good or should i get percentage of which of all action most plausible
        self.linear3 = torch.nn.Linear(128, len(config.Actions))  # out_features: move_right percentage, move_left_percentage, jump percentage, move_right_jump percentage

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

    def forward(self, x):
        pass
