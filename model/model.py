import torch


class BehavioralCloning(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(60, 60)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(60, 60)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(60, 4)  # out_features: move_right percentage, move_left_percentage, jump percentage, move_right_jump percentage
        self.softmax = torch.nn.Softmax() # TODO investigate: Is softmax necessary in my model?

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x


# TODO create Q-Learning model
class QLearningModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
