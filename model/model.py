import torch


class BehavioralCloning(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(60, 60)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(60, 60)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(60, 3)  # out_features: move_right percentage, jump percentage, move_right_jump percentage
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x
