from torch.utils.data import Dataset, DataLoader
import numpy as np


class EnvironmentDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # tuple list length 2 env and opt_path
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        env = self.data[0][idx]
        path = np.array(self.data[1][idx])
        return (env, path)
