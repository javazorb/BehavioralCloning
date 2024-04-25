from torch.utils.data import Dataset, DataLoader


class EnvironmentDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data  # tuple list length 2 env and opt_path
        self.transform = transform

    def __len__(self):
        return len(self.data[0][0])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]
