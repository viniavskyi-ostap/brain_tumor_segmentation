import os
import numpy as np
import torch


class BratsDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.x_path = os.path.join(path, "X")
        self.y_path = os.path.join(path, "y")

        self.length = len(os.listdir(self.x_path))

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        X = np.load(os.path.join(self.x_path, f"{i}.npy"))
        y = np.load(os.path.join(self.y_path, f"{i}.npy"))

        return torch.FloatTensor(X), torch.FloatTensor(y)
