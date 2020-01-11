import os
import numpy as np
import torch


class BratsDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.x_path = os.path.join(path, "X")
        self.y_path = os.path.join(path, "y")

        self.length = len(os.listdir(self.x_path))

    def __len__(self):
        # return self.length
        return 100

    def __getitem__(self, i):
        X = np.load(os.path.join(self.x_path, f"{i}.npy")).astype(np.float32)
        y = np.load(os.path.join(self.y_path, f"{i}.npy"))
        y[y == 4] = 3

        return torch.from_numpy(X), torch.from_numpy(y).long()

