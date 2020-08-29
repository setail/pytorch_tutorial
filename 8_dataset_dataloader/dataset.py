import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self):
        txt_data = np.loadtxt('./sample_data.txt', delimiter=',')
        self._x = torch.from_numpy(txt_data[:, :2])
        self._y = torch.from_numpy(txt_data[:, 2])
        self._len = len(txt_data)

    def __getitem__(self, item):
        return self._x[item], self._y[item]

    def __len__(self):
        return self._len


data = MyDataset()
print(len(data))

first = next(iter(data))
print(first)

print(type(first[0]))