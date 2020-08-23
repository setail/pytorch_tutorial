import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


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
dataloader = DataLoader(data, batch_size=3, shuffle=False, drop_last=True, num_workers=0)
n = 0
for data_val, label_val in dataloader:
    print('x:', data_val, 'y:', label_val)
    n += 1
print('iteration:', n)