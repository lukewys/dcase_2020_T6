import glob
import numpy as np
import os
from torch.utils.data import Dataset


class ClothoTestset(Dataset):
    def __init__(self, data_dir):
        super(ClothoTestset, self).__init__()
        self.data_dir = data_dir
        self.data = glob.glob(f'{data_dir}/*.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):  # return: mel, filename (with out extension)
        return np.load(self.data[item]), os.path.splitext(os.path.basename(self.data[item]))[0]