import numpy as np
import torch; torch.set_default_dtype(torch.float64)
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class base_data_class():
    '''
    Base data class initializes the dataloader class.
    Each class must have train_loader, test_loader, and val_loader which should be dataloader objects
    '''
    def __init__(self):
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None


class CustomDataset(Dataset):
    '''
    Inherits from the pytorch dataset class. Has the option to add noise to the data
    '''
    def __init__(self, data, labels, noise = None):
        self.data = data
        if noise is not None:
            self.data = self.data + np.reshape(np.random.normal(noise[0], noise[1], self.data.numel()), self.data.shape)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

