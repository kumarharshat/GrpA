import os

import numpy as np
import torch; torch.set_default_dtype(torch.float64)
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
import warnings
from base_data_class import base_data_class, CustomDataset

###########################
#
# Rotated MNIST Data Loader
#
###########################



class RotMNIST(base_data_class):
    def __init__(self, config):
        super().__init__()
        self.grid_config = config['grid_config']
        self.filename = config['filename']
        self.num_targets = 10
        self.train_noise = config['train_noise']
        self.test_noise = config['test_noise']
        self.load_raw_data()
        self.grid = self.generate_grid()
        self.projected_points = self.proj_on_grid_bary()
        self.train_loader, self.val_loader, self.test_loader = self.split_dataset()

        # Don't need the raw data anymore
        self.x_val = None
        self.x_test = None
        self.x_train = None

    def load_raw_data(self):
        # Make sure this data is downloaded and in the correct folder
        fileName = os.getcwd() + '/data/mnist_all_rotation_normalized_float_train_valid.amat'
        data = np.loadtxt(fileName, delimiter=' ')
        # Hard coded train-val split to mimic related works
        self.x_train = torch.as_tensor(data[0:10000, :-1].reshape(-1, 28, 28).astype(np.float32), dtype=torch.float32)
        self.y_train = torch.as_tensor(data[0:10000, -1].astype(np.int64))
        self.x_val = torch.as_tensor(data[10000:, :-1].reshape(-1, 28, 28).astype(np.float32), dtype=torch.float32)
        self.y_val = torch.as_tensor(data[10000:, -1].astype(np.int64))
        fileName = os.getcwd() + '/data/mnist_all_rotation_normalized_float_test.amat'
        data = np.loadtxt(fileName, delimiter=' ')
        self.x_test = torch.as_tensor(data[:, :-1].reshape(-1, 28, 28).astype(np.float32), dtype=torch.float32)
        self.y_test = torch.as_tensor(data[:, -1].astype(np.int64))

    def generate_grid(self):
        if self.grid_config['sample_type'] =='grid':
            cuberootgran = len(self.x_train[0])
            x = np.linspace(self.grid_config['x'][0], self.grid_config['x'][1],
                            cuberootgran)  # should be able to customize these
            y = np.linspace(self.grid_config['y'][0], self.grid_config['y'][1],
                            cuberootgran)  # should be able to customize these
            x, y = np.meshgrid(x, y)

            return np.array([x.flatten(order='C'), y.flatten(order='C')]).T
        else:
            assert False, 'Only GRID is supported for rotated MNIST class'



    def proj_on_grid_bary(self):
        if self.grid_config['sample_type'] == 'grid':
            # All we need to do is flatten here, no projection
            self.x_train = self.x_train.flatten(1)
            self.x_val = self.x_val.flatten(1)
            self.x_test = self.x_test.flatten(1)
        else:
            raise NotImplementedError

    def reshape_fortran(self, x, shape):
        if len(x.shape) > 0:
            x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

    def split_dataset(self):
        train_loader = CustomDataset(self.x_train, self.y_train, noise=self.train_noise)
        val_loader = CustomDataset(self.x_val, self.y_val,
                                       noise=self.train_noise)
        test_loader = CustomDataset(self.x_test, self.y_test, noise=self.test_noise)
        return train_loader, val_loader, test_loader


# Function calls which type of knots we want to classify
def load_data(config):
    return RotMNIST(config)
