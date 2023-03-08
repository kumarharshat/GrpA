import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import Knots
import RotMNIST
import pointcloud
from torch.utils.data import Dataset


class DataLoader():
    '''
    DataLoader class loads data given by the specified dataset config file
    '''
    def __init__(self, config):
        self.config = config
        self.dataset = None
        if config['datatype'] == 'knots':
            self.dataset = Knots.load_knots(config)
        if config['datatype'] == 'RotMNIST':
            self.dataset = RotMNIST.load_data(config)
        if config['datatype'] == 'pointcloud':
            self.dataset = pointcloud.load_data(config)

        if self.dataset is None:
            assert False, "Need to specify correct dataset"



