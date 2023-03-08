import numpy as np
import torch;

# import dataloader

torch.set_default_dtype(torch.float64)
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from base_data_class import base_data_class, CustomDataset


################################
#
# Build the custom knot dataset
#
################################

# Main class with projection steps
class Knots(base_data_class):
    def __init__(self, config):
        super().__init__()
        self.grid = self.generate_grid() # Generate the grid on which to project the knots
        self.jitter_points() # Jitter the knots
        self.projected_points = self.proj_on_grid_bary() # project the knots on the grid
        self.train_loader, self.val_loader, self.test_loader = self.split_dataset() # Split the dataset and create the data loader classes
        self.get_loss_score() # computes loss score for knot projection

    def jitter_points(self):
        # move the points of the knot according to the train and test noise
        val1 = int(self.num_samples*self.val_test_split[0])
        val2 = int(self.num_samples*(self.val_test_split[0]+ self.val_test_split[1]))
        self.rawdata[0:val1] = self.rawdata[0:val1] + np.reshape(np.random.normal(self.train_noise[0], self.train_noise[1], self.rawdata[0:val1].size), self.rawdata[0:val1].shape)
        self.rawdata[val1:val2] = self.rawdata[val1:val2] + np.reshape(
            np.random.normal(self.train_noise[0], self.train_noise[1], self.rawdata[val1:val2].size),
            self.rawdata[val1:val2].shape)
        self.rawdata[val2:] = self.rawdata[val2:] + np.reshape(
            np.random.normal(self.test_noise[0], self.test_noise[1], self.rawdata[val2:].size),
            self.rawdata[val2:].shape)

    def get_loss_score(self):
        if hasattr(self, 'rawdata'):
            self.loss_score = self.projected_points.sum(axis=1).mean() # proxy for projected loss
        else:
            self.loss_score = None

    def split_dataset(self):
        val1 = int(self.num_samples*self.val_test_split[0])
        val2 = int(self.num_samples*(self.val_test_split[0]+ self.val_test_split[1]))

        train_loader = CustomDataset(self.projected_points[0:val1], self.labels[0:val1])
        val_loader = CustomDataset(self.projected_points[val1:val2], self.labels[val1:val2])
        test_loader = CustomDataset(self.projected_points[val2:], self.labels[val2:])
        return train_loader, val_loader, test_loader

    def generate_grid(self):
        # Generate the grid according to the dataset config
        cuberootgran = self.grid_config['cuberootgran']
        if self.grid_config['sample_type'] == 'grid': # Regular grid
            x = np.linspace(self.grid_config['x'][0], self.grid_config['x'][1], self.grid_config['cuberootgran'])  # should be able to customize these
            y = np.linspace(self.grid_config['y'][0], self.grid_config['y'][1], self.grid_config['cuberootgran'])  # should be able to customize these
            z = np.linspace(self.grid_config['z'][0], self.grid_config['z'][1], self.grid_config['cuberootgran'])  # should be able to customize these
            x, y, z = np.meshgrid(x, y, z)

            return np.array([x.flatten(order='C'), y.flatten(order='C'), z.flatten(order='C')]).T
        elif self.grid_config['sample_type'] == 'uniform': # uniform distribution on grid

            x = np.random.rand(cuberootgran ** 3, )  * (self.grid_config['x'][1] - self.grid_config['x'][0]) + self.grid_config['x'][0]
            y = np.random.rand(cuberootgran ** 3, )  * (self.grid_config['x'][1] - self.grid_config['y'][0]) + self.grid_config['y'][0]
            z = np.random.rand(cuberootgran ** 3, )  * (self.grid_config['x'][1] - self.grid_config['z'][0]) + self.grid_config['z'][0]
            return np.array([x, y, z]).T
        elif self.grid_config['sample_type'] == 'gaussian': # gaussian distribution
            return np.array(
                [np.random.normal(0.5*self.grid_config['x'][1] + 0.5*self.grid_config['x'][0], 0.5*self.grid_config['x'][1] - 0.5*self.grid_config['x'][0], cuberootgran ** 3), np.random.normal(0.5*self.grid_config['y'][1] -0.5* self.grid_config['y'][0], 0.5*self.grid_config['y'][1] - 0.5*self.grid_config['y'][0], cuberootgran ** 3),
                 np.random.normal(0.5*self.grid_config['z'][1] - 0.5*self.grid_config['z'][0], 0.5*self.grid_config['z'][1] - 0.5*self.grid_config['z'][0], cuberootgran ** 3)]).T
        elif self.grid_config['sample_type'] == 'circle': # uniform on sphere
            theta = np.random.rand(cuberootgran ** 3, ) * np.pi
            phi = np.random.rand(cuberootgran ** 3, ) * np.pi * 2
            r = np.random.rand(cuberootgran ** 3, ) * (0.5*self.grid_config['z'][1] - 0.5*self.grid_config['z'][0])
            return np.array([r * np.cos(phi) * np.sin(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(theta)]).T
        else:
            assert False, "Invalid grid type"

    def proj_on_grid_bary(self):
        '''
        Project the raw data on the generated grid
        '''
        tol = 1e-5
        x_train_new = []
        for x_ in self.rawdata:
            knn = NearestNeighbors(n_neighbors=self.grid_config['k'])
            knn.fit(x_.T)
            distances, indicies = knn.kneighbors(self.grid)
            neg_distances = 1 - distances.mean(axis=1)
            neg_distances[neg_distances < 0] = 0
            x_train_new.append(neg_distances)
        x_train = np.array(x_train_new)
        return x_train


# Function calls which type of knots we want to classify
def load_knots(config):
    if config['knottype'] == 'saved':
        assert False, "Need to add this functionality"
    elif config['knottype'] == 'custom':
        return CustomKnots(config)
    return

class CustomKnots(Knots):
    def __init__(self, config):
        self.grid_config = config['grid_config']
        self.num_samples = config['num_samples']
        self.num_features = config['num_features']
        self.filename = config['filename']
        self.num_targets = 2
        self.train_noise = config['train_noise']
        self.test_noise = config['test_noise']
        self.rawdata, self.labels = self.generate_knots()
        self.val_test_split = config['val_test_split']
        super().__init__(self)

        # self.projected_knots = self.proj_on_grid_bary(self.generate_grid())

    def generate_knots(self):
        '''
        Generates the custom knots (trefoil and Figure eight knot) parametrized
        :return:
        '''

        # Generation of parametrized knots
        t = np.linspace(0, 2 * np.pi, self.num_features)
        x_train_base = []
        # Trefoil Knot
        x_train_base.append(np.array([np.sin(t) - 2 * np.sin(2 * t), np.cos(t) + 2 * np.cos(2 * t), -np.sin(3 * t)]))
        # Figure Eight Knot
        x_train_base.append(
            np.array([(2 + np.cos(2 * t)) * np.cos(3 * t), (2 + np.cos(2 * t)) * np.sin(3 * t), np.sin(4 * t)]))

        # Random rotations about x and y axis
        thetas = np.random.rand(int(self.num_samples/2), ) * np.pi * 2
        phis = np.random.rand(int(self.num_samples/2), ) * np.pi * 2
        zs = np.random.rand(int(self.num_samples/2), ) * np.pi * 2
        R_thetas = np.array([[np.ones(int(self.num_samples/2)), np.zeros(int(self.num_samples/2)), np.zeros(int(self.num_samples/2))],
                            [np.zeros(int(self.num_samples/2)), np.cos(thetas), -np.sin(thetas)],
                            [np.zeros(int(self.num_samples/2)), np.sin(thetas), np.cos(thetas)]]).transpose()
        R_phis = np.array([[np.cos(phis), np.zeros(int(self.num_samples/2)), np.sin(phis)],
                          [np.zeros(int(self.num_samples/2)), np.ones(int(self.num_samples/2)), np.zeros(int(self.num_samples/2))],
                          [-np.sin(phis), np.zeros(int(self.num_samples/2)), np.cos(phis)]]).transpose()
        R_zs = np.array([[np.cos(zs), -np.sin(phis),np.zeros(int(self.num_samples/2))],
                        [np.sin(phis), np.cos(zs),np.zeros(int(self.num_samples/2))],
                        [np.zeros(int(self.num_samples/2)),np.zeros(int(self.num_samples/2)),np.ones(int(self.num_samples/2))]]).transpose()

        x_set = []
        y_set = []
        # Apply the rotations with random order given by 'order_'
        for R_theta, R_phi, R_z in zip(R_thetas, R_phis, R_zs):
            order_ = np.random.randint(6)
            if order_ == 0:
                R = R_theta @ R_phi
                R = R @ R_z
            if order_ == 1:
                R = R_theta @ R_z
                R = R @ R_phi
            if order_ == 2:
                R = R_phi @ R_theta
                R = R @ R_z
            if order_ == 3:
                R = R_phi @ R_z
                R = R @ R_theta
            if order_ == 4:
                R = R_z @ R_phi
                R = R @ R_theta
            if order_ == 5:
                R = R_z @ R_theta
                R = R @ R_phi
            for iter_ in range(2):
                x_set.append(R @ (x_train_base[iter_]))
                y_set.append(iter_)
        x_set = np.stack(x_set)
        y_set = np.stack(y_set)
        return x_set, y_set


