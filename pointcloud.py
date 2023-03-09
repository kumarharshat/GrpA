# import os
#
# import numpy as np
# import torch; torch.set_default_dtype(torch.float64)
# from sklearn.neighbors import NearestNeighbors
# from torch.utils.data import Dataset
# import warnings
# from torch_points3d.datasets.classification.modelnet import SampledModelNet
# import torch_points3d.core.data_transform as T3D
# import torch_geometric.transforms as T
# import tqdm
# from base_data_class import base_data_class, CustomDataset
#
#############################################
## Uncomment to run code with ModelPoint10!##
#############################################
# class PointCloud(base_data_class):
#     def __init__(self, config):
#         super().__init__()
#         self.grid_config = config['grid_config']
#         self.filename = config['filename']
#         self.num_targets = 10
#         self.train_noise = config['train_noise']
#         self.test_noise = config['test_noise']
#         self.load_raw_data()
#         self.grid = self.generate_grid()
#         self.projected_points, self.labels = self.proj_on_grid_bary(self.rawdata)
#         self.projected_points_test, self.labels_test = self.proj_on_grid_bary(self.rawdata_test)
#         self.train_loader, self.val_loader, self.test_loader = self.split_dataset()
#         self.rawdata = None
#         self.rawdata_test = None
#
#     def load_raw_data(self):
#         MODELNET_VERSION = "10"  # @param ["10", "40"]
#         USE_NORMAL = True  # @param {type:"boolean"}
#         dataroot = os.path.join("", "data/modelnet")
#         pre_transform = T.Compose([T.NormalizeScale(), T3D.GridSampling3D(0.02)])
#         dataset = SampledModelNet(dataroot, name=MODELNET_VERSION, train=True, transform=T.FixedPoints(2048, replace=True),
#                                   pre_transform=pre_transform, pre_filter=None)
#         dataset_test = SampledModelNet(dataroot, name=MODELNET_VERSION, train=False, transform=T.FixedPoints(2048, replace=True),
#                                   pre_transform=pre_transform, pre_filter=None)
#
#         self.rawdata = dataset
#         self.rawdata_test = dataset_test
#
#     def generate_grid(self):
#         cuberootgran = self.grid_config['cuberootgran']
#
#         if self.grid_config['sample_type'] == 'grid':
#             x = np.linspace(self.grid_config['x'][0], self.grid_config['x'][1], self.grid_config['cuberootgran'])  # should be able to customize these
#             y = np.linspace(self.grid_config['y'][0], self.grid_config['y'][1], self.grid_config['cuberootgran'])  # should be able to customize these
#             z = np.linspace(self.grid_config['z'][0], self.grid_config['z'][1], self.grid_config['cuberootgran'])  # should be able to customize these
#             self.linspace = (x,y,z)
#             x, y, z = np.meshgrid(x, y, z)
#
#             return np.array([x.flatten(order='C'), y.flatten(order='C'), z.flatten(order='C')]).T
#         elif self.grid_config['sample_type'] == 'uniform':
#
#             x = np.random.rand(cuberootgran ** 3, )  * (self.grid_config['x'][1] - self.grid_config['x'][0]) + self.grid_config['x'][0]
#             y = np.random.rand(cuberootgran ** 3, )  * (self.grid_config['x'][1] - self.grid_config['y'][0]) + self.grid_config['y'][0]
#             z = np.random.rand(cuberootgran ** 3, )  * (self.grid_config['x'][1] - self.grid_config['z'][0]) + self.grid_config['z'][0]
#             return np.array([x, y, z]).T
#         elif self.grid_config['sample_type'] == 'gaussian':
#             return np.array(
#                 [np.random.normal(0.5*self.grid_config['x'][1] + 0.5*self.grid_config['x'][0], 0.5*self.grid_config['x'][1] - 0.5*self.grid_config['x'][0], cuberootgran ** 3), np.random.normal(0.5*self.grid_config['y'][1] -0.5* self.grid_config['y'][0], 0.5*self.grid_config['y'][1] - 0.5*self.grid_config['y'][0], cuberootgran ** 3),
#                  np.random.normal(0.5*self.grid_config['z'][1] - 0.5*self.grid_config['z'][0], 0.5*self.grid_config['z'][1] - 0.5*self.grid_config['z'][0], cuberootgran ** 3)]).T
#         elif self.grid_config['sample_type'] == 'circle':
#             theta = np.random.rand(cuberootgran ** 3, ) * np.pi
#             phi = np.random.rand(cuberootgran ** 3, ) * np.pi * 2
#
#             r = np.random.rand(cuberootgran ** 3, ) * (0.5*self.grid_config['z'][1] - 0.5*self.grid_config['z'][0])
#             return np.array([r * np.cos(phi) * np.sin(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(theta)]).T
#
#         else:
#             assert False, "Invalid grid type"
#
#
#
#     def proj_on_grid_bary(self, rawdata):
#         '''
#
#         :param grid:
#         :return: Projected knots on the grid
#         '''
#         tol = 1e-5
#         x_train_new = []
#         y_s = []
#         count1 = 0
#         for x_ in rawdata:
#             count1 +=1
#             if count1 %10 == 0:
#                 print(count1)
#             y_s.append(x_.y.numpy())
#             num_samples = 2024
#             thetas = np.random.rand() * np.pi * 2
#             phis = np.random.rand() * np.pi *2
#             zs = np.random.rand() * np.pi *2
#             R_theta = np.array([[1, 0, 0],
#                                 [0, np.cos(thetas), -np.sin(thetas)],
#                                 [0, np.sin(thetas), np.cos(thetas)]]).transpose()
#             R_phi = np.array([[np.cos(phis), 0, np.sin(phis)],
#                               [0, 1, 0],
#                               [-np.sin(phis), 0, np.cos(phis)]]).transpose()
#             R_z = np.array([[np.cos(zs), -np.sin(zs), 0], [np.sin(zs), np.cos(zs), 0], [0, 0, 1]])
#
#             order_ = np.random.randint(6)
#             if order_ == 0:
#                 R = R_theta @ R_phi
#                 R = R @ R_z
#             if order_ == 1:
#                 R = R_theta @ R_z
#                 R = R @ R_phi
#             if order_ == 2:
#                 R = R_phi @ R_theta
#                 R = R @ R_z
#             if order_ == 3:
#                 R = R_phi @ R_z
#                 R = R @ R_theta
#             if order_ == 4:
#                 R = R_z @ R_phi
#                 R = R @ R_theta
#             if order_ == 5:
#                 R = R_z @ R_theta
#                 R = R @ R_phi
#
#             x_ = torch.from_numpy(R)@x_.pos.T
#             knn = NearestNeighbors(n_neighbors=self.grid_config['k'])
#             knn.fit(x_.T)
#             distances, indicies = knn.kneighbors(self.grid)
#
#             neg_distances = 1-distances.mean(axis = 1)
#             neg_distances[neg_distances < 0] = 0
#             # x_train_new.append(1-distances.mean(axis = 1))
#             x_train_new.append(neg_distances)
#         x_train = np.array(x_train_new)
#         y_train = np.array(y_s)
#         return x_train, y_train
#
#     def reshape_fortran(self, x, shape):
#         if len(x.shape) > 0:
#             x = x.permute(*reversed(range(len(x.shape))))
#         return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
#
#     def split_dataset(self):
#         n_val = 20
#         label_idx = [np.flatnonzero(self.labels == label) for label in range(0, 40)]
#         label_idx = [np.random.RandomState(seed=42).permutation(idx) for idx in label_idx]
#         val_subset = [idx[:n_val] for idx in label_idx]
#         train_subset = [idx[n_val:] for idx in label_idx]
#
#         val_subset = np.concatenate(val_subset)
#         train_subset = np.concatenate(train_subset)
#
#         train_loader = CustomDataset(torch.from_numpy(self.projected_points[train_subset]), torch.from_numpy(self.labels[train_subset].squeeze()), noise=self.train_noise)
#         val_loader = CustomDataset(torch.from_numpy(self.projected_points[val_subset]), torch.from_numpy(self.labels[val_subset].squeeze()),
#                                        noise=self.train_noise)
#         test_loader = CustomDataset(torch.from_numpy(self.projected_points_test), torch.from_numpy(self.labels_test.squeeze()), noise=self.test_noise)
#         return train_loader, val_loader, test_loader
#
#
# # Function calls which type of knots we want to classify
# def load_data(config):
#     return PointCloud(config)
