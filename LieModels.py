import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import baseline_models
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import warnings
import matplotlib.pyplot as plt
import interp_utils
from tqdm import tqdm
import itertools



# Base class for network with Lie GrpA Filters
class LieSPmain(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cpu'
    if torch.cuda.is_available():
      self.device = 'cuda'
    self.config = config
    self.layers = [] # stores the GrpA filters
    self.dataset = [] # stores the grid on which the signal lies
    self.l1 = None
    self.l2 = nn.Linear(self.config['linear_bottleneck'], self.config['output'])
    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.nonlin = nn.SiLU()

    # Whether or not we want to prefilter with a CNN.. Only for 2d images
    self.conv1 = None

  def forward(self, x, add_noise = False):
    if self.conv1 is not None: # apply prefilter convolution if it's there
      x = x.reshape(x.shape[0], 28,28)
      x = self.conv1(x.unsqueeze(1))
      x = x.flatten(2)
      x = x.permute(0,2,1)
    else:
      x = x.unsqueeze(dim = -1).tile([1,1,self.config['num_filters']]) # otherwise expand input by number of filters
    for layer_ in self.layers: # Apply each layer
      x = layer_(x,add_noise = add_noise)
    x = torch.max(x,axis = -1)[0]
    x = self.l1(self.dropout(x)) # Final linear layer
    return x

# def add_epsilon_noise(sparse_coo_tensor, epsilon = 0.01):
#   dense_tensor = sparse_coo_tensor.to_dense()
#   dense_tensor += torch.from_numpy(np.diag(np.random.rand(dense_tensor.shape[0])*epsilon))
#   return dense_tensor.to_sparse_coo()


# FOR SO(3) transformations
class LieModel_SO3(LieSPmain):
  def __init__(self, config, dataset, output):
    super().__init__(config)
    self.isGrid = dataset.config['grid_config']['sample_type']
    self.dim = 2
    self.output = output
    if hasattr(dataset.dataset, 'linspace'): # This is meant for trilnear inteprolation on 3d grid structures
      self.dim = 3
      self.linspace = dataset.dataset.linspace
    self.dataset.append(dataset.dataset.grid)
    self.create_action()
    if self.l1 is None:
      raise NotImplementedError # The layer defined in the child class

  def create_action(self):
    '''
    Uses the information from config to determine the lie group sampling scheme and generates the induced transformations
    :return:
    '''
    degree = self.config['granularity']
    range1 = self.config['taps']
    flat1 = self.dataset[0]
    shape1 = len(flat1)
    # For each layer (built this way to enable maxpool for future work)
    for iter_ in range(self.config['layers']):
      rot_matrix_theta = []
      rot_matrix_phi = []
      rot_z = []
      # Find the induced transformations for each degree on SO(3)
      for i in range(range1):
        degree1 = i * degree
        degree1 = degree1 * np.pi / 360
        R_theta = np.array([[1, 0, 0], [0, np.cos(degree1), -np.sin(degree1)], [0, np.sin(degree1), np.cos(degree1)]])
        R_phi = np.array([[np.cos(degree1), 0, np.sin(degree1)], [0, 1, 0], [-np.sin(degree1), 0, np.cos(degree1)]])
        R_z = np.array([[np.cos(degree1), -np.sin(degree1), 0],[np.sin(degree1), np.cos(degree1), 0],[0, 0, 1]])
        val, x, y = self.createR_sparse_bary(flat1, R_theta)
        m_tensor_theta = torch.sparse_coo_tensor(np.array([x, y]), val, size=(shape1, shape1))
        val, x, y = self.createR_sparse_bary(flat1, R_phi)
        m_tensor_phi = torch.sparse_coo_tensor(np.array([x, y]), val, size=(shape1, shape1))
        val, x, y = self.createR_sparse_bary(flat1, R_z)
        m_tensor_z = torch.sparse_coo_tensor(np.array([x, y]), val, size=(shape1, shape1))

        rot_matrix_theta.append(m_tensor_theta.float().to(self.device))
        rot_matrix_phi.append(m_tensor_phi.float().to(self.device))
        rot_z.append(m_tensor_z.float().to(self.device))
        if i > 0:
          degree1 = 360 - i * degree
          degree1 = degree1 * np.pi / 180
          R_theta = np.array(
            [[1, 0, 0], [0, np.cos(degree1), -np.sin(degree1)], [0, np.sin(degree1), np.cos(degree1)]])
          R_phi = np.array(
            [[np.cos(degree1), 0, np.sin(degree1)], [0, 1, 0], [-np.sin(degree1), 0, np.cos(degree1)]])
          R_z = np.array([[np.cos(degree1), -np.sin(degree1), 0], [np.sin(degree1), np.cos(degree1), 0], [0, 0, 1]])

          val, x, y = self.createR_sparse_bary(flat1, R_theta)
          m_tensor_theta = torch.sparse_coo_tensor(np.array([x, y]), val, size=(shape1, shape1))
          # m_tensor_theta = add_epsilon_noise(m_tensor_theta)
          val, x, y = self.createR_sparse_bary(flat1, R_phi)
          m_tensor_phi = torch.sparse_coo_tensor(np.array([x, y]), val, size=(shape1, shape1))
          val, x, y = self.createR_sparse_bary(flat1, R_z)
          m_tensor_z = torch.sparse_coo_tensor(np.array([x, y]), val, size=(shape1, shape1))

          rot_matrix_theta.append(m_tensor_theta.float().to(self.device))
          rot_matrix_phi.append(m_tensor_phi.float().to(self.device))
          rot_z.append(m_tensor_z.float().to(self.device))

      if len(self.layers) == 0:
        inputsize = self.config['num_filters']
      else:
        inputsize = self.layers[-1].output
      self.layers.append(LieLayer((rot_matrix_phi, rot_matrix_theta, rot_z), self.config, inputsize=inputsize))
      self.register_parameter(name='filter' + str(iter_), param=self.layers[-1].params)
      if self.layers[-1].linear.weight is not None:
        self.register_parameter(name='filter_linear' + str(iter_), param=self.layers[-1].linear.weight)
    self.l1 = torch.nn.Linear(self.layers[-1].max_out,
                              self.output)    # self.l1 = torch.nn.Linear(self.layers[-1].generators.shape[-1],self.config['output'])

  def createR_sparse_bary(self, source, T, k=3):
    if np.all(T == np.eye(3)):
      # Identity, no need to multiply
      length1 = source.shape[0]
      return np.ones(length1), np.arange(length1), np.arange(length1)

    tol = 1e-5
    # check the dimensionality
    if self.isGrid and self.dim == 3:
      reference = interp_utils.create_reference(self.linspace[0],self.linspace[1],self.linspace[2])
      # Need to consider 2dim case
      # print('here')
      newflat = T @ source.T
      length1 = source.shape[0]
      weights = []
      coords = []
      index = []
      for iter_, data_ in enumerate(newflat.T):
        if interp_utils.check_inside(data_, self.linspace):
          weights_, coords_ = interp_utils.trilinear_interpolation(self.linspace[0],self.linspace[1],self.linspace[2], reference, data_[0], data_[1], data_[2])
          weights.append(weights_)
          coords.append(coords_)
          index.append([iter_ for i in range(len(coords_))])

      return np.concatenate(weights), np.concatenate(coords), np.concatenate(index)
    else:
      newflat = T @ source.T
      length1 = source.shape[0]

      knn = NearestNeighbors(n_neighbors=k)
      knn.fit(source)
      distances, indicies = knn.kneighbors(newflat.T)
      weights = np.zeros_like(distances)
      for iter_, index_ in enumerate(indicies):
        mat1 = np.ones((k, source.shape[1] + 1))
        mat1[:, 1:] = source[index_, :]
        target = np.ones(newflat.shape[0] + 1, )
        target[1:] = newflat[:, iter_]
        lstsq_solution = np.linalg.lstsq(mat1.transpose(), target, rcond=None)
        weights[iter_, :] = lstsq_solution[0]

      corr_row = np.arange(length1)
      corr_row = np.tile(corr_row, (k, 1)).T
      return np.concatenate(weights), np.concatenate(indicies), np.concatenate(corr_row)


# For singals with SO2 group symmetries
class LieModel_SO2(LieSPmain):
  def __init__(self, config, dataset):
    super().__init__(config)
    if config['conv_prefilter']:
      self.conv1 = nn.Sequential(
              nn.Conv2d(
                  in_channels=1,
                  out_channels=config['num_filters'],# should be from config
                  kernel_size=5,
                  stride=1,
                  padding=2,
              ),
              self.nonlin,
          )

    self.dataset.append(dataset.dataset.grid)
    self.createRay_action() # populate the self.layers with the induced group actions
    if self.l1 is None:
      raise NotImplementedError # The layer defined in the child class

  def createRay_action(self):
    degree_ = self.config['granularity']
    range1_ = self.config['taps']
    ray_start = self.config['raystart']
    ray_end = self.config['rayend']
    ray_density = self.config['raydensity']
    constants = np.linspace(ray_start, ray_end, ray_density)
    flat1 = self.dataset[0]
    shape1 = len(flat1)

    # For each layer
    for iter_, (degree, range1) in enumerate(zip(degree_, range1_)):
      # create the rotation filters
      rot_matrix_theta = []
      for i in range(range1):
        degree1 = i * degree
        degree1 = degree1 * np.pi / 360
        R_theta = np.array([[np.cos(degree1), -np.sin(degree1)],[np.sin(degree1), np.cos(degree1)]])
        val, x, y = createR_sparse_bary(flat1, R_theta)
        m_tensor_theta = torch.sparse_coo_tensor(np.array([x, y]), val, size=(shape1, shape1))
        rot_matrix_theta.append(m_tensor_theta.float().to(self.device))
        if i > 0:
          degree1 = 360 - i * degree
          degree1 = degree1 * np.pi / 180
          R_theta = np.array([[np.cos(degree1), -np.sin(degree1)], [np.sin(degree1), np.cos(degree1)]])
          val, x, y = createR_sparse_bary(flat1, R_theta)
          m_tensor_theta = torch.sparse_coo_tensor(np.array([x, y]), val, size=(shape1, shape1))
          rot_matrix_theta.append(m_tensor_theta.float().to(self.device))

      # Create the ray fitlers
      ray_matrix = []
      for constant_ in constants:
        val, x, y =createD_sparse(flat1, constant_)
        m_tensor_ray = torch.sparse_coo_tensor(np.array([x, y]), val, size=(shape1, shape1))
        ray_matrix.append(m_tensor_ray.float().to(self.device))
      if len(self.layers) == 0:
        inputsize = self.config['num_filters']
      else:
        inputsize = self.layers[-1].output

      if iter_ < self.config['layers']:
        if self.config['include_ray']:
          self.layers.append(LieLayer((rot_matrix_theta,ray_matrix), self.config, inputsize=inputsize))
        else:
          self.layers.append(LieLayer([rot_matrix_theta], self.config, inputsize=inputsize))
        self.register_parameter(name='filter' + str(iter_), param=self.layers[-1].params)
        if self.layers[-1].linear is not None:
          self.register_parameter(name='filter_linear' + str(iter_), param=self.layers[-1].linear.weight)
    # add the last linear layer
    self.l1 = torch.nn.Linear(self.layers[-1].max_out, self.config['output'])


###########Previous iterations to implement the filter############
# def ExpandFilterFunction2(x, params, rotations):
#   x = x.permute(1,0,2)#.flatten(1)
#   repeat_shape = [1 for i in x.shape]
#   repeat_shape[-1] = params.shape[0]
#   xout = torch.zeros_like(x).repeat(repeat_shape)
#   count1 = 0
#   for ell in range(params.shape[0]):
#     for i in range(x.shape[-1]): # i is the number of inputs
#       for k in range(rotations.shape[0]):
#         xout[:,:,count1] += params[ell, i, k]*torch.mm(rotations[k], x[:,:,i])
#       count1 +=1
#   return xout#.permute(1,0,2)
#
#
# def ExpandFilterFunctionNew(x,params, rotations):
#   rotations = list(rotations)
#   iter_order = itertools.permutations(np.arange(3))
#   i = 0
#   x_out = torch.zeros_like(x).squeeze(2).T
#   for order_ in iter_order:
#     new_list = [rotations[order_[0]], rotations[order_[1]], rotations[order_[2]]]
#     for iter_, element in enumerate(itertools.product(*new_list)):
#       transform = torch.sparse.mm(torch.sparse.mm(element[0], element[1]), element[2])
#       output = params[i]*torch.sparse.mm(transform.double(),x.T.squeeze(0))
#       x_out += output
#       i+=1
#   return x_out
# def ExpandFilterFunction(x, params, rotations):
#   '''
#
#   :param x: should have shape (batchsize, feature size, channels)
#   :param params: should have size (expand ratio size, channels, taps)
#   :param rotations:  should have size (taps, feature size, feature size)
#   :return: should return (feature size, batchsize, channels*expand ratio)
#   '''
#
#   x = x.permute(1,0,2)#.flatten(1)
#   n, b, c = x.shape
#   repeat_shape = [1 for i in x.shape]
#   repeat_shape[-1] = params.shape[0]
#   xout = torch.zeros_like(x).repeat(repeat_shape) # target shape
#   count1 = 0
#   for ell in range(params.shape[0]): # for each expansion
#     for i in range(params.shape[1]):
#       param1 = params[ell, i, :].unsqueeze(-1).unsqueeze(-1).repeat(1, rotations.shape[-1], rotations.shape[-1])
#       rotationsnew =param1*rotations
#       x1 = x[:, :, i]
#       xout[:, :, count1] = torch.mm(rotationsnew.sum(axis = 0), x1.flatten(1)).reshape(n, b)
#       count1 += 1
#       del param1
#       del rotationsnew
#       del x1
#   return xout#.permute(1,0,2)
#################################################



class LieLayer(nn.Module):
  def __init__(self, generators, config, inputsize, stdv = .01):
    '''
    :param generators: Induced transformations from the Lie Algebra basis via exponential map
    :param config: model config
    :param inputsize: Input size
    :param stdv: variance for initalization
    '''
    super().__init__()
    self.device = 'cpu'
    if torch.cuda.is_available():
      self.device = 'cuda'

    self.generators_length = None
    self.inputsize = inputsize
    self.linear = None
    self.epsilon = config['epsilon']
    self.expand_generators(generators)
    self.output = inputsize*config['expand_ratio']
    if config['add_linear']:
      self.linear = nn.Linear(in_features=inputsize*config['expand_ratio'], out_features=inputsize*config['contract_ratio']).to(self.device)
      self.output = inputsize*config['contract_ratio']

    self.params = nn.Parameter(stdv*torch.rand(np.math.factorial(len(generators))*len(generators[0])**len(generators), config['expand_ratio']) - stdv)
    self.max_out = generators[0][0].shape[0]
    self.activation = Swish()#nn.SiLU()
    self.bn = nn.BatchNorm1d(generators[0][0].shape[0]).to(self.device)
  def expand_generators(self, generators):
    '''
    Currently only handles up to 3 generators
    :param generators:
    :return: transformed generators in self.generator
    '''
    # Assuming generators are non-commutative: iterate through all permutations
    n_params = np.math.factorial(len(generators))*len(generators[0])**len(generators)
    iter_order = itertools.permutations(np.arange(len(generators)))
    transform_full = torch.zeros(n_params, generators[0][0].shape[0], generators[0][0].shape[0])
    transform_epsilon = torch.zeros(n_params, generators[0][0].shape[0], generators[0][0].shape[0])
    i = 0
    for order_ in iter_order:
      new_list = [generators[order_[idx_]] for idx_ in range(len(generators))]
      for iter_, element in enumerate(itertools.product(*new_list)):
        sparse_dummy = element[0]
        if len(element) > 0:
          iter_elem = 1
          # Multiply generatros in order given by itertools
          while iter_elem < len(element):
            sparse_dummy = torch.sparse.mm(sparse_dummy, element[iter_elem])
            iter_elem += 1
        transform_full[i,:,:] = sparse_dummy.to_dense()
        transform_epsilon[i,:,:] = transform_full[i,:,:] + torch.from_numpy(np.diag(np.random.rand(transform_full[i,:,:].shape[0])*self.epsilon))
        i+=1

    self.generators = transform_full
    self.epsilon_generators = transform_epsilon
  def forward(self, x, dataset = None, add_noise = False):
    if add_noise:
      x = ExpandFilterFunctionNew2(x, self.params, self.epsilon_generators)  # Apply the filter
    else:
      x = ExpandFilterFunctionNew2(x, self.params, self.generators) # Apply the filter
    if self.linear is not None: #Apply linear layer if it exists
      x = self.linear(x)
    return self.activation(x) # apply nonlinearity


# HELPER FUNCTIONS

def ExpandFilterFunctionNew2(x,params, transformations):
  # Outputs the filter given by (3) of https://arxiv.org/pdf/2210.17425.pdf
  if torch.cuda.is_available():
    device = 'cuda'
  else:
    device = 'cpu'
  output = torch.zeros((x.shape[0], x.shape[1], params.shape[-1])).to(device)
  for dim_ in range(params.shape[-1]):
    val1 = transformations.shape[-1]
    filter = (params[:,dim_].cpu() @ transformations.flatten(1)).reshape(val1,val1)
    output[:,:,dim_] = (filter.to(device)@x.flatten(1).permute(1,0)).permute(1,0)
  return output

def createD_sparse(source, constant, k = 3):
  source = source.T
  newflat = np.zeros_like(source)
  newflat[0] = constant * np.multiply(np.linalg.norm(source, axis=0) , np.cos(np.arctan2(source[1], source[0])))
  newflat[1]=  constant * np.linalg.norm(source, axis=0) * np.sin(np.arctan2(source[1], source[0]))
  length1 = source.shape[1]
  source = source.T
  newflat = newflat.T
  knn = NearestNeighbors(n_neighbors=k)
  knn.fit(source)
  distances, indicies = knn.kneighbors(newflat)
  weights = np.zeros_like(distances)
  newflat = newflat.T
  for iter_, index_ in enumerate(indicies):
    mat1 = np.ones((k, source.shape[1]+1))
    mat1[:,1:] = source[index_,:]
    target = np.ones(newflat.shape[0] + 1, )
    target[1:] = newflat[:,iter_]
    lstsq_solution = np.linalg.lstsq(mat1.transpose(), target, rcond=None)
    weights[iter_,:] = lstsq_solution[0]

  corr_row = np.arange(length1)
  corr_row = np.tile(corr_row, (k, 1)).T
  return np.concatenate(weights), np.concatenate(indicies), np.concatenate(corr_row)


def createR_sparse_bary(source, T, k = 3):
  tol = 1e-5
  # check the dimensionality
  newflat = T @ source.T
  length1 = source.shape[0]

  knn = NearestNeighbors(n_neighbors=k)
  knn.fit(source)
  distances, indicies = knn.kneighbors(newflat.T)
  weights = np.zeros_like(distances)
  for iter_, index_ in enumerate(indicies):
    mat1 = np.ones((k, source.shape[1]+1))
    mat1[:,1:] = source[index_,:]
    target = np.ones(newflat.shape[0] + 1, )
    target[1:] = newflat[:,iter_]
    lstsq_solution = np.linalg.lstsq(mat1.transpose(), target, rcond=None)
    weights[iter_,:] = lstsq_solution[0]

  corr_row = np.arange(length1)
  corr_row = np.tile(corr_row, (k, 1)).T
  return np.concatenate(weights), np.concatenate(indicies), np.concatenate(corr_row)


'''
Below is copied from mfinzi/LieConv
'''
class Expression(nn.Module):

  def __init__(self, func):
    super().__init__()
    self.func = func

  def forward(self, x):
    return self.func(x)

def Swish():
  return Expression(lambda x: x * torch.sigmoid(x))