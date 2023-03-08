import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn

'''
Baseline models contains class for FCNN models
'''

class regular2(nn.Module):
  '''
  Two layer FCNN
  '''
  def __init__(self, input_size, num_classes,dropout = 0):
    super().__init__()
    self.inputsize = input_size[0]
    self.l1 = torch.nn.Linear(self.inputsize,50) # Hardcode.. should generalize this
    self.l2 = torch.nn.Linear(50,num_classes)
    self.relu = nn.ReLU()
    self.dropout = torch.nn.Dropout(dropout)
    self.sigmoid = torch.nn.Sigmoid()
    self.softmax = torch.nn.Softmax(dim = 1)

  def forward(self, x):
    x = self.l1(x)
    # x = self.dropout(x)
    x = self.relu(x)
    x = self.l2(x)
    return self.relu(x)

class regular1(nn.Module):
  '''
  1 Layer FCNN
  '''
  def __init__(self, input_size, num_classes,dropout  = 0):
    super().__init__()
    self.inputsize = input_size[0]
    self.l1 = torch.nn.Linear(self.inputsize,num_classes)
    self.relu = nn.ReLU()
    self.dropout = torch.nn.Dropout(dropout)
    self.sigmoid = torch.nn.Sigmoid()
    self.softmax = torch.nn.Softmax(dim = 1)

  def forward(self, x):
    x = self.l1(x)
    # x = self.dropout(x)
    return self.relu(x)

