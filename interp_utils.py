import numpy as np
from itertools import product

'''
Helper functions for interpolation in 3D
'''

def check_inside(data_, source):
    '''

    :param data_: Data after group action
    :param source:
    :return:
    '''

    if data_[0] < source[0][0] or data_[0] > source[0][-1]:
        return False
    if data_[1] < source[1][0] or data_[1] > source[1][-1]:
        return False
    if data_[2] < source[2][0] or data_[2] > source[2][-1]:
        return False


    return True

def create_reference(x_volume, y_volume, z_volume):
    x = np.arange(0, len(x_volume))
    y = np.arange(0, len(y_volume))
    z = np.arange(0, len(z_volume))
    x1, y1, z1 = np.meshgrid(x,y,z)
    vector = np.array([x1.flatten(order='C'), y1.flatten(order='C'), z1.flatten(order='C')]).T
    return vector

# Taken from https://stackoverflow.com/questions/21836067/interpolate-3d-volume-with-numpy-and-or-scipy
def trilinear_interpolation(x_volume, y_volume, z_volume, reference, x_needed, y_needed, z_needed):
    reference = list(reference)
    """
    Trilinear interpolation (from Wikipedia)

    :param x_volume: x points of the volume grid
    :type crack_type: list or numpy.ndarray
    :param y_volume: y points of the volume grid
    :type crack_type: list or numpy.ndarray
    :param x_volume: z points of the volume grid
    :type crack_type: list or numpy.ndarray
    :param volume:   volume
    :type crack_type: list or numpy.ndarray
    :param x_needed: desired x coordinate of volume
    :type crack_type: float
    :param y_needed: desired y coordinate of volume
    :type crack_type: float
    :param z_needed: desired z coordinate of volume
    :type crack_type: float

    :return volume_needed: desired value of the volume, i.e. volume(x_needed, y_needed, z_needed)
    :type volume_needed: float
    """
    # dimensinoal check
    # if np.shape(volume) != (len(x_volume), len(y_volume), len(z_volume)):
    #     raise ValueError(f'dimension mismatch, volume must be a ({len(x_volume)}, {len(y_volume)}, {len(z_volume)}) list or numpy.ndarray')
    # check of the indices needed for the correct control volume definition
    i = searchsorted(x_volume, x_needed)
    j = searchsorted(y_volume, y_needed)
    k = searchsorted(z_volume, z_needed)
    # control volume definition
    control_volume_coordinates = np.array(
        [[x_volume[i - 1], y_volume[j - 1], z_volume[k - 1]], [x_volume[i], y_volume[j], z_volume[k]]])
    xd = (np.array([x_needed, y_needed, z_needed]) - control_volume_coordinates[0]) / (control_volume_coordinates[1] - control_volume_coordinates[0])
    Q = np.array([1, xd[0], xd[1], xd[2], xd[0]*xd[1], xd[1]*xd[2], xd[2]*xd[0], xd[0]*xd[1]*xd[2]])
    # matrix from https://spie.org/samples/PM159.pdf
    B = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                 [-1, 0, 0, 0, 1, 0, 0, 0],
                 [-1, 0, 1, 0, 0, 0, 0, 0],
                  [-1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, -1, 0, -1, 0, 1, 0],
                  [1, -1, -1, 1, 0, 0, 0, 0],
                  [1, -1, 0, 0, -1, 1, 0, 0],
                  [-1, 1, 1, -1, 1, -1, -1, 1]])
    weights = Q.T@B
    coord = []
    for m, n, p in product([1, 0], [1, 0], [1,0]):
    # for m, n, p in product([0, 1], [0, 1], [0, 1]):
        iref = i - m
        jref = j - n
        kref = k - p
        val_ = np.array([iref, jref, kref])
        loc_ = [iter_ for iter_, item in enumerate(reference) if np.all(val_ == item)]
        coord.append(loc_[0])
        # print(loc_[0])

    return weights, coord
    # return the corresponding indicies

def searchsorted(l, x):
    for iter_, i in enumerate(l):
        if i > x: break
    return iter_
