# coding:utf-8
import numpy as np
from  scipy.linalg import orth
from  lasagne.init import Glorot
rng = np.random.RandomState(47)


# shape=(# feature maps, # channels, # filter rows, # filter cols)
def convfilterinit(shape):
    filters = rng.randn(*shape)
    flatShape = np.prod(shape[2:])
    filters = filters.reshape((shape[0], shape[1], flatShape))
    filters = np.transpose(filters, axes=(1, 2, 0))
    orthFilters = []
    if flatShape >= shape[0]:
        for i in xrange(shape[1]):
            orthFilters.append(orth(filters[i]))
    else:
        for i in xrange(shape[1]):
            orthFilters.append(orth(filters[i].T).T)
    orthFilters = np.array(orthFilters)
    orthFilters = np.transpose(orthFilters, axes=(2, 0, 1))
    orthFilters = orthFilters.reshape(shape)
    return orthFilters
