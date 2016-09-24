import lasagne
import theano
from theano import tensor as T
import numpy as np
import cPickle

# size=50
# batchSize=3
# startRange = range(0, size - batchSize+1, batchSize)
# endRange = range(batchSize, size+1, batchSize)
# if size % batchSize != 0:
#     startRange.append(size - size % batchSize)
#     endRange.append(size)
# print startRange
# print endRange
# for start, end in zip(startRange, endRange):
#     print slice(start, end)

# class DotLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, num_units, W=lasagne.init.Normal(0.01), **kwargs):
#         super(DotLayer, self).__init__(incoming, **kwargs)
#         num_inputs = self.input_shape[1]
#         self.num_units = num_units
#         self.W = self.add_param(W, (num_inputs, num_units), name='W')
#
#     def get_output_for(self, input, **kwargs):
#         return T.dot(input, self.W)
#
#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], self.num_units)
#
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# _srng = RandomStreams()
#
# class DropoutLayer(lasagne.layers.Layer):
#     def __init__(self, incoming, p=0.5, **kwargs):
#         super(DropoutLayer, self).__init__(incoming, **kwargs)
#         self.p = p
#
#     def get_output_for(self, input, deterministic=False, **kwargs):
#         if deterministic:  # do nothing in the deterministic case
#             return input
#         else:  # add dropout noise otherwise
#             retain_prob = 1 - self.p
#             input /= retain_prob
#             return input * _srng.binomial(input.shape, p=retain_prob,
#                                           dtype=theano.config.floatX)

# x=np.arange(6).reshape((2,3))
# y=np.arange(30).reshape((3,10))
#
# print np.dot(x,y)
#
# a=np.dot(x,y).transpose().reshape((5,2,2))
# a=np.max(a,axis=1).transpose()
# print a
#
# b=np.dot(x,y).reshape((2,5,2))
# b=np.max(b,axis=2)
# print b


from pylearn2.scripts.datasets import make_cifar10_whitened,make_cifar10_gcn_whitened
