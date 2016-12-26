# coding:utf-8

import lasagne
import theano
from theano import tensor as T
import numpy as np
import cPickle
import os
import myUtils

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

# def listFiles(path, numPerDir=None):
#     fileList = []
#     try:
#         dirs=os.listdir(path)
#     except Exception:
#         return []
#     dirs=dirs[:numPerDir]
#     for n,file in enumerate(dirs):
#         subFile=os.path.join(path,file)
#         if os.path.isdir(subFile):
#             fileList.extend(listFiles(subFile, numPerDir))
#         else:
#             fileList.append(subFile)
#     return fileList
#
# for file in listFiles('/home/zfh/PycharmProjects',3):
#     print file

# from theano.tensor.signal.conv import conv2d
# x=np.arange(50).reshape(2,5,5)
# f=np.ones((3,3))
#
# X=T.tensor3()
# F=theano.shared(f)
# out=conv2d(X,F,border_mode='valid')
# func=theano.function([X],out)
# print x
# print func(x)

# import numpy as np
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import skcuda.magma as magma
# np.random.seed(0)
# n = 1000
#
# # Note that the matrices have to either be stored in column-major form,
# # otherwise MAGMA will compute the solution for the system dot(a.T, x) == b:
# a = np.asarray(np.random.rand(n, n), order='F')
# b = np.asarray(np.random.rand(n), order='F')
#
# # Compute solution on the CPU:
# x = np.linalg.solve(a, b)
#
# # Compute the solution on the GPU - b_gpu is subsequently overwritten with the solution:
# a_gpu = gpuarray.to_gpu(a)
# b_gpu = gpuarray.to_gpu(b)
# magma.magma_init()
# magma.magma_dgesv_nopiv_gpu(n, 1, int(a_gpu.gpudata), n, int(b_gpu.gpudata), n)
# magma.magma_finalize()
#
# # Check that the solutions match:
# print np.allclose(x, b_gpu.get())

# def softmax(X):
#     e_x = np.exp(X - X.max(axis=1).reshape(-1,1))
#     return e_x / e_x.sum(axis=1).reshape(-1,1)
#
# x=np.random.randn(10,5)
# tinyfloat = np.finfo(theano.config.floatX).tiny
# Tmat = np.full_like(x, np.log(tinyfloat), dtype=theano.config.floatX, order='A')
# Tmat[np.arange(len(x)),np.argmax(x, axis=1)] = 0
# print x,Tmat
# print np.asarray(softmax(Tmat),np.float32)

# import myUtils
# import pylab as plt
# from numpy.linalg import pinv, solve
# tr_X, te_X, tr_y, te_y = myUtils.load.mnist()
# img = tr_X[0, 0, :, :].reshape(1, 784)
# w=np.random.randn(784, 784)
# y=img.dot(w)
# wpinv=pinv(w.T)
# xrec=(wpinv.dot(y.T)).T
# # xrec=solve(w.T,y.T).T
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(img.reshape(28,28))
# plt.subplot(1,2,2)
# plt.imshow(xrec.reshape(28,28))
# plt.show()

# from lasagne import layers
# x = T.tensor4()
# l_in = layers.InputLayer((2,2,3, 3), input_var=x)
# # gp = layers.GlobalPoolLayer(l_in, T.mean)
# # gp=layers.NonlinearityLayer(l_in,nonlinearity=lasagne.nonlinearities.softmax)
# # gpinv=layers.InverseLayer(gp,gp)
# # out1 = lasagne.layers.get_output(gp)
# # out2 = lasagne.layers.get_output(gpinv)
#
# uppool=layers.Upscale2DLayer(l_in,scale_factor=2)
# out1=lasagne.layers.get_output(uppool)
#
# func1 = theano.function([x], out1)
# a=np.random.randn(2,2,3,3)
# b=func1(a)
#
# print a
# print b

# from lasagne.theano_extensions.padding import pad
#
# x = T.tensor4()
# y = T.tensor4()
# index = myUtils.basic.patchIndex((10, 1, 5, 5), (3, 3))
# patch = T.flatten(x, 1)[index]
# patch = patch.reshape((-1, 9))
# f = y.reshape((8, 9)).T
# out1 = T.dot(patch, f)
# out2 = T.nnet.conv2d(x, y, border_mode='valid', filter_flip=False)
# func = theano.function([x, y], [out1, out2], allow_input_downcast=True)
#
# a = np.random.randn(10, 1, 5, 5)
# b = np.random.randn(8, 1, 3, 3)
# o1, o2 = func(a, b)
# o2 = o2.transpose((0, 2, 3, 1)).reshape((-1, 8))
# print np.allclose(o1, o2)

# a = np.random.randn(3,2,5, 5)
# print np.pad(a,((0,0),(0,0),(1,1),(1,1)),mode='constant',constant_values=0).shape

# from decaf import base
# from decaf.layers import core_layers, fillers

# ourblob=base.Blob()
# inblob=base.Blob(shape=(1,5,5,1),filler=fillers.GaussianRandFiller())
# layer=core_layers.Im2colLayer(name='im2col',psize=3,stride=1)
# layer.forward([inblob],[ourblob])
# print inblob.data().transpose((0,3,1,2)), ourblob.data()

# from decaf.layers.cpp import wrapper
# def get_new_shape(features, psize, stride):
#     """Gets the new shape of the im2col operation."""
#     if features.ndim != 4:
#         raise ValueError('Input features should be 4-dimensional.')
#     num, height, width, channels = features.shape
#     return (num,
#             (height - psize) / stride + 1,
#             (width - psize) / stride + 1,
#             channels * psize * psize)
#
# features=np.arange(50,dtype='float').reshape(2,5,5,1)
# print features.dtype
# shape=get_new_shape(features, 4, 2)
# output=np.zeros(shape,dtype='float')
# wrapper.im2col_forward(features, output, 4, 2)
# print features.transpose((0,3,1,2)), output

# features=np.arange(50).reshape(2,1,5,5)
# im2col=myUtils.basic.Im2ColOp(psize=3, stride=1)
# a=im2col.transform(features).reshape((-1,9))
# index=myUtils.basic.patchIndex((2,1,5,5),(3,3),(1,1))
# b=features.flat[index].reshape((-1,9))
# print a,b

# import pylearn2.scripts.datasets.make_cifar10_gcn_whitened
# from keras.preprocessing.image import ImageDataGenerator

# a=np.random.randn(10,8)
# b=np.random.randn(8,10)
# c1=a.dot(b)
# asplit=np.split(a,2,axis=1)
# bsplit=np.split(b,2,axis=0)
# c2=asplit[0].dot(bsplit[0])+asplit[1].dot(bsplit[1])
# print np.allclose(c1,c2)

# xnp=np.random.randn(20,150)
# ynp=np.random.randn(150,20)
# subsize=13
# maxdim = np.max(xnp.shape)
# parts = maxdim // subsize + 1  # 分块矩阵的分块数
# index = [subsize * i for i in range(1, parts)]
# print index
# xparts = np.split(xnp, index, axis=1)
# yparts = np.split(ynp, index, axis=0)
# partsum = []
# for x, y in zip(xparts, yparts):
#     partsum.append(np.dot(x, y))
# print np.allclose(sum(partsum), np.dot(xnp,ynp))

# def convactfn(pad, stride):
#     xt = T.ftensor4()
#     ft = T.ftensor4()
#     convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=stride, filter_flip=False)
#     convxf = T.tanh(convxf)
#     conv2d = theano.function([xt, ft], convxf, allow_input_downcast=True)
#     return conv2d
#
# def dotfn():
#     xt = T.fmatrix()
#     yt = T.fmatrix()
#     dotxy = T.dot(xt, yt)
#     dotact = theano.function([xt, yt], dotxy, allow_input_downcast=True)
#     return dotact
#
#
# dot = dotfn()
# conv=convactfn(3, (1,1))
#
# filters=np.random.randn(49,32)
# image=np.random.randn(2,2,28,28)
# output=0.
# for ch in range(2):
#     oneChannel = image[:, ch, :, :].reshape((2, 1, 28, 28))
#     padding = ((0, 0), (0, 0), (3, 3), (3, 3))
#     oneChannel = np.pad(oneChannel, padding, mode='constant', constant_values=0)
#     im2col = myUtils.basic.Im2ColOp(psize=7, stride=1)
#     patches = im2col.transform(oneChannel)
#     patches = patches.reshape((-1, 49))
#     output+=dot(patches,filters).reshape((2, 28, 28, 32)).transpose((0, 3, 1, 2))
# output=np.tanh(output)
#
# filters=filters.reshape((7,7,1,32)).transpose((3,2,0,1))
# convout=conv(image,filters)
#
# print np.allclose(output,convout)

# def get(inputX):
#     im2col = myUtils.basic.Im2ColOp(psize=3, stride=1)
#     out = im2col.transform(inputX).reshape((-1, 9))
#     return out
#
# features=np.arange(375).reshape(5,3,5,5)
#
# for i in range(3):
#     onechannel =features[:,i,:,:].reshape(5,1,5,5)
#     print onechannel, get(onechannel)
# index=myUtils.basic.patchIndex((5,3,5,5),(3,3),(1,1))
# b=features.flat[index].reshape((-1,9))
# print b

# from theano.sandbox.neighbours import images2neibs
# images = T.tensor4('images')
# neibs = images2neibs(images, neib_shape=(3, 3), neib_step=(1,1))
# f = theano.function([images], neibs)
# features=np.arange(225).reshape(3,3,5,5)
# l=[]
# for i in range(3):
#     l.append(f(features[:,i,:,:].reshape((3,1,5,5))))
# print np.concatenate(l), f(features)

# a=np.arange(200).reshape((10,5,2,2))
# index1=np.arange(10)
# index2=np.random.randint(5,size=10)
# print a[index1,index2,...]

# from sklearn.linear_model import MultiTaskLasso, Lasso, LassoLars
#
# rng = np.random.RandomState(42)
#
# # Generate some 2D coefficients with sine waves with random frequency and phase
# n_samples, n_features, n_tasks = 100, 30, 40
# n_relevant_features = 5
# coef = np.zeros((n_tasks, n_features))
# times = np.linspace(0, 2 * np.pi, n_tasks)
# for k in range(n_relevant_features):
#     coef[:, k] = np.sin((1. + rng.randn(1)) * times + 3 * rng.randn(1))
#
# X = rng.randn(n_samples, n_features)
# Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)
#
# # coef_lasso_ = np.array([Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T])
# coef_lasso_2 = Lasso(alpha=0.5).fit(X, Y).coef_
# coef_lasso_3 = LassoLars(alpha=0.01).fit(X, Y).coef_
# coef_lasso_4 = LassoLars(alpha=0.1).fit(X, Y).coef_
# coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.).fit(X, Y).coef_
# print coef_lasso_2,coef_lasso_3,coef_lasso_4, coef_multi_task_lasso_

# a=np.arange(10)
# a=np.tile(a,5).reshape((5,10))
# map(np.random.shuffle,a)
# print a

# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
# distances, indices = nbrs.kneighbors()
# print indices

# a=np.arange(5,25)
# i=np.random.randint(20, size=(5,3))
# x=np.take(a,i)
# y=x-np.repeat(x[:,0].reshape((-1,1)),3,axis=1)
# y[np.where(y!=0)]=-1
# aa=np.where(y!=0)
# print y

# w=np.zeros((5,5))
# index=np.arange(5)
# index=np.repeat(index,3,axis=0)
# nn=np.random.randint(5,size=15)
# graph=np.arange(15)
# w[index,nn] = graph
# print index, nn,graph,w


# from scipy.sparse import csr_matrix,dok_matrix
# a=dok_matrix((10,10))
# idx1=np.random.randint(10,size=15)
# idx2=np.random.randint(10, size=15)
# a[idx1,idx2]=1
# a[idx1,idx2]=1
# b=dok_matrix((10,10))
# b[idx1,idx2]=1
# b[idx1,idx2]=1
# print a,b,a.dot(b)

# def cartesian(arrays, out=None):
#     """
#     Generate a cartesian product of input arrays.
#
#     Parameters
#     ----------
#     arrays : list of array-like
#         1-D arrays to form the cartesian product of.
#     out : ndarray
#         Array to place the cartesian product in.
#
#     Returns
#     -------
#     out : ndarray
#         2-D array of shape (M, len(arrays)) containing cartesian products
#         formed of input arrays.
#
#     Examples
#     --------
#     >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
#     array([[1, 4, 6],
#            [1, 4, 7],
#            [1, 5, 6],
#            [1, 5, 7],
#            [2, 4, 6],
#            [2, 4, 7],
#            [2, 5, 6],
#            [2, 5, 7],
#            [3, 4, 6],
#            [3, 4, 7],
#            [3, 5, 6],
#            [3, 5, 7]])
#
#     """
#
#     arrays = [np.asarray(x) for x in arrays]
#     dtype = arrays[0].dtype
#
#     n = np.prod([x.size for x in arrays])
#     if out is None:
#         out = np.zeros([n, len(arrays)], dtype=dtype)
#
#     m = n / arrays[0].size
#     out[:,0] = np.repeat(arrays[0], m)
#     if arrays[1:]:
#         cartesian(arrays[1:], out=out[0:m,1:])
#         for j in xrange(1, arrays[0].size):
#             out[j*m:(j+1)*m,1:] = out[0:m,1:]
#     return out
#
# idx= cartesian([np.array([1,2,3]),np.array([1,2,3])])

# a=np.arange(10).reshape((5,2))
# print (a>3)*(a<8)
# print np.where((a>3) *(a<8))

# a=np.random.uniform(size=(15,15))
# index=np.where(a<0.1)
# idx1=index[0]
# idx2=index[1]
# print idx1
# print idx2
# length=len(idx1)
#
# idx1=np.tile(np.repeat(idx1,3),3)
# tmp1=np.repeat(np.array([0,1,2]),length*3)
# idx1+=tmp1
#
# idx2=np.repeat(np.tile(idx2, 3),3)
# tmp2=np.tile(np.array([0,1,2]),length*3)
# idx2+=tmp2
#
# print np.vstack((idx1,idx2))
#
# bond= np.where((idx2>=15)+(idx1>=15))
# idx1, idx2= np.delete(idx1, bond), np.delete(idx2,bond)
# a[idx1,idx2]=0
# print a

# def mask(row_idx, col_idx, row_size, col_size):
#     assert len(row_idx)==len(col_idx)
#     length = len(row_idx)
#     row_idx = np.tile(np.repeat(row_idx, col_size), row_size)
#     bias = np.repeat(np.arange(row_size), length * col_size)
#     row_idx += bias
#     col_idx = np.repeat(np.tile(col_idx, row_size), col_size)
#     bias = np.tile(np.arange(col_size), length * row_size)
#     col_idx += bias
#     return row_idx, col_idx
#
# block_row_size1=3
# block_col_size2=4
# X=np.ones((5,1,15,15))
# uniform = np.random.uniform(low=0., high=1., size=X.shape)
# index = np.where((uniform > 0) * (uniform <= 0.01))
# index_b_ch = map(lambda x: np.repeat(np.tile(x, block_row_size1), block_col_size2), index[:2])
# index_r, index_c = mask(index[2], index[3], block_row_size1, block_col_size2)
# out_of_bond = np.where((index_r >= 15) + (index_c >= 15))
# index = map(lambda x: np.delete(x, out_of_bond), index_b_ch + [index_r, index_c])
# X[index] = 0.
# print X

# from theano.sandbox.neighbours import images2neibs
# from lasagne.theano_extensions.padding import pad as lasagnepad
# def im2col(inputX, fsize, stride, pad):
#     assert inputX.ndim == 4
#     Xrows, Xcols = inputX.shape[-2:]
#     X = T.tensor4()
#     if pad is None:  # 保持下和右的边界
#         rowpad = colpad = 0
#         rowrem = (Xrows - fsize) % stride
#         if rowrem: rowpad = stride - rowrem
#         colrem = (Xcols - fsize) % stride
#         if colrem: colpad = stride - colrem
#         pad = ((0, rowpad), (0, colpad))
#     Xpad = lasagnepad(X, pad, batch_ndim=2)
#     neibs = images2neibs(Xpad, (fsize, fsize), (stride, stride), 'ignore_borders')
#     im2colfn = theano.function([X], neibs, allow_input_downcast=True)
#     return im2colfn(inputX)
#
# x=np.arange(100).reshape((2,2,5,5))
# a=im2col(x,3,1,None)
# print a

# for i in range(28):
#     print np.ceil(1.414*(i+0.5)),np.ceil(1.414*(i+1.5))

# def maxpool(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0)):
#     batch, channel, img_row, img_col = xnp.shape
#     if stride is None:
#         stride = pool_size
#     if pad != (0, 0):
#         img_row += 2 * pad[0]
#         img_col += 2 * pad[1]
#         ynp = np.zeros((xnp.shape[0], xnp.shape[1], img_row, img_col), dtype=xnp.dtype)
#         ynp[:, :, pad[0]:(img_row - pad[0]), pad[1]:(img_col - pad[1])] = xnp
#         print ynp
#     else:
#         ynp = xnp
#     fsize = (channel, channel) + pool_size
#     out_shape = myUtils.basic.conv_out_shape(xnp.shape, fsize, pad, stride)
#     out_shape = list(out_shape)
#     if not ignore_border:
#         out_shape[2] += (img_row - pool_size[0]) % stride[0]
#         out_shape[3] += (img_col - pool_size[1]) % stride[1]
#     out = np.empty(out_shape, dtype=xnp.dtype)
#     for b in xrange(batch):
#         for ch in xrange(channel):
#             for r in xrange(out_shape[2]):
#                 row_start = r * stride[0]
#                 row_end = min(row_start + pool_size[0], img_row)
#                 for c in xrange(out_shape[3]):
#                     col_start = c * stride[1]
#                     col_end = min(col_start + pool_size[1], img_col)
#                     patch = ynp[b, ch, row_start:row_end, col_start:col_end]
#                     out[b, ch, r, c] = np.max(patch)
#     return out
#
#
# x = np.arange(196).reshape((2, 2, 7, 7))
# print x, maxpool(x, (2, 2), True)

# def fmp_overlap(xnp, pool_ratio, constant, overlap=True):
#     batch, channel, img_row, img_col = xnp.shape
#     out_row = int((float(img_row) / pool_ratio))
#     out_col = int((float(img_col) / pool_ratio))
#     row_idx = [int(pool_ratio * (i + constant)) for i in xrange(out_row + 1)]
#     if row_idx[-1] != img_row:
#         row_idx.append(img_row)
#         out_row += 1
#     row_idx = np.array(row_idx, dtype=np.int)
#     col_idx = [int(pool_ratio * (i + constant)) for i in xrange(out_col + 1)]
#     if col_idx[-1] != img_col:
#         col_idx.append(img_col)
#         out_col += 1
#     col_idx = np.array(col_idx, dtype=np.int)
#     out_shape = (batch, channel, out_row, out_col)
#     out = np.empty(out_shape, dtype=xnp.dtype)
#     for b in xrange(batch):
#         for ch in xrange(channel):
#             for r in xrange(out_row):
#                 row_start = row_idx[r]
#                 row_end = row_idx[r + 1] + 1 if overlap else row_idx[r + 1]
#                 for c in xrange(out_col):
#                     col_start = col_idx[c]
#                     col_end = col_idx[c + 1] + 1 if overlap else col_idx[c + 1]
#                     patch = xnp[b, ch, row_start:row_end, col_start:col_end]
#                     out[b, ch, r, c] = np.max(patch)
#     return out
#
#
# x = np.arange(484).reshape((2, 2, 11, 11))
# print x, fmp_overlap(x, 1.414, 0.5, True)

# # Sparse
# def sparse_random(input_unit, hidden_unit):
#     filters = np.random.uniform(low=0, high=1, size=(input_unit, hidden_unit))
#     neg_idx = np.where((filters >= 0) * (filters < 1. / 6.))
#     zero_idx = np.where((filters >= 1. / 6.) * (filters < 5. / 6.))
#     pos_idx = np.where((filters >= 5. / 6.) * (filters < 1.))
#     filters[neg_idx] = -1.
#     filters[zero_idx] = 0.
#     filters[pos_idx] = 1.
#     ranges = np.sqrt(2.0 / input_unit)
#     bias = np.random.uniform(low=-ranges, high=ranges, size=hidden_unit)
#     return filters, bias

x = 22
for i in range(10):
    x = int(float(x) / 2.1)
    print x

# x=np.random.rand(10,5)
# c=np.cov(x,rowvar=0)
# y=x-x.mean(0)
# c1=y.T.dot(y)/(10-1)
# print c, c1

# def partition_channels(channels, n_jobs):
#     if n_jobs < 0:
#         n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
#     n_jobs = min(n_jobs, channels)
#     n_channels_per_job = (channels // n_jobs) * np.ones(n_jobs, dtype=np.int)
#     n_channels_per_job[:channels % n_jobs] += 1
#     starts = np.cumsum(n_channels_per_job)
#     return n_jobs, [0] + starts.tolist()
#
#
# def job(inputX, func, n_hidden, filter_size, stride, pool_size):
#     batches, channels, rows, cols = inputX.shape
#     _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
#                                                       (n_hidden, channels, filter_size, filter_size),
#                                                       pad=filter_size // 2, stride=stride)
#     output = 0.
#     filters = []
#     for ch in xrange(channels):
#         oneChannel = inputX[:, [ch], :, :]
#         beta = func(oneChannel)
#         patches = im2col(oneChannel, filter_size, pad=filter_size // 2, stride=stride)
#         patches = patches.reshape((batches, orows, ocols, -1))
#         output += dot_pool(patches, beta, pool_size)
#         filters.append(beta)
#     return output, filters

# n_jobs, starts = partition_channels(inputX.shape[1], -1)
# result = Parallel(n_jobs=n_jobs)(
#     delayed(job)(inputX[:, starts[i]:starts[i + 1], :, :], self._get_beta,
#                  self.n_hidden, self.filter_size, self.stride, self.pool_size)
#     for i in xrange(n_jobs))

# x= np.random.rand(10,10,3,3)
# y=np.sum(x,axis=(0,2,3))
# idx=np.argsort(y)
# idx1=idx[-5:]
# # z=np.array([x[b,ch] for b,ch in enumerate(idx)])
# # a=np.sum(z,axis=(2,3))
# # zz=np.array([x[b,ch] for b,ch in enumerate(idx1)])
# # aa=np.sum(zz,axis=(2,3))
# # print a,aa
# print idx,y

# def split_map(x, split_size):
#     assert x.ndim == 4
#     batches, channels, rows, cols = x.shape
#     splited_rows = int(np.ceil(float(rows) / split_size))
#     splited_cols = int(np.ceil(float(cols) / split_size))
#     rowpad = splited_rows * split_size - rows
#     colpad = splited_cols * split_size - cols
#     pad = (0, rowpad, 0, colpad)
#     x = myUtils.basic.pad2d(x, pad)
#     result = []
#     for i in xrange(split_size):
#         for j in xrange(split_size):
#             result.append(x[:, :,
#                           i * splited_rows:(i + 1) * splited_rows,
#                           j * splited_cols:(j + 1) * splited_cols])
#     return result
#
# def join_map(x_list, split_size):
#     result=[]
#     for i in xrange(split_size):
#         result.append(np.concatenate(x_list[i*split_size:(i+1)*split_size], axis=3))
#     return np.concatenate(result, axis=2)
#
# def split_neib(x, split_size):
#     assert x.ndim == 4
#     batches, channels, rows, cols = x.shape
#     splited_rows = int(np.ceil(float(rows) / split_size))
#     splited_cols = int(np.ceil(float(cols) / split_size))
#     rowpad = splited_rows * split_size - rows
#     colpad = splited_cols * split_size - cols
#     pad = (0, rowpad, 0, colpad)
#     x = myUtils.basic.pad2d(x, pad)
#     result = []
#     for i in xrange(split_size):
#         for j in xrange(split_size):
#             result.append(x[:, :, i::split_size, j::split_size])
#     return result
#
#
# def join_neib(x_list, split_size):
#     result = []
#     for i in xrange(split_size):
#         x_row = x_list[i * split_size:(i + 1) * split_size]
#         x_row = map(lambda x: x[:, :, :, np.newaxis, :, np.newaxis], x_row)
#         x_row = np.concatenate(x_row, axis=5)
#         x_row = x_row.reshape(x_row.shape[:-2] + (-1,))
#         result.append(x_row)
#     result = np.concatenate(result, axis=3)
#     result = result.reshape(result.shape[:2] + (-1, result.shape[-1]))
#     return result
#
# x=np.arange(100*2).reshape((2,1,10,10))
# y=split_neib(x, 3)
# print x,y
# z=join_neib(y, 3)
# print z

# import pylearn2.scripts.datasets.make_cifar10_gcn_whitened

# x=np.array([[1,2,3,4,5,6,7,8],[0,0,0,0,0,0,0,0]],dtype='float')
# print np.var(x,axis=1),((x-x.mean(axis=1)[:,np.newaxis]) ** 2).mean(axis=1)

# X=np.random.randn(10,5)
# cov = np.dot(X.T, X) / X.shape[0]
# D, V = np.linalg.eig(cov)
# P = V.dot(np.diag(np.sqrt(1 / (D + 0.1)))).dot(V.T)
# print D,V,P
#
# X/=10
# cov = np.dot(X.T, X) / X.shape[0]
# D, V = np.linalg.eig(cov)
# P = V.dot(np.diag(np.sqrt(1 / (D + 0.1)))).dot(V.T)
# print D,V,P

# def _get_block_idx(blockr, blockc,orows, ocols):
#     nr = int(np.ceil(float(orows) / blockr))
#     nc = int(np.ceil(float(ocols) / blockc))
#     idx = []
#     for row in xrange(nr):
#         row_bias = row * blockr
#         for col in xrange(nc):
#             col_bias = col * blockc
#             base = np.arange(blockc) if col_bias + blockc < ocols else np.arange(ocols - col_bias)
#             block_row = blockr if row_bias + blockr < orows else orows - row_bias
#             one_block = []
#             for br in xrange(block_row):
#                 one_row = base + orows * br + col_bias + row_bias * orows
#                 one_block = np.concatenate([one_block, one_row]) if len(one_block) != 0 else one_row
#             idx.append(one_block)
#     return idx
#
# def join_result(blocks, blockr, blockc, orows, ocols):
#     batches = blocks[0].shape[0] / (blockr * blockc)
#     channels = blocks[0].shape[1]
#     nr = int(np.ceil(float(orows) / blockr))
#     nc = int(np.ceil(float(ocols) / blockc))
#     output = []
#     for row in xrange(nr):
#         one_row = []
#         for col in xrange(nc):
#             one_block = blocks.pop(0)
#             if col == nc - 1 and row != nr - 1:
#                 one_block = one_block.reshape((batches, blockr, -1, channels))
#             elif row == nr - 1 and col != nc - 1:
#                 one_block = one_block.reshape((batches, -1, blockc, channels))
#             elif row == nr - 1 and col == nc - 1:
#                 rsize = orows % blockr if orows % blockr else blockr
#                 csize = ocols % blockc if ocols % blockc else blockc
#                 one_block = one_block.reshape((batches, rsize, csize, channels))
#             else:
#                 one_block = one_block.reshape((batches, blockr, blockc, channels))
#             one_block = one_block.transpose((0, 3, 1, 2))
#             one_row = np.concatenate([one_row, one_block], axis=3) if len(one_row) != 0 else one_block
#         output = np.concatenate([output, one_row], axis=2) if len(output) != 0 else one_row
#     return output
#
# idx=_get_block_idx(5,5,10,10)
# print idx
# idx=map(lambda x:x[:,None],idx)
# print join_result(idx,5,5, 10,10)


# def get_rand_idx(n_rand, orows, ocols):
#     size = orows * ocols
#     split_size = int(round(float(size) / n_rand))
#     all_idx = np.random.permutation(size)
#     split_range = [split_size + split_size * i for i in xrange(n_rand - 1)]
#     split_idx = np.split(all_idx, split_range)
#     return split_idx
#
#
# print get_rand_idx(10, 7, 7)

# def get_neib_idx(neibr, neibc, orows, ocols):
#     idx = []
#     for i in xrange(neibr):
#         row_idx = np.arange(i, orows, neibr)
#         for j in xrange(neibc):
#             col_idx = np.arange(j, ocols, neibc)
#             one_neib = []
#             for row_step in row_idx:
#                 one_row = col_idx + row_step * orows
#                 one_neib = np.concatenate([one_neib, one_row]) if len(one_neib) != 0 else one_row
#             idx.append(one_neib)
#     return idx
#
# print np.arange(100).reshape((10,10)),get_neib_idx(4,4,10,10)

