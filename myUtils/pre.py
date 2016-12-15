# coding:utf-8

import theano
import numpy as np
import os
import cPickle
import skimage.transform
from lasagne.utils import floatX

'''
图像预处理，只进行零均值化和归一化，在训练集上计算RGB三个通道每个位置的均值，分别在训练、验证、测试集上减去
不用归一化有时候会出现nan，即计算的数值太大
如果要使用标准差归一化，要注意有的位置上标准差为0，容易产生nan
'''
epsilon = 1e-3


def norm2d(tr_X, vate_X):
    avg = np.mean(tr_X, axis=None, dtype=theano.config.floatX, keepdims=True)
    # var = np.var(tr_X, axis=None, dtype=theano.config.floatX, keepdims=True)
    return (tr_X - avg) / 127.5, (vate_X - avg) / 127.5


# def norm4d(tr_X, vate_X):
#     avg = np.mean(tr_X, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
#     # var = np.var(tr_X, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
#     return (tr_X - avg) / 127.5, (vate_X - avg) / 127.5

def norm4d(X, avg=None, var=None):
    if avg is None and var is None:
        avg = np.mean(X, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
        var = np.var(X, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
        return (X - avg) / (np.sqrt(var + epsilon)), avg, var
    else:
        return (X - avg) / (np.sqrt(var + epsilon))


# pylearn2
def norm4d_per_sample(X, scale=1., reg=0.1):
    raw_shape = X.shape
    X = X.reshape((X.shape[0], -1))
    mean = X.mean(axis=1)
    X = X - mean[:, np.newaxis]
    normalizer = np.sqrt((X ** 2).mean(axis=1) + reg) / scale
    X = X / normalizer[:, np.newaxis]
    return X.reshape(raw_shape)


def cifarWhiten(name='cifar10'):
    PYLEARN2_DATA_PATH = '/home/zhufenghao/dataset'
    whitenedData = os.path.join(PYLEARN2_DATA_PATH, name, 'pylearn2_gcn_whitened')
    trainFile = os.path.join(whitenedData, 'train.pkl')
    testFile = os.path.join(whitenedData, 'test.pkl')
    train = cPickle.load(open(trainFile, 'r'))
    test = cPickle.load(open(testFile, 'r'))
    tr_X, tr_y = train.get_data()
    te_X, te_y = test.get_data()
    tr_X = tr_X.reshape((-1, 3, 32, 32))
    te_X = te_X.reshape((-1, 3, 32, 32))
    tr_y = tr_y.squeeze()
    te_y = te_y.squeeze()
    return tr_X, te_X, tr_y, te_y


def imagenetPre(im, MEAN_IMAGE):
    h, w, _ = im.shape  # Resize so smallest dim = 256, preserving aspect ratio
    if h < w:
        im = skimage.transform.resize(im, (256, w * 256 / h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h * 256 / w, 256), preserve_range=True)
    h, w, _ = im.shape
    im = im[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112]  # Central crop to 224x224
    rawim = np.copy(im).astype('uint8')
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)  # Shuffle axes to c01
    im = im[::-1, :, :]  # Convert to BGR
    im = im - np.reshape(MEAN_IMAGE, (-1, 1, 1))
    return rawim, floatX(im[np.newaxis])
