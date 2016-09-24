# coding:utf-8

import theano
import numpy as np
import os
import cPickle

'''
图像预处理，只进行零均值化和归一化，在训练集上计算RGB三个通道每个位置的均值，分别在训练、验证、测试集上减去
不用归一化有时候会出现nan，即计算的数值太大
如果要使用标准差归一化，要注意有的位置上标准差为0，容易产生nan
'''
epsilon = 1e-8


def norm2d(tr_X, vate_X):
    avg = np.mean(tr_X, axis=None, dtype=theano.config.floatX, keepdims=True)
    var = np.var(tr_X, axis=None, dtype=theano.config.floatX, keepdims=True)
    return (tr_X - avg) / np.sqrt(var + epsilon), (vate_X - avg) / np.sqrt(var + epsilon)


def norm4d(tr_X, vate_X):
    avg = np.mean(tr_X, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
    var = np.var(tr_X, axis=(0, 2, 3), dtype=theano.config.floatX, keepdims=True)
    return (tr_X - avg) / np.sqrt(var + epsilon), (vate_X - avg) / np.sqrt(var + epsilon)


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
    return tr_X, te_X, tr_y, te_y
