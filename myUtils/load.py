# coding:utf-8
import numpy as np
import os, getpass
import cPickle

datasets_dir = '/home/' + getpass.getuser() + '/dataset'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


# 数据格式为2D矩阵（样本数，图像行数*图像列数）
def mnist(onehot=False):
    data_dir = os.path.join(datasets_dir, 'mnist')

    def load_mnist_images(filename):
        filename = os.path.join(data_dir, filename)
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def load_mnist_labels(filename):
        filename = os.path.join(data_dir, filename)
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    tr_X = load_mnist_images('train-images.idx3-ubyte')
    tr_y = load_mnist_labels('train-labels.idx1-ubyte')
    te_X = load_mnist_images('t10k-images.idx3-ubyte')
    te_y = load_mnist_labels('t10k-labels.idx1-ubyte')
    if onehot:
        tr_y = one_hot(tr_y, 10)
        te_y = one_hot(te_y, 10)
    else:
        tr_y = np.asarray(tr_y)
        te_y = np.asarray(te_y)
    return tr_X, te_X, tr_y, te_y


# 数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
def cifar(onehot=False):
    data_dir = os.path.join(datasets_dir, 'cifar10', 'cifar-10-batches-py')
    allFiles = os.listdir(data_dir)
    trFiles = [f for f in allFiles if f.startswith('data_batch')]
    tr_X = []
    tr_y = []
    for file in trFiles:
        fd = open(os.path.join(data_dir, file))
        dict = cPickle.load(fd)
        batchData = dict['data'].reshape(-1, 3, 32, 32)
        batchLabel = dict['labels']
        tr_X.append(batchData)
        tr_y.extend(batchLabel)
        fd.close()
    tr_X = np.vstack(tr_X)
    teFiles = [f for f in allFiles if f.find('test_batch') != -1]
    te_X = []
    te_y = []
    for file in teFiles:
        fd = open(os.path.join(data_dir, file))
        dict = cPickle.load(fd)
        batchData = dict['data'].reshape(-1, 3, 32, 32)
        batchLabel = dict['labels']
        te_X.append(batchData)
        te_y.extend(batchLabel)
        fd.close()
    te_X = np.vstack(te_X)
    if onehot:
        tr_y = one_hot(tr_y, 10)
        te_y = one_hot(te_y, 10)
    else:
        tr_y = np.asarray(tr_y)
        te_y = np.asarray(te_y)
    return tr_X, te_X, tr_y, te_y
