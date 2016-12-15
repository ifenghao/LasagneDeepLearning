# coding:utf-8
'''
基于ELM的局部感知自编码器
仿照多通道的卷积层,将上一层所有通道卷积后的特征图求和,得到下一层的一张特征图
测试stl10数据集
非监督数据最后一层不用计算输出
'''

import gc
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.neighbours import images2neibs
from numpy.linalg import solve
from scipy.linalg import orth, svd
from lasagne.theano_extensions.padding import pad as lasagnepad
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from collections import OrderedDict
from copy import deepcopy
import myUtils

compute_beta_val_times = 4


def compute_beta(Hmat, Tmat, C):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = np.dot(Hmat.T, solve(np.eye(rows) / C + np.dot(Hmat, Hmat.T), Tmat))
    else:
        beta = solve(np.eye(cols) / C + np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
    return beta


def compute_beta_val(Hmat, Tmat, n_iter):
    neg = -5.
    pos = 3.
    CList = np.random.uniform(0.1, 1.1, n_iter) * \
            10 ** np.arange(neg, pos, (pos - neg) / n_iter)
    optbeta = None
    optcost = np.inf
    for C in CList:
        beta = compute_beta(Hmat, Tmat, C)
        cost = np.mean(abs(np.dot(Hmat, beta) - Tmat))
        if cost < optcost:
            optcost = cost
            optbeta = beta
    return optbeta


def compute_beta_rand(Hmat, Tmat, C):
    Crand = abs(np.random.uniform(0.1, 1.1)) * C
    return compute_beta(Hmat, Tmat, Crand)


def orthonormalize(filters):
    ndim = filters.ndim
    if ndim != 2:
        filters = np.expand_dims(filters, axis=0)
    rows, cols = filters.shape
    if rows >= cols:
        orthonormal = orth(filters)
    else:
        orthonormal = orth(filters.T).T
    if ndim != 2:
        orthonormal = np.squeeze(orthonormal, axis=0)
    return orthonormal


# 随机投影矩阵不同于一般的BP网络的初始化,要保持和输入一样的单位方差
def normal_random(input_unit, hidden_unit):
    std = np.sqrt(2.0 / input_unit)
    return np.random.normal(loc=0, scale=std, size=(input_unit, hidden_unit)), \
           np.random.normal(loc=0, scale=std, size=hidden_unit)


def uniform_random(input_unit, hidden_unit):
    ranges = 1.
    return np.random.uniform(low=-ranges, high=ranges, size=(input_unit, hidden_unit)), \
           np.random.uniform(low=-ranges, high=ranges, size=hidden_unit)


def im2col(inputX, fsize, stride, pad):
    assert inputX.ndim == 4
    Xrows, Xcols = inputX.shape[-2:]
    X = T.tensor4()
    if pad is None:  # 保持下和右的边界
        rowpad = colpad = 0
        rowrem = (Xrows - fsize) % stride
        if rowrem: rowpad = stride - rowrem
        colrem = (Xcols - fsize) % stride
        if colrem: colpad = stride - colrem
        pad = ((0, rowpad), (0, colpad))
    Xpad = lasagnepad(X, pad, batch_ndim=2)
    neibs = images2neibs(Xpad, (fsize, fsize), (stride, stride), 'ignore_borders')
    im2colfn = theano.function([X], neibs, allow_input_downcast=True)
    return im2colfn(inputX)


def relu(x):
    return np.maximum(x, 0.)


def add_mn(X, p=0.5):
    retain_prob = 1. - p
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=theano.config.floatX)
    return X * binomial


def add_mn_row(X, p=0.5, reduced=False):
    if reduced:  # 由于是按行mask,比例可能需要减少
        p = p / X.shape[1]
    retain_prob = 1. - p
    binomial = np.random.uniform(low=0., high=1., size=(X.shape[0], 1))
    binomial = np.asarray(binomial < retain_prob, dtype=theano.config.floatX)
    return X * binomial


# def block_mask(index, row_size, col_size):
#     assert len(index) == 4
#     batch_idx, channel_idx, row_idx, col_idx = index
#     # 样本和通道简单重复
#     batch_idx = np.repeat(np.tile(batch_idx, row_size), col_size)
#     channel_idx = np.repeat(np.tile(channel_idx, row_size), col_size)
#     # 行列计算相邻索引
#     length = len(row_idx)
#     row_idx = np.tile(np.repeat(row_idx, col_size), row_size)
#     bias = np.repeat(np.arange(row_size), length * col_size)
#     row_idx += bias
#     col_idx = np.repeat(np.tile(col_idx, row_size), col_size)
#     bias = np.tile(np.arange(col_size), length * row_size)
#     col_idx += bias
#     return batch_idx, channel_idx, row_idx, col_idx
#
#
# def add_mn_block(X, p=0.5, blocks=((1, 1),)):
#     assert X.ndim == 4
#     row_bond, col_bond = X.shape[-2:]
#     block_types = len(blocks)
#     interval = 1. / block_types
#     pequal = p / sum(map(lambda x: np.prod(x), blocks))
#     uniform = np.random.uniform(low=0., high=1., size=X.shape)
#     start = 0.
#     for block_row_size, block_col_size in blocks:
#         index = np.where((uniform > start) * (uniform <= start + pequal))
#         index_mask = block_mask(index, block_row_size, block_col_size)
#         out_of_bond = np.where((index_mask[2] >= row_bond) + (index_mask[3] >= col_bond))
#         index_mask = map(lambda x: np.delete(x, out_of_bond), index_mask)
#         X[index_mask] = 0.
#         start += interval
#     return X


def add_sp(X, p=0.5):
    Xmin, Xmax = np.min(X), np.max(X)
    uniform = np.random.uniform(low=0., high=1., size=X.shape)
    salt = np.where((uniform > 0) * (uniform <= p / 2.))
    pepper = np.where((uniform > p / 2.) * (uniform <= p))
    X[salt] = Xmin
    X[pepper] = Xmax
    return X


def add_gs(X, std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / 200.
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    X += normal
    return X


def add_gs_part(X, p=0.5, std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / 20.
    retain_prob = 1. - p
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=theano.config.floatX)
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    normal *= binomial
    X += normal
    return X


def add_mn_gs(X, p=0.5, std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / 20.
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    X += normal
    retain_prob = 1. - p
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=theano.config.floatX)
    X *= binomial
    return X


def poolfn(pool_size, ignore_border, stride, pad, mode):
    xt = T.tensor4()
    poolx = pool_2d(xt, pool_size, ignore_border=ignore_border, st=stride, padding=pad, mode=mode)
    pool = theano.function([xt], poolx, allow_input_downcast=True)
    return pool


def pool_decomp(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
    pool = poolfn(pool_size, ignore_border, stride, pad, mode)
    size = xnp.shape[0]
    batchSize = size
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(pool(xnp[start:end]))
    return np.concatenate(result, axis=0)


def fpfn(pool_ratio, constant, overlap, mode):
    xt = T.tensor4()
    fpx = myUtils.pool.fp(xt, pool_ratio, constant, overlap, mode)
    fp = theano.function([xt], fpx, allow_input_downcast=True)
    return fp


def fp_cpu(xnp, pool_ratio, constant=0.5, overlap=True, mode='max'):
    fmp = fpfn(pool_ratio, constant, overlap, mode)
    return fmp(xnp)


# def norm2d(X, regularization=10):
#     # 对每一个patch里的元素去均值归一化
#     size = X.shape[0]
#     batchSize = size // 100
#     startRange = range(0, size - batchSize + 1, batchSize)
#     endRange = range(batchSize, size + 1, batchSize)
#     if size % batchSize != 0:
#         startRange.append(size - size % batchSize)
#         endRange.append(size)
#     result = []
#     for start, end in zip(startRange, endRange):
#         Xtmp = X[start:end]
#         mean = np.mean(Xtmp, axis=1, keepdims=True)
#         var = np.var(Xtmp, axis=1, keepdims=True)
#         result.append((Xtmp - mean) / np.sqrt(var + regularization))
#     return np.concatenate(result, axis=0)
def norm2d(X, regularization=10):
    # 对每一个patch里的元素去均值归一化
    mean = np.mean(X, axis=1, keepdims=True)
    var = np.var(X, axis=1, keepdims=True)
    return (X - mean) / np.sqrt(var + regularization)


def whiten(X, P=None):
    if P is None:
        [D, V] = np.linalg.eig(np.cov(X, rowvar=0))
        P = V.dot(np.diag(np.sqrt(1 / (D + 0.1)))).dot(V.T)
        X = X.dot(P)
        return X, P
    else:
        X = X.dot(P)
        return X


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


def decomp(oneChannel, beta, layer, output):
    assert oneChannel.ndim == 4 and oneChannel.shape[1] == 1
    batches, channels, rows, cols = oneChannel.shape
    _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, 1, rows, cols),
                                                      (layer.n_hidden, 1, layer.filter_size, layer.filter_size),
                                                      pad=layer.pad, stride=layer.stride)
    pool = poolfn(layer.pool_size, ignore_border=False, stride=None, pad=(0, 0), mode=layer.mode)
    size = oneChannel.shape[0]
    batchSize = 3000
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        patches = im2col(oneChannel[start:end], layer.filter_size, pad=layer.pad, stride=layer.stride)
        patches = norm2d(patches)
        patches = whiten(patches, layer.P)
        patches = patches.reshape((end - start, orows, ocols, -1))
        patches = np.dot(patches, beta)
        patches = relu(patches)
        patches = patches.transpose((0, 3, 1, 2))
        patches = pool(patches)
        if type(output) == float: output = np.zeros((size,) + patches.shape[1:], dtype=np.float)
        output[start:end] += patches
        print start, '~', end,
    return output


class ELMAEPoolLayer(Layer):
    def __init__(self, C, n_hidden, filter_size, pad, stride, pad_, stride_, noise_level, pool_size, mode, getoutput):
        self.C = C
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise_level = noise_level
        self.pool_size = pool_size
        self.mode = mode
        self.getoutput=getoutput

    def _get_beta(self, oneChannel, min_patch=40):
        assert oneChannel.ndim == 4 and oneChannel.shape[1] == 1  # ELMAE的输入通道数必须为1,即只有一张特征图
        # 生成随机正交滤波器
        filters, bias = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        filters = orthonormalize(filters)
        bias = orthonormalize(bias)
        # 卷积前向输出,和取patch时一致
        patches = im2col(oneChannel, self.filter_size, stride=self.stride_, pad=self.pad_)
        batches = oneChannel.shape[0]
        n_patch = patches.shape[0] / batches
        n_select = n_patch if n_patch < min_patch else min_patch  # 每张图片选择的patch数量
        index = [np.random.permutation(n_patch)[:n_select] + i * n_patch for i in xrange(batches)]
        index = np.hstack(index)
        patches = patches[index]
        gc.collect()
        patches = norm2d(patches)
        patches, self.P = whiten(patches)
        # 在原图上加噪
        # noiseChannel = add_mn(oneChannel, p=self.noise_level)
        # noiseChannel = add_sp(oneChannel, p=0.25)
        # noiseChannel = add_gs(oneChannel, std=None)
        # noisePatch = im2col(noiseChannel, self.filter_size, stride=self.stride_, pad=self.pad_)
        # 在patches上加噪
        noisePatch = add_mn(patches, p=self.noise_level)
        # noisePatch = add_mn_row(patches, p=0.25)
        # noisePatch = add_sp(patches, p=0.25)
        # noisePatch = add_gs(patches, p=0.25)
        hiddens = np.dot(noisePatch, filters) + bias
        del noisePatch
        hiddens = relu(hiddens)
        # 计算beta
        beta = compute_beta_val(hiddens, patches, compute_beta_val_times)
        beta = beta.T
        return beta

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        self.filters = []
        output = 0.
        for ch in xrange(channels):
            oneChannel = inputX[:, [ch], :, :]
            beta = self._get_beta(oneChannel)
            output = decomp(oneChannel, beta, self, output) if self.getoutput else 0
            self.filters.append(beta)
            print 'channel', ch
        # output = relu(output)
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        output = 0.
        for ch in xrange(channels):
            oneChannel = inputX[:, [ch], :, :]
            output = decomp(oneChannel, self.filters[ch], self, output)
        # output = relu(output)
        return output


class PoolLayer(Layer):
    def __init__(self, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
        self.pool_size = pool_size
        self.ignore_border = ignore_border
        self.stride = stride
        self.pad = pad
        self.mode = mode

    def get_train_output_for(self, inputX):
        output = pool_decomp(inputX, self.pool_size, self.ignore_border, self.stride, self.pad, self.mode)
        return output

    def get_test_output_for(self, inputX):
        output = pool_decomp(inputX, self.pool_size, self.ignore_border, self.stride, self.pad, self.mode)
        return output


class L2PoolLayer(Layer):
    def __init__(self, pool_size, ignore_border=False, stride=None, pad=(0, 0)):
        self.pool_size = pool_size
        self.ignore_border = ignore_border
        self.stride = stride
        self.pad = pad

    def get_train_output_for(self, inputX):
        output = np.square(inputX)
        output = pool_decomp(output, self.pool_size, self.ignore_border, self.stride, self.pad, 'sum')
        output = np.sqrt(output)
        return output

    def get_test_output_for(self, inputX):
        output = np.square(inputX)
        output = pool_decomp(output, self.pool_size, self.ignore_border, self.stride, self.pad, 'sum')
        output = np.sqrt(output)
        return output


class FPLayer(Layer):
    def __init__(self, pool_ratio, mode):
        self.pool_ratio = pool_ratio
        self.mode = mode

    def get_train_output_for(self, inputX):
        output = fp_cpu(inputX, self.pool_ratio, mode=self.mode)
        return output

    def get_test_output_for(self, inputX):
        output = fp_cpu(inputX, self.pool_ratio, mode=self.mode)
        return output


class BNLayer(Layer):
    def __init__(self):
        self.mean = None
        self.var = None

    def get_train_output_for(self, inputX, regularization=1e-5):
        # 每个特征在整个训练集上归一化
        self.mean = np.mean(inputX, axis=(0, 2, 3), keepdims=True)
        self.var = np.var(inputX, axis=(0, 2, 3), keepdims=True)
        return (inputX - self.mean) / np.sqrt(self.var + regularization)

    def get_test_output_for(self, inputX, regularization=1e-5):
        return (inputX - self.mean) / np.sqrt(self.var + regularization)


class MergeLayer(Layer):
    def __init__(self, subnet):
        assert isinstance(subnet, OrderedDict)
        self.subnet = subnet

    def get_train_output_for(self, inputX):
        output = []
        for name, layer in self.subnet.iteritems():
            out = layer.get_train_output_for(inputX)
            output.append(np.copy(out))
            print 'add ' + name,
        return np.concatenate(output, axis=1)

    def get_test_output_for(self, inputX):
        output = []
        for name, layer in self.subnet.iteritems():
            out = layer.get_test_output_for(inputX)
            output.append(np.copy(out))
        return np.concatenate(output, axis=1)


class SPPLayer(Layer):
    def __init__(self, pool_dims):
        self.pool_dims = pool_dims

    def get_train_output_for(self, inputX):
        input_size = inputX.shape[2:]
        pool_list = []
        for pool_dim in self.pool_dims:
            win_size = tuple((i + pool_dim - 1) // pool_dim for i in input_size)
            str_size = tuple(i // pool_dim for i in input_size)
            pool = pool_decomp(inputX, pool_size=win_size, stride=str_size)
            pool = pool.reshape((pool.shape[0], pool.shape[1], -1))
            pool_list.append(pool)
        pooled = np.concatenate(pool_list, axis=2)
        return pooled.reshape((-1, pooled.shape[2]))

    def get_test_output_for(self, inputX):
        return self.get_train_output_for(inputX)


class Classifier(Layer):
    def __init__(self, C, n_times):
        self.C = C
        self.n_times = n_times

    def get_train_output_for(self, inputX, inputy=None):
        n_hidden = int(self.n_times * inputX.shape[1])
        self.W, self.b = normal_random(input_unit=inputX.shape[1], hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        self.b = orthonormalize(self.b)
        H = np.dot(inputX, self.W) + self.b
        H = relu(H)
        self.beta = compute_beta_rand(H, inputy, self.C)
        out = np.dot(H, self.beta)
        return out

    def get_test_output_for(self, inputX):
        H = np.dot(inputX, self.W) + self.b
        H = relu(H)
        out = np.dot(H, self.beta)
        return out


def accuracy(ypred, ytrue):
    if ypred.ndim == 2:
        ypred = np.argmax(ypred, axis=1)
    if ytrue.ndim == 2:
        ytrue = np.argmax(ytrue, axis=1)
    return np.mean(ypred == ytrue)


class Classifier_cv(object):
    def __init__(self, n_rep, C_range, times_range):
        self.C_range = C_range
        self.n_rep = n_rep
        self.times_range = times_range
        self.clf_list = []

    def train_cv(self, inputX, inputy):
        optacc = 0.
        optout = None
        opttimes = None
        optC = None
        for n_times in self.times_range:
            for C in self.C_range:
                for j in xrange(self.n_rep):
                    clf = Classifier(C, n_times)
                    predout = clf.get_train_output_for(inputX, inputy)
                    acc = accuracy(predout, inputy)
                    self.clf_list.append(deepcopy(clf))
                    print n_times, C, 'repeat', j, ':', acc
                    if acc > optacc:
                        optacc = acc
                        optout = predout
                        opttimes = n_times
                        optC = C
                        self.optclf = clf  # 训练集最优分类器
        print 'train opt', opttimes, optC, optacc
        return optout

    def test(self, inputX):
        predout = self.optclf.get_test_output_for(inputX)  # 将训练集最优分类器应用在测试集
        return predout

    def test_cv(self, inputX, inputy):
        optacc = 0.
        optout = None
        opttimes = None
        optC = None
        for clf in self.clf_list:
            predout = clf.get_test_output_for(inputX)
            acc = accuracy(predout, inputy)
            print clf.n_times, clf.C, ':', acc
            if acc > optacc:
                optacc = acc
                optout = predout
                opttimes = clf.n_times
                optC = clf.C
                self.optclf = clf  # 测试集最优分类器
        print 'test opt', opttimes, optC, optacc
        return optout


class LRFELMAE(object):
    def __init__(self, C):
        self.C = C

    def _build(self):
        net = OrderedDict()
        # layer1
        net['layer1'] = ELMAEPoolLayer(C=self.C, n_hidden=192, filter_size=13, pad=0, stride=1, pad_=None, stride_=7,
                                       noise_level=0.2, pool_size=(6, 6), mode='max', getoutput=True)
        net['bn1'] = BNLayer()
        # layer2
        net['layer2'] = ELMAEPoolLayer(C=self.C, n_hidden=256, filter_size=8, pad=0, stride=1, pad_=None, stride_=4,
                                       noise_level=0.2, pool_size=(8, 8), mode='average_exc_pad', getoutput=False)
        net['bn2'] = BNLayer()
        # layer3
        # net['layer3'] = ELMAEPoolLayer(C=self.C, n_hidden=324, filter_size=7, stride=1,
        #                                stride_=1, noise_level=0.25, pool_size=(2, 2))
        # net['bn3'] = BNLayer()
        # # layer4
        # net['layer4'] = ELMAEPoolLayer(C=self.C, n_hidden=68, filter_size=7, stride=1,
        #                                noise_level=0.25, pool_size=(2, 2))
        # net['bn4'] = BNLayer()
        # net['cccp'] = CCCPLayer(C=self.C, n_out=32, noise_level=0.25)  # 降维
        return net

    def _get_train_output(self, net, inputX):
        out = inputX
        for name, layer in net.iteritems():
            out = layer.get_train_output_for(out)
            print 'add ' + name,
        return out

    def _get_test_output(self, net, inputX):
        out = inputX
        for name, layer in net.iteritems():
            out = layer.get_test_output_for(out)
        return out

    def train(self, unlabelX, inputX, inputy):
        self.net = self._build()
        self._get_train_output(self.net, unlabelX)
        netout = self._get_test_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        self.classifier = Classifier_cv(n_rep=2, C_range=10 ** np.arange(-2., 3., 1.), times_range=[5, ])
        return self.classifier.train_cv(netout, inputy)

    def test(self, inputX, inputy):
        netout = self._get_test_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        return self.classifier.test_cv(netout, inputy)


def main():
    # tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    # tr_X = myUtils.pre.norm4d_per_sample_channel(tr_X)
    # te_X = myUtils.pre.norm4d_per_sample_channel(te_X)
    # tr_X, te_X, tr_y, te_y = myUtils.pre.cifarWhiten('cifar10')
    # tr_y = myUtils.load.one_hot(tr_y, 10)
    # te_y = myUtils.load.one_hot(te_y, 10)
    # tr_X, te_X, tr_y, te_y = myUtils.load.cifar(onehot=True)
    tr_X, te_X, tr_y, te_y, un_X = myUtils.load.stl10(onehot=True)
    un_X, avg, var = myUtils.pre.norm4d(un_X)
    tr_X = myUtils.pre.norm4d(tr_X, avg, var)
    te_X = myUtils.pre.norm4d(te_X, avg, var)
    model = LRFELMAE(C=None)
    model.train(un_X, tr_X, tr_y)
    model.test(te_X, te_y)


if __name__ == '__main__':
    main()
