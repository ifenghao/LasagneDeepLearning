# coding:utf-8
'''
基于ELM的局部感知自编码器
仿照多通道的卷积层,将上一层所有通道卷积后的特征图求和,得到下一层的一张特征图
不使用全局的权值共享卷积,而是将特征图分块,在块内权值共享
使用邻域分割方法
------
将归一化激活和CCCP放在拼接后的特征图后处理
归一化为每个样本计算,计算beta时归一化和白化,激活前归一化
分块计算beta时采用相同的投影矩阵,但是加噪不同,还是会学习出不同差别大的beta
bugs: whiten要减去同一个均值
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
from sklearn.decomposition.pca import PCA
from collections import OrderedDict
from copy import deepcopy
import myUtils

compute_beta_val_times = 4
dir_name = 'large4b'


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
    size = x.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        xtmp = x[start:end]
        result.append(0.5 * (xtmp + abs(xtmp)))
    return np.concatenate(result, axis=0)


def leaky_relu(x, alpha=0.2):
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    size = x.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        xtmp = x[start:end]
        result.append(f1 * xtmp + f2 * abs(xtmp))
    return np.concatenate(result, axis=0)


def elu(x, alpha=1):
    size = x.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        xtmp = x[start:end]
        xtmp[xtmp < 0.] = alpha * (np.exp(xtmp[xtmp < 0.]) - 1)
        result.append(xtmp)
    return np.concatenate(result, axis=0)


def add_noise_decomp(X, fn, arg):
    size = X.shape[0]
    batchSize = size / 100
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(fn(X[start:end], arg))
    return np.concatenate(result, axis=0)


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


def add_sp(X, p=0.5):
    Xmin, Xmax = np.min(X), np.max(X)
    uniform = np.random.uniform(low=0., high=1., size=X.shape)
    salt = np.where((uniform > 0) * (uniform <= p / 2.))
    pepper = np.where((uniform > p / 2.) * (uniform <= p))
    X[salt] = Xmin
    X[pepper] = Xmax
    return X


def add_gs(X, scale=1., std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / (2. * scale)
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


def pool_cpu(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
    pool = poolfn(pool_size, ignore_border, stride, pad, mode)
    return pool(xnp)


def pool_l2_cpu(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0)):
    pool = poolfn(pool_size, ignore_border, stride, pad, 'sum')
    xnp = np.square(xnp)
    xnp = pool(xnp)
    xnp = np.sqrt(xnp)
    return xnp


def fpfn(pool_ratio, constant, overlap, mode):
    xt = T.tensor4()
    fpx = myUtils.pool.fp(xt, pool_ratio, constant, overlap, mode)
    fp = theano.function([xt], fpx, allow_input_downcast=True)
    return fp


def fp_cpu(xnp, pool_ratio, constant=0.5, overlap=True, mode='max'):
    fp = fpfn(pool_ratio, constant, overlap, mode)
    return fp(xnp)


def fp_l2_cpu(xnp, pool_ratio, constant=0.5, overlap=True):
    fp = fpfn(pool_ratio, constant, overlap, 'sum')
    xnp = np.square(xnp)
    xnp = fp(xnp)
    xnp = np.sqrt(xnp)
    return xnp


# 对每一个patch里的元素去均值归一化
def norm2d(X, reg=1e-5):
    size = X.shape[0]
    batchSize = size // 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        Xtmp = X[start:end]
        mean = Xtmp.mean(axis=1)
        Xtmp -= mean[:, np.newaxis]
        normalizer = np.sqrt((Xtmp ** 2).sum(axis=1))
        normalizer[normalizer < reg] = 1.
        Xtmp /= normalizer[:, np.newaxis]
        result.append(Xtmp)
    return np.concatenate(result, axis=0)


# 对每一个样本的每个通道元素去均值归一化
def norm4d(X, reg=1e-5):
    raw_shape = X.shape
    X = X.reshape((X.shape[0], -1))
    X = norm2d(X, reg)
    X = X.reshape(raw_shape)
    return X


def whiten2d(X, mean=None, P=None):
    if mean is None and P is None:
        mean = X.mean(axis=0)
        X -= mean
        cov = np.dot(X.T, X) / X.shape[0]
        D, V = np.linalg.eig(cov)
        P = V.dot(np.diag(np.sqrt(1 / (D + 0.1)))).dot(V.T)
        X = X.dot(P)
        return X, mean, P
    else:
        X -= mean
        X = X.dot(P)
        return X


def whiten4d(X, mean=None, P=None):
    raw_shape = X.shape
    X = X.reshape((X.shape[0], -1))
    if mean is None and P is None:
        mean = X.mean(axis=0)
        X -= mean
        cov = np.dot(X.T, X) / X.shape[0]
        D, V = np.linalg.eig(cov)
        P = V.dot(np.diag(np.sqrt(1 / (D + 0.1)))).dot(V.T)
        X = X.dot(P)
        X = X.reshape(raw_shape)
        return X, mean, P
    else:
        X -= mean
        X = X.dot(P)
        X = X.reshape(raw_shape)
        return X


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


class ELMAEPoolLayer(Layer):
    def __init__(self, w_list, b_list, C, n_hidden, filter_size, pad, stride, pad_, stride_, noise_level,
                 pool_type, pool_size, mode):
        self.C = C
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise_level = noise_level
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.w_list = w_list
        self.b_list = b_list

    def _get_beta(self, oneChannel, w, b, min_patch=25):
        assert oneChannel.ndim == 4 and oneChannel.shape[1] == 1  # ELMAE的输入通道数必须为1,即只有一张特征图
        # 生成随机正交滤波器
        # filters, bias = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        # filters = orthonormalize(filters)
        # bias = orthonormalize(bias)
        # 卷积前向输出,和取patch时一致
        patches = im2col(oneChannel, self.filter_size, stride=self.stride_, pad=self.pad_)
        # batches = oneChannel.shape[0]
        # n_patch = patches.shape[0] / batches
        # n_select = n_patch if n_patch < min_patch else min_patch  # 每张图片选择的patch数量
        # index = [np.random.permutation(n_patch)[:n_select] + i * n_patch for i in xrange(batches)]
        # index = np.hstack(index)
        # patches = patches[index]
        # gc.collect()
        ##########################
        patches = norm2d(patches)
        patches, self.mean, self.P = whiten2d(patches)
        ##########################
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
        hiddens = np.dot(noisePatch, w) + b
        del noisePatch
        hiddens = relu(hiddens)
        # 计算beta
        beta = compute_beta_val(hiddens, patches, compute_beta_val_times)
        beta = beta.T
        return beta

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        if self.pool_type == 'pool':
            pool_op = pool_cpu
        elif self.pool_type == 'fp':
            pool_op = fp_cpu
        else:
            raise NameError
        self.filters = []
        self.cccps = []
        self.bns = []
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [ch], :, :]
            beta = self._get_beta(oneChannel, self.w_list[ch], self.b_list[ch])
            patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride)
            ##########################
            patches = norm2d(patches)
            patches = whiten2d(patches, self.mean, self.P)
            ##########################
            patches = np.dot(patches, beta)
            patches = patches.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
            # 池化
            patches = pool_op(patches, self.pool_size, mode=self.mode)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmraw')
            # # 归一化
            # bn = BNLayer()
            # patches = bn.get_train_output_for(patches)
            # # 激活
            # patches = relu(patches)
            # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrelu')
            # # 添加cccp层组合
            # cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_level=self.noise_level)
            # patches = cccp.get_train_output_for(patches)
            output.append(patches)
            self.filters.append(beta)
            # self.cccps.append(deepcopy(cccp))
            # self.bns.append(deepcopy(bn))
            print ch,
            gc.collect()
        output = np.concatenate(output, axis=1)
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        if self.pool_type == 'pool':
            pool_op = pool_cpu
        elif self.pool_type == 'fp':
            pool_op = fp_cpu
        else:
            raise NameError
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [ch], :, :]
            patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride)
            ##########################
            patches = norm2d(patches)
            patches = whiten2d(patches, self.mean, self.P)
            ##########################
            patches = np.dot(patches, self.filters[ch])
            patches = patches.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
            # 池化
            patches = pool_op(patches, self.pool_size, mode=self.mode)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrawte')
            # # 归一化
            # patches = self.bns[ch].get_test_output_for(patches)
            # # 激活
            # patches = relu(patches)
            # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrelute')
            # # 添加cccp层组合
            # patches = self.cccps[ch].get_test_output_for(patches)
            output.append(patches)
            print ch,
            gc.collect()
        output = np.concatenate(output, axis=1)
        return output


def split_block(x, split_size):
    assert x.ndim == 4
    batches, channels, rows, cols = x.shape
    splited_rows = int(np.ceil(float(rows) / split_size))
    splited_cols = int(np.ceil(float(cols) / split_size))
    rowpad = splited_rows * split_size - rows
    colpad = splited_cols * split_size - cols
    pad = (0, rowpad, 0, colpad)
    x = myUtils.basic.pad2d(x, pad)
    result = []
    for i in xrange(split_size):
        for j in xrange(split_size):
            result.append(x[:, :,
                          i * splited_rows:(i + 1) * splited_rows,
                          j * splited_cols:(j + 1) * splited_cols])
    return result


def join_block(x_list, split_size):
    result = []
    for i in xrange(split_size):
        result.append(np.concatenate(x_list[i * split_size:(i + 1) * split_size], axis=3))
    return np.concatenate(result, axis=2)


def split_neib(x, split_size):
    assert x.ndim == 4
    batches, channels, rows, cols = x.shape
    splited_rows = int(np.ceil(float(rows) / split_size))
    splited_cols = int(np.ceil(float(cols) / split_size))
    rowpad = splited_rows * split_size - rows
    colpad = splited_cols * split_size - cols
    pad = (0, rowpad, 0, colpad)
    x = myUtils.basic.pad2d(x, pad)
    result = []
    for i in xrange(split_size):
        for j in xrange(split_size):
            result.append(x[:, :, i::split_size, j::split_size])
    return result


def join_neib(x_list, split_size):
    result = []
    for i in xrange(split_size):
        x_row = x_list[i * split_size:(i + 1) * split_size]
        x_row = map(lambda x: x[:, :, :, np.newaxis, :, np.newaxis], x_row)
        x_row = np.concatenate(x_row, axis=5)
        x_row = x_row.reshape(x_row.shape[:-2] + (-1,))
        result.append(x_row)
    result = np.concatenate(result, axis=3)
    result = result.reshape(result.shape[:2] + (-1, result.shape[-1]))
    return result


class ELMAELayer(Layer):
    def __init__(self, C, split_list, params, cccp_out_list, noise_level):
        self.C = C
        self.split_list = split_list
        self.elmaepool_list = []
        self.params = params
        self.cccp_out_list = cccp_out_list
        self.noise_level = noise_level

    def get_train_output_for(self, inputX):
        self.min_rows = self.min_cols = np.inf
        self.cccps = []
        outputX = []
        for num in xrange(len(self.split_list)):
            w_list = []
            b_list = []
            for ch in xrange(inputX.shape[1]):
                w, b = normal_random(input_unit=self.params[num]['filter_size'] ** 2,
                                     hidden_unit=self.params[num]['n_hidden'])
                w_list.append(orthonormalize(w))
                b_list.append(orthonormalize(b))
            splitX_list = split_neib(inputX, self.split_list[num])
            resultX = []
            for x in splitX_list:
                elmaepool = ELMAEPoolLayer(w_list, b_list, **self.params[num])
                x = elmaepool.get_train_output_for(x)
                resultX.append(x)
                self.elmaepool_list.append(deepcopy(elmaepool))
            resultX = join_neib(resultX, self.split_list[num])
            # 归一化
            resultX = norm4d(resultX)
            myUtils.visual.save_map(resultX[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            resultX = relu(resultX)
            myUtils.visual.save_map(resultX[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, n_out=self.cccp_out_list[num], noise_level=self.noise_level)
            resultX = cccp.get_train_output_for(resultX)
            outputX.append(resultX)
            self.cccps.append(deepcopy(cccp))
            if resultX.shape[2] < self.min_rows: self.min_rows = resultX.shape[2]
            if resultX.shape[3] < self.min_cols: self.min_cols = resultX.shape[3]
        outputX = map(lambda x: x[:, :, :self.min_rows, :self.min_cols], outputX)
        return np.concatenate(outputX, axis=1)

    def get_test_output_for(self, inputX):
        outputX = []
        elm_num = 0
        for num in xrange(len(self.split_list)):
            splitX_list = split_neib(inputX, self.split_list[num])
            resultX = []
            for x in splitX_list:
                x = self.elmaepool_list[elm_num].get_test_output_for(x)
                elm_num += 1
                resultX.append(x)
            resultX = join_neib(resultX, self.split_list[num])
            # 归一化
            resultX = norm4d(resultX)
            myUtils.visual.save_map(resultX[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            resultX = relu(resultX)
            myUtils.visual.save_map(resultX[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            resultX = self.cccps[num].get_test_output_for(resultX)
            outputX.append(resultX)
        outputX = map(lambda x: x[:, :, :self.min_rows, :self.min_cols], outputX)
        return np.concatenate(outputX, axis=1)


class PoolLayer(Layer):
    def __init__(self, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
        self.pool_size = pool_size
        self.ignore_border = ignore_border
        self.stride = stride
        self.pad = pad
        self.mode = mode

    def get_train_output_for(self, inputX):
        output = pool_cpu(inputX, self.pool_size, self.ignore_border, self.stride, self.pad, self.mode)
        return output

    def get_test_output_for(self, inputX):
        output = pool_cpu(inputX, self.pool_size, self.ignore_border, self.stride, self.pad, self.mode)
        return output


class L2PoolLayer(Layer):
    def __init__(self, pool_size, pool_type='pool'):
        self.pool_size = pool_size
        if pool_type == 'pool':
            self.pool_op = pool_l2_cpu
        elif pool_type == 'fp':
            self.pool_op = fp_l2_cpu
        else:
            raise NameError

    def get_train_output_for(self, inputX):
        output = self.pool_op(inputX, self.pool_size)
        return output

    def get_test_output_for(self, inputX):
        output = self.pool_op(inputX, self.pool_size)
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

    def get_train_output_for(self, inputX, regularization=10):
        # 每个特征在整个训练集上归一化
        self.mean = np.mean(inputX, axis=(0, 2, 3), keepdims=True)
        self.var = np.var(inputX, axis=(0, 2, 3), keepdims=True)
        return (inputX - self.mean) / np.sqrt(self.var + regularization)

    def get_test_output_for(self, inputX, regularization=10):
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
            pool = pool_cpu(inputX, pool_size=win_size, stride=str_size)
            pool = pool.reshape((pool.shape[0], pool.shape[1], -1))
            pool_list.append(pool)
        pooled = np.concatenate(pool_list, axis=2)
        return pooled.reshape((-1, pooled.shape[2]))

    def get_test_output_for(self, inputX):
        return self.get_train_output_for(inputX)


class CCCPLayer(Layer):
    def __init__(self, C, n_out, noise_level):
        self.C = C
        self.n_out = n_out
        self.noise_level = noise_level

    def _get_beta(self, inputX):
        assert inputX.ndim == 4
        batches, channels, rows, cols = inputX.shape
        W, b = normal_random(input_unit=channels, hidden_unit=self.n_out)
        W = orthonormalize(W)
        b = orthonormalize(b)
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, channels))
        # 在转化的矩阵上加噪
        # noiseX = add_mn(inputX, p=self.noise_level)
        # noiseX = add_sp(inputX, p=0.25)
        # noiseX = add_gs(inputX, p=0.25)
        H = np.dot(inputX, W) + b
        H = elu(H)
        beta = compute_beta_val(H, inputX, compute_beta_val_times)
        beta = beta.T
        return beta

    def get_train_output_for(self, inputX):
        batches, n_in, rows, cols = inputX.shape
        self.beta = self._get_beta(inputX)
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        outputX = np.dot(inputX, self.beta)
        del inputX
        outputX = outputX.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        myUtils.visual.save_map(outputX[[10, 100, 1000]], dir_name, 'cccpraw')
        # 归一化
        self.bn = BNLayer()
        outputX = self.bn.get_train_output_for(outputX)
        # 激活
        outputX = elu(outputX)
        myUtils.visual.save_map(outputX[[10, 100, 1000]], dir_name, 'cccprelu')
        return outputX

    def get_test_output_for(self, inputX):
        batches, n_in, rows, cols = inputX.shape
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        outputX = np.dot(inputX, self.beta)
        del inputX
        outputX = outputX.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        myUtils.visual.save_map(outputX[[10, 100, 1000]], dir_name, 'cccprawte')
        # 归一化
        outputX = self.bn.get_test_output_for(outputX)
        # 激活
        outputX = elu(outputX)
        myUtils.visual.save_map(outputX[[10, 100, 1000]], dir_name, 'cccprelute')
        return outputX


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
        params1 = []
        params1.append(
            {'C': self.C, 'n_hidden': 180, 'filter_size': 11, 'pad': 0, 'stride': 1, 'pad_': None, 'stride_': 4,
             'noise_level': 0.2, 'pool_type': 'fp', 'pool_size': 1.414, 'mode': 'max'})
        params1.append(
            {'C': self.C, 'n_hidden': 90, 'filter_size': 6, 'pad': 0, 'stride': 1, 'pad_': None, 'stride_': 2,
             'noise_level': 0.2, 'pool_type': 'fp', 'pool_size': 1.414, 'mode': 'max'})
        params1.append(
            {'C': self.C, 'n_hidden': 60, 'filter_size': 5, 'pad': 0, 'stride': 1, 'pad_': None, 'stride_': 1,
             'noise_level': 0.2, 'pool_type': 'fp', 'pool_size': 1.414, 'mode': 'max'})
        net['layer1'] = ELMAELayer(C=self.C, split_list=[1, 2, 3], params=params1,
                                   cccp_out_list=[60, 30, 20], noise_level=0.2)
        # net['bn1'] = BNLayer()
        net['cccp1'] = CCCPLayer(C=self.C, n_out=64, noise_level=0.2)
        # net['bn2'] = BNLayer()
        # layer2
        params2 = []
        params2.append(
            {'C': self.C, 'n_hidden': 80, 'filter_size': 7, 'pad': 0, 'stride': 1, 'pad_': None, 'stride_': 2,
             'noise_level': 0.2, 'pool_type': 'pool', 'pool_size': (2, 2), 'mode': 'max'})
        params2.append(
            {'C': self.C, 'n_hidden': 40, 'filter_size': 3, 'pad': 0, 'stride': 1, 'pad_': None, 'stride_': 1,
             'noise_level': 0.2, 'pool_type': 'pool', 'pool_size': (2, 2), 'mode': 'max'})
        net['layer2'] = ELMAELayer(C=self.C, split_list=[1, 2], params=params2,
                                   cccp_out_list=[40, 20], noise_level=0.2)
        # net['bn3'] = BNLayer()
        net['cccp2'] = CCCPLayer(C=self.C, n_out=512, noise_level=0.2)
        # net['bn4'] = BNLayer()
        # layer3
        # net['layer3'] = ELMAEPoolLayer(C=self.C, n_hidden=14, filter_size=5, pad=2, stride=1, pad_=None, stride_=1,
        #                                noise_level=0.2, pool_type='pool', pool_size=(3, 3), mode='average_exc_pad',
        #                                cccp_out=2)
        # net['bn3'] = BNLayer()
        # net['cccp2'] = CCCPLayer(C=self.C, n_out=32, noise_level=0.2)
        # net['bn3'] = BNLayer()
        # net['pool'] = PoolLayer(pool_size=(3, 3), mode='average_exc_pad')
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

    def train(self, inputX, inputy):
        self.net = self._build()
        netout = self._get_train_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        self.classifier = Classifier_cv(n_rep=2, C_range=10 ** np.arange(-2., 3., 1.), times_range=[5, ])
        return self.classifier.train_cv(netout, inputy)

    def test(self, inputX, inputy):
        netout = self._get_test_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        return self.classifier.test_cv(netout, inputy)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X = myUtils.pre.norm4d_per_sample(tr_X)
    te_X = myUtils.pre.norm4d_per_sample(te_X)
    # tr_X, te_X, tr_y, te_y = myUtils.pre.cifarWhiten('cifar10')
    # tr_y = myUtils.load.one_hot(tr_y, 10)
    # te_y = myUtils.load.one_hot(te_y, 10)
    # tr_X, te_X, tr_y, te_y = myUtils.load.cifar(onehot=True)
    # tr_X = myUtils.pre.norm4d_per_sample(tr_X, scale=55.)
    # te_X = myUtils.pre.norm4d_per_sample(te_X, scale=55.)
    model = LRFELMAE(C=None)
    model.train(tr_X, tr_y)
    model.test(te_X, te_y)


if __name__ == '__main__':
    main()
