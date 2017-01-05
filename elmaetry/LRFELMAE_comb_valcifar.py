# coding:utf-8
'''
基于ELM的局部感知自编码器
仿照多通道的卷积层,将上一层所有通道卷积后的特征图求和,得到下一层的一张特征图
cifar数据集
第一层使用crossall层
------
crossall采用分解计算防止内存溢出
在分解取inputX时,将已经取出的部分删除减少内存使用
增加block mn方法加噪
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
from collections import OrderedDict
from copy import copy, deepcopy
import myUtils

dir_name = 'val'


def compute_beta_reg(Hmat, Tmat, C):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = np.dot(Hmat.T, solve(np.eye(rows) / C + np.dot(Hmat, Hmat.T), Tmat))
    else:
        beta = solve(np.eye(cols) / C + np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
    return beta


def compute_beta_rand(Hmat, Tmat, C):
    Crand = abs(np.random.uniform(0.1, 1.1)) * C
    return compute_beta_reg(Hmat, Tmat, Crand)


def compute_beta_direct(Hmat, Tmat):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = np.dot(Hmat.T, solve(np.dot(Hmat, Hmat.T), Tmat))
    else:
        beta = solve(np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
    return beta


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
    std = 1.
    return np.random.normal(loc=0, scale=std, size=(input_unit, hidden_unit)), \
           np.random.normal(loc=0, scale=std, size=hidden_unit)


def uniform_random(input_unit, hidden_unit):
    ranges = 1.
    return np.random.uniform(low=-ranges, high=ranges, size=(input_unit, hidden_unit)), \
           np.random.uniform(low=-ranges, high=ranges, size=hidden_unit)


def im2col(inputX, fsize, stride, pad, ignore_border=False):
    assert inputX.ndim == 4
    if isinstance(fsize, (int, float)): fsize = (int(fsize), int(fsize))
    if isinstance(stride, (int, float)): stride = (int(stride), int(stride))
    Xrows, Xcols = inputX.shape[-2:]
    X = T.tensor4()
    if not ignore_border:  # 保持下和右的边界
        rowpad = colpad = 0
        rowrem = (Xrows - fsize[0]) % stride[0]
        if rowrem: rowpad = stride[0] - rowrem
        colrem = (Xcols - fsize[1]) % stride[1]
        if colrem: colpad = stride[1] - colrem
        pad = ((0, rowpad), (0, colpad))
    Xpad = lasagnepad(X, pad, batch_ndim=2)
    neibs = images2neibs(Xpad, fsize, stride, 'ignore_borders')
    im2colfn = theano.function([X], neibs, allow_input_downcast=True)
    return im2colfn(inputX)


def relu(X):
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        xtmp = X[start:end]
        X[start:end] = 0.5 * (xtmp + abs(xtmp))
    return X


def leaky_relu(X, alpha=0.2):
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        xtmp = X[start:end]
        X[start:end] = f1 * xtmp + f2 * abs(xtmp)
    return X


def elu(X, alpha=1):
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        xtmp = X[start:end]
        X[start:end][xtmp < 0.] = alpha * (np.exp(xtmp[xtmp < 0.]) - 1)
    return X


def add_noise_decomp(X, noise_type, args):
    noise_dict = {'mn': add_mn, 'gs': add_gs, 'sp': add_sp,
                  'gs_part': add_gs_part, 'mn_gs': add_mn_gs,
                  'mn_block': add_mn_block, 'mn_block_mch': add_mn_block_mch,
                  'mn_array': add_mn_array}
    if noise_type not in noise_dict.keys():
        raise NotImplementedError
    if noise_type == ('mn_block', 'mn_block_mch', 'mn_array'):
        assert X.ndim == 4
    noise_fn = noise_dict[noise_type]
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        X[start:end] = noise_fn(X[start:end], **args)
    return X


def add_mn(X, percent=0.5):
    retain_prob = 1. - percent
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=theano.config.floatX)
    return X * binomial


def add_mn_row(X, percent=0.5, reduced=False):
    if reduced:  # 由于是按行mask,比例可能需要减少
        percent = percent / X.shape[1]
    retain_prob = 1. - percent
    binomial = np.random.uniform(low=0., high=1., size=(X.shape[0], 1))
    binomial = np.asarray(binomial < retain_prob, dtype=theano.config.floatX)
    return X * binomial


def add_sp(X, percent=0.5):
    Xmin, Xmax = np.min(X), np.max(X)
    uniform = np.random.uniform(low=0., high=1., size=X.shape)
    salt = np.where((uniform > 0) * (uniform <= percent / 2.))
    pepper = np.where((uniform > percent / 2.) * (uniform <= percent))
    X[salt] = Xmin
    X[pepper] = Xmax
    return X


def add_gs(X, scale=100., std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / (2. * scale)
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    X += normal
    return X


def add_gs_part(X, percent=0.5, scale=100., std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / (2. * scale)
    retain_prob = 1. - percent
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=theano.config.floatX)
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    normal *= binomial
    X += normal
    return X


def add_mn_gs(X, percent=0.5, scale=100., std=None):
    if std is None:
        Xmin, Xmax = np.min(X), np.max(X)
        std = (Xmax - Xmin) / (2. * scale)
    normal = np.random.normal(loc=0, scale=std, size=X.shape)
    X += normal
    retain_prob = 1. - percent
    binomial = np.random.uniform(low=0., high=1., size=X.shape)
    binomial = np.asarray(binomial < retain_prob, dtype=theano.config.floatX)
    X *= binomial
    return X


def block_mask(index, row_size, col_size):
    assert len(index) == 3
    batch_idx, row_idx, col_idx = index
    # 样本简单重复
    batch_idx = np.repeat(np.tile(batch_idx, row_size), col_size)
    # 行列计算相邻索引
    length = len(row_idx)
    row_idx = np.tile(np.repeat(row_idx, col_size), row_size)
    bias = np.repeat(np.arange(row_size), length * col_size)
    row_idx += bias
    col_idx = np.repeat(np.tile(col_idx, row_size), col_size)
    bias = np.tile(np.arange(col_size), length * row_size)
    col_idx += bias
    return batch_idx, row_idx, col_idx


def add_mn_block(X, percent=0.5, block_list=((1, 1), (2, 2))):
    assert X.ndim == 4
    Xshape = X.shape
    batches, channels, rows, cols = Xshape
    X = X.reshape((-1, rows, cols))
    block_list = filter(lambda x: x[0] < rows and x[1] < cols, block_list)
    interval = 1. / len(block_list)
    pequal = percent / sum(map(lambda x: np.prod(x), block_list))
    uniform = np.random.uniform(low=0., high=1., size=X.shape)
    start = 0.
    for block_row, block_col in block_list:
        index = np.where((uniform > start) * (uniform <= start + pequal))
        index_mask = block_mask(index, block_row, block_col)
        out_of_bond = np.where((index_mask[-2] >= rows) + (index_mask[-1] >= cols))
        index_mask = map(lambda x: np.delete(x, out_of_bond), index_mask)
        X[index_mask] = 0.
        start += interval
    X = X.reshape(Xshape)
    return X


def add_mn_block_mch(X, percent=0.5, block_list=((1, 1), (2, 2))):
    assert X.ndim == 4
    batches, channels, rows, cols = X.shape
    block_list = filter(lambda x: x[0] < rows and x[1] < cols, block_list)
    interval = 1. / len(block_list)
    pequal = percent / (channels * sum(map(lambda x: np.prod(x), block_list)))
    uniform = np.random.uniform(low=0., high=1., size=(batches, rows, cols))
    start = 0.
    for block_row, block_col in block_list:
        index = np.where((uniform > start) * (uniform <= start + pequal))
        index_mask = block_mask(index, block_row, block_col)
        out_of_bond = np.where((index_mask[-2] >= rows) + (index_mask[-1] >= cols))
        index_mask = map(lambda x: np.delete(x, out_of_bond), index_mask)
        X[index_mask[0], :, index_mask[1], index_mask[2]] = 0.  # 所有通道的同一位置
        start += interval
    return X


class MNArray(object):
    def _scat_stride(self, blockr, blockc, total_blocks):
        strider = strider_tmp = 1
        stridec = stridec_tmp = 1
        while True:
            arrayr = (self.orows - blockr) // strider_tmp + 1
            arrayc = (self.ocols - blockc) // stridec_tmp + 1
            if arrayr * arrayc < total_blocks: break
            strider = strider_tmp
            stridec = stridec_tmp
            if strider_tmp < stridec_tmp:
                strider_tmp += 1
            else:
                stridec_tmp += 1
        return strider, stridec

    def _assign_ch_idx_permap(self, channels, block_size, n_blocks):
        ch_idx = np.repeat(np.arange(channels), n_blocks * block_size)
        return ch_idx

    def _assign_ch_idx_uniform(self, channels, block_size, n_blocks):
        ch_idx = np.random.permutation(channels)[:n_blocks]
        if channels < n_blocks:
            times = np.ceil(float(n_blocks) / channels)
            ch_idx = np.tile(ch_idx, times)[:n_blocks]
        ch_idx.sort()
        ch_idx = np.repeat(ch_idx, block_size)
        return ch_idx

    def _assign_ch_idx_rand(self, channels, block_size, n_blocks):
        ch_idx = np.random.randint(0, channels, n_blocks)
        ch_idx.sort()
        ch_idx = np.repeat(ch_idx, block_size)
        return ch_idx

    def _assign_onemap_idx(self, array_idx, array_size, n_blocks):
        idx_for_array_idx = np.random.permutation(array_size)[:n_blocks]  # 一定要均匀的分配map索引
        if array_size < n_blocks:
            times = np.ceil(float(n_blocks) / (array_size))
            idx_for_array_idx = np.random.permutation(np.tile(idx_for_array_idx, times)[:n_blocks])
        map_idx = array_idx[idx_for_array_idx].reshape(-1)
        return map_idx

    def _add_per_map(self, X, percent, block_list):
        assert X.ndim == 3
        equal_size = self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            map_blocks = int(round(equal_size / (blockr * blockc)))
            total_blocks = self.channels * map_blocks
            strider, stridec = self._scat_stride(blockr, blockc, total_blocks)
            arrayr = (self.orows - blockr) // strider + 1
            arrayc = (self.ocols - blockc) // stridec + 1
            array_idx = im2col(self.oidx, (blockr, blockc), (strider, stridec), 0, ignore_border=True).astype(int)
            for b in xrange(self.batches):  # 不同样本不同噪声
                ch_idx = self._assign_ch_idx_permap(self.channels, blockr * blockc, map_blocks)
                map_idx = self._assign_onemap_idx(array_idx, arrayr * arrayc, total_blocks)
                X[b][ch_idx, map_idx] = 0.
        return X

    def _add_cross_ch(self, X, percent, block_list):
        assert X.ndim == 3
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            strider, stridec = self._scat_stride(blockr, blockc, total_blocks)  # 尝试最分散的stride
            arrayr = (self.orows - blockr) // strider + 1  # 不考虑边界
            arrayc = (self.ocols - blockc) // stridec + 1
            array_idx = im2col(self.oidx, (blockr, blockc), (strider, stridec), 0, ignore_border=True).astype(int)
            for b in xrange(self.batches):  # 不同样本不同噪声
                ch_idx = self._assign_ch_idx_rand(self.channels, blockr * blockc, total_blocks)
                map_idx = self._assign_onemap_idx(array_idx, arrayr * arrayc, total_blocks)
                X[b][ch_idx, map_idx] = 0.
        return X

    def _add_cross_batch(self, X, percent, block_list):
        assert X.ndim == 3
        X = X.reshape((-1, self.orows * self.ocols))
        equal_size = self.channels * self.orows * self.ocols * percent / float(len(block_list))
        for blockr, blockc in block_list:
            total_blocks = int(round(equal_size / (blockr * blockc)))
            strider, stridec = self._scat_stride(blockr, blockc, total_blocks)  # 尝试最分散的stride
            arrayr = (self.orows - blockr) // strider + 1  # 不考虑边界
            arrayc = (self.ocols - blockc) // stridec + 1
            array_idx = im2col(self.oidx, (blockr, blockc), (strider, stridec), 0, ignore_border=True).astype(int)
            channels = self.batches * self.channels
            total_blocks *= self.batches
            ch_idx = self._assign_ch_idx_rand(channels, blockr * blockc, total_blocks)
            map_idx = self._assign_onemap_idx(array_idx, arrayr * arrayc, total_blocks)
            X[ch_idx, map_idx] = 0.
        return X

    def apply(self, X, percent, block_list, mode):
        assert X.ndim == 4
        add_fn = {'permap': self._add_per_map, 'channel': self._add_cross_ch, 'batch': self._add_cross_batch}
        if mode not in add_fn.keys():
            raise NotImplementedError
        Xshape = X.shape
        self.batches, self.channels, self.orows, self.ocols = Xshape
        self.oidx = np.arange(self.orows * self.ocols).reshape((1, 1, self.orows, self.ocols))
        X = X.reshape((self.batches, self.channels, -1))
        block_list = filter(lambda x: x[0] < self.orows and x[1] < self.ocols, block_list)
        X = add_fn[mode](X, percent, block_list)
        X = X.reshape(Xshape)
        return X


def add_mn_array(X, percent=0.5, block_list=((1, 1), (2, 2)), mode='channel'):
    assert X.ndim == 4
    X = MNArray().apply(X, percent, block_list, mode)
    return X


def poolfn(pool_size, ignore_border, stride, pad, mode):
    xt = T.tensor4()
    poolx = pool_2d(xt, pool_size, ignore_border=ignore_border, st=stride, padding=pad, mode=mode)
    pool = theano.function([xt], poolx, allow_input_downcast=True)
    return pool


def pool_cpu(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
    if mode == 'avg': mode = 'average_exc_pad'
    if isinstance(pool_size, (int, float)): pool_size = (int(pool_size), int(pool_size))
    pool = poolfn(pool_size, ignore_border, stride, pad, mode)
    return pool(xnp)


def pool_l2_cpu(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0)):
    if isinstance(pool_size, (int, float)): pool_size = (int(pool_size), int(pool_size))
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


def pool_op(xnp, psize, pool_type, mode, args=None):
    if pool_type not in ('pool', 'fp'):
        raise NotImplementedError
    if mode not in ('max', 'sum', 'avg', 'l2'):
        raise NotImplementedError
    if pool_type == 'pool':
        if mode in ('max', 'sum', 'avg'):
            xnp = pool_cpu(xnp, psize, mode=mode) if args is None \
                else pool_cpu(xnp, psize, mode=mode, **args)
        else:
            xnp = pool_l2_cpu(xnp, psize) if args is None \
                else pool_l2_cpu(xnp, psize, **args)
    else:
        if mode in ('max', 'sum', 'avg'):
            xnp = fp_cpu(xnp, psize, mode=mode) if args is None \
                else fp_cpu(xnp, psize, mode=mode, **args)
        else:
            xnp = fp_l2_cpu(xnp, psize) if args is None \
                else fp_l2_cpu(xnp, psize, **args)
    return xnp


# 对每一个patch里的元素去均值归一化
def norm2d(X, reg=0.1):
    size = X.shape[0]
    batchSize = size // 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        Xtmp = X[start:end]
        mean = Xtmp.mean(axis=1)
        Xtmp -= mean[:, None]
        normalizer = np.sqrt((Xtmp ** 2).mean(axis=1) + reg)
        Xtmp /= normalizer[:, None]
        X[start:end] = Xtmp
    return X


# 对每一个样本的每个通道元素去均值归一化
def norm4d(X, reg=0.1):
    raw_shape = X.shape
    X = X.reshape((X.shape[0], -1))
    X = norm2d(X, reg)
    X = X.reshape(raw_shape)
    return X


def norm2dglobal(X, mean=None, normalizer=None, reg=0.1):
    if mean is None or normalizer is None:
        mean = X.mean(axis=0)
        normalizer = 0.  # 分解求方差
        size = X.shape[0]
        batchSize = size // 10
        startRange = range(0, size - batchSize + 1, batchSize)
        endRange = range(batchSize, size + 1, batchSize)
        if size % batchSize != 0:
            startRange.append(size - size % batchSize)
            endRange.append(size)
        for start, end in zip(startRange, endRange):
            Xtmp = X[start:end]
            Xtmp -= mean[None, :]  # no copy,原始X的相应元素也被改变
            normalizer += (Xtmp ** 2).sum(axis=0) / size
        normalizer = np.sqrt(normalizer + reg)
        X /= normalizer[None, :]
        return X, mean, normalizer
    else:
        X = (X - mean[None, :]) / normalizer[None, :]
        return X


def norm4dglobal(X, mean=None, normalizer=None, reg=0.1):
    if mean is None or normalizer is None:
        mean = X.mean(axis=(0, 2, 3))
        normalizer = 0.  # 分解求方差
        size = X.shape[0]
        batchSize = size // 10
        startRange = range(0, size - batchSize + 1, batchSize)
        endRange = range(batchSize, size + 1, batchSize)
        if size % batchSize != 0:
            startRange.append(size - size % batchSize)
            endRange.append(size)
        for start, end in zip(startRange, endRange):
            Xtmp = X[start:end]
            Xtmp -= mean[None, :, None, None]  # no copy,原始X的相应元素也被改变
            normalizer += (Xtmp ** 2).sum(axis=(0, 2, 3)) / size
        normalizer = np.sqrt(normalizer + reg)
        X /= normalizer[None, :, None, None]
        return X, mean, normalizer
    else:
        X = (X - mean[None, :, None, None]) / normalizer[None, :, None, None]
        return X


def whiten2d(X, mean=None, P=None):
    if mean is None or P is None:
        mean = X.mean(axis=0)
        X -= mean
        cov = np.dot(X.T, X) / X.shape[0]
        D, V = np.linalg.eig(cov)
        reg = np.mean(D)
        P = V.dot(np.diag(np.sqrt(1 / (D + reg)))).dot(V.T)
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
        reg = np.mean(D)
        P = V.dot(np.diag(np.sqrt(1 / (D + reg)))).dot(V.T)
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


class ELMAELayer(Layer):
    def __init__(self, C, n_hidden, filter_size, pad, stride, pad_, stride_,
                 noise_type, noise_args, pool_type, pool_size, mode, pool_args,
                 cccp_out, cccp_noise_type, cccp_noise_args, add_cccp):
        self.C = C
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise_type = noise_type
        self.noise_args = noise_args
        assert noise_type != 'mn_block_mch'
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.pool_args = pool_args
        self.cccp_out = cccp_out
        self.cccp_noise_type = cccp_noise_type
        self.cccp_noise_args = cccp_noise_args
        self.add_cccp = add_cccp

    def _get_beta(self, oneChannel, bias_scale=10):
        assert oneChannel.ndim == 4 and oneChannel.shape[1] == 1  # ELMAE的输入通道数必须为1,即只有一张特征图
        batches = oneChannel.shape[0]
        patch_size = self.filter_size ** 2
        # 生成随机正交滤波器
        W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 卷积前向输出,和取patch时一致
        patches = im2col(oneChannel, self.filter_size, stride=self.stride_, pad=self.pad_, ignore_border=False)
        del oneChannel
        ##########################
        patches = norm2d(patches)
        # patches, self.mean1, self.P1 = whiten2d(patches)
        ##########################
        # 在patches上加噪
        noise_patches = np.copy(patches)
        if self.noise_type in ('mn_block', 'mn_array'):  # 4d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, patch_size))
            noise_patches = noise_patches.transpose((0, 3, 1, 2))
        noise_patches = add_noise_decomp(noise_patches, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block', 'mn_array'):
            noise_patches = noise_patches.transpose((0, 2, 3, 1))
            noise_patches = noise_patches.reshape((-1, patch_size))
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = relu(hiddens)
        # 计算beta
        beta = compute_beta_direct(hiddens, patches)
        beta = beta.T
        return beta

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        oshape_ = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                               (self.n_hidden, channels, self.filter_size, self.filter_size),
                                               pad=self.pad_, stride=self.stride_, ignore_border=False)
        self.orows_, self.ocols_ = oshape_[-2:]
        oshape = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                              (self.n_hidden, channels, self.filter_size, self.filter_size),
                                              pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        self.filters = []
        self.cccps = []
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            myUtils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elmin')
            beta = self._get_beta(oneChannel)
            myUtils.visual.save_beta(beta, dir_name, 'beta')
            patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride, ignore_border=False)
            del oneChannel
            ##########################
            patches = norm2d(patches)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            patches = np.dot(patches, beta)
            patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmraw')
            # 归一化
            # patches = norm4d(patches)
            # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            patches = relu(patches)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_type=self.cccp_noise_type,
                             noise_args=self.cccp_noise_args)
            if self.add_cccp: patches = cccp.get_train_output_for(patches)
            # 池化
            patches = pool_op(patches, self.pool_size, self.pool_type, self.mode, self.pool_args)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmpool')
            # 组合最终结果
            output = np.concatenate([output, patches], axis=1) if len(output) != 0 else patches
            self.filters.append(copy(beta))
            self.cccps.append(deepcopy(cccp))
            print ch,
            gc.collect()
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            myUtils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elminte')
            patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride, ignore_border=False)
            del oneChannel
            ##########################
            patches = norm2d(patches)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            patches = np.dot(patches, self.filters[ch])
            patches = patches.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrawte')
            # 归一化
            # patches = norm4d(patches)
            # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            patches = relu(patches)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            if self.add_cccp: patches = self.cccps[ch].get_test_output_for(patches)
            # 池化
            patches = pool_op(patches, self.pool_size, self.pool_type, self.mode, self.pool_args)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmpoolte')
            # 组合最终结果
            output = np.concatenate([output, patches], axis=1) if len(output) != 0 else patches
            print ch,
            gc.collect()
        return output


def im2col_catch(inputX, fsize, stride, pad, ignore_border=False):
    assert inputX.ndim == 4
    patches = []
    for ch in xrange(inputX.shape[1]):
        patches1ch = im2col(inputX[:, [0], :, :], fsize, stride, pad, ignore_border)
        inputX = inputX[:, 1:, :, :]
        patches = np.concatenate([patches, patches1ch], axis=1) if len(patches) != 0 else patches1ch
    return patches


class ELMAECrossAllLayer(Layer):
    def __init__(self, C, n_hidden, filter_size, pad, stride, pad_, stride_,
                 noise_type, noise_args, pool_type, pool_size, mode, pool_args,
                 cccp_out, cccp_noise_type, cccp_noise_args, add_cccp):
        self.C = C
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise_type = noise_type
        self.noise_args = noise_args
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.pool_args = pool_args
        self.cccp_out = cccp_out
        self.cccp_noise_type = cccp_noise_type
        self.cccp_noise_args = cccp_noise_args
        self.add_cccp = add_cccp

    def _get_beta(self, inputX, bias_scale=10):
        assert inputX.ndim == 4
        batches, channels = inputX.shape[:2]
        patch_size = self.filter_size ** 2
        # 生成随机正交滤波器
        W, b = normal_random(input_unit=channels * self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 将所有输入的通道的patch串联
        patches = im2col_catch(inputX, self.filter_size, stride=self.stride_, pad=self.pad_)
        del inputX
        ##########################
        patches = norm2d(patches)
        # patches, self.mean1, self.P1 = whiten2d(patches)
        ##########################
        # 在patches上加噪
        noise_patches = np.copy(patches)
        if self.noise_type == 'mn_block_mch':  # 4d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, channels, patch_size))
            noise_patches = noise_patches.transpose((0, 4, 3, 1, 2))
            noise_patches = noise_patches.reshape((-1, channels, self.orows_, self.ocols_))
        elif self.noise_type in ('mn_block', 'mn_array'):  # 4d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, channels * patch_size))
            noise_patches = noise_patches.transpose((0, 3, 1, 2))
            # 直接对patch
            # noise_patches = noise_patches.reshape((patches.shape[0] * channels, -1))
        noise_patches = add_noise_decomp(noise_patches, self.noise_type, self.noise_args)
        if self.noise_type == 'mn_block_mch':
            noise_patches = noise_patches.reshape((batches, patch_size, channels, self.orows_, self.ocols_))
            noise_patches = noise_patches.transpose((0, 3, 4, 2, 1))
            noise_patches = noise_patches.reshape((-1, channels * patch_size))
        elif self.noise_type in ('mn_block', 'mn_array'):
            noise_patches = noise_patches.transpose((0, 2, 3, 1))
            noise_patches = noise_patches.reshape((-1, channels * patch_size))
            # 直接对patch
            # noise_patches = noise_patches.reshape((patches.shape[0], -1))
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = relu(hiddens)
        # 计算beta
        beta = compute_beta_direct(hiddens, patches)
        beta = beta.T
        return beta

    def forward_decomp(self, inputX, beta):
        assert inputX.ndim == 4
        batchSize = int(round(float(inputX.shape[0]) / 10))
        splits = int(np.ceil(float(inputX.shape[0]) / batchSize))
        patches = []
        for _ in xrange(splits):
            patchestmp = im2col_catch(inputX[:batchSize], self.filter_size, stride=self.stride, pad=self.pad)
            inputX = inputX[batchSize:]
            # 归一化
            patchestmp = norm2d(patchestmp)
            # patchestmp = whiten2d(patchestmp, self.mean1, self.P1)
            patchestmp = np.dot(patchestmp, beta)
            patches = np.concatenate([patches, patchestmp], axis=0) if len(patches) != 0 else patchestmp
        return patches

    def get_train_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmin')
        batches, channels, rows, cols = inputX.shape
        oshape_ = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                               (self.n_hidden, channels, self.filter_size, self.filter_size),
                                               pad=self.pad_, stride=self.stride_)
        self.orows_, self.ocols_ = oshape_[-2:]
        oshape = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                              (self.n_hidden, channels, self.filter_size, self.filter_size),
                                              pad=self.pad, stride=self.stride)
        self.orows, self.ocols = oshape[-2:]
        # 学习beta
        self.beta = self._get_beta(inputX)
        myUtils.visual.save_beta_mch(self.beta, channels, dir_name, 'beta')
        # 前向计算
        inputX = self.forward_decomp(inputX, self.beta)
        inputX = inputX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmraw')
        # 归一化
        # inputX = norm4d(inputX)
        # myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmnorm')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrelu')
        # 添加cccp层组合
        self.cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_type=self.cccp_noise_type,
                              noise_args=self.cccp_noise_args)
        if self.add_cccp: inputX = self.cccp.get_train_output_for(inputX)
        # 池化
        inputX = pool_op(inputX, self.pool_size, self.pool_type, self.mode, self.pool_args)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmpool')
        return inputX

    def get_test_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elminte')
        batches, channels, rows, cols = inputX.shape
        # 前向计算
        inputX = self.forward_decomp(inputX, self.beta)
        inputX = inputX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrawte')
        # 归一化
        # inputX = norm4d(inputX)
        # myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmnormte')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrelute')
        # 添加cccp层组合
        if self.add_cccp: inputX = self.cccp.get_test_output_for(inputX)
        # 池化
        inputX = pool_op(inputX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmpoolte')
        return inputX


class ELMAECrossPartLayer(Layer):
    def __init__(self, C, n_hidden, filter_size, pad, stride, pad_, stride_, noise_type, noise_args,
                 cross_size, pool_type, pool_size, mode, pool_args,
                 cccp_out, cccp_noise_type, cccp_noise_args, add_cccp):
        self.C = C
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise_type = noise_type
        self.noise_args = noise_args
        self.cross_size = cross_size
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.pool_args = pool_args
        self.cccp_out = cccp_out
        self.cccp_noise_type = cccp_noise_type
        self.cccp_noise_args = cccp_noise_args
        self.add_cccp = add_cccp

    def _get_beta(self, partX, bias_scale=10):
        assert partX.ndim == 4
        batches, channels = partX.shape[:2]
        patch_size = self.filter_size ** 2
        # 生成随机正交滤波器
        W, b = normal_random(input_unit=channels * self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 将所有输入的通道的patch串联
        patches = im2col_catch(partX, self.filter_size, stride=self.stride_, pad=self.pad_)
        del partX
        ##########################
        patches = norm2d(patches)
        # patches, self.mean1, self.P1 = whiten2d(patches)
        ##########################
        # 在patches上加噪
        noise_patches = np.copy(patches)
        if self.noise_type == 'mn_block_mch':  # 4d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, channels, patch_size))
            noise_patches = noise_patches.transpose((0, 4, 3, 1, 2))
            noise_patches = noise_patches.reshape((-1, channels, self.orows_, self.ocols_))
        elif self.noise_type in ('mn_block', 'mn_array'):  # 4d
            noise_patches = noise_patches.reshape((batches, self.orows_, self.ocols_, channels * patch_size))
            noise_patches = noise_patches.transpose((0, 3, 1, 2))
            # 直接对patch
            # noise_patches = noise_patches.reshape((patches.shape[0] * channels, -1))
        noise_patches = add_noise_decomp(noise_patches, self.noise_type, self.noise_args)
        if self.noise_type == 'mn_block_mch':
            noise_patches = noise_patches.reshape((batches, patch_size, channels, self.orows_, self.ocols_))
            noise_patches = noise_patches.transpose((0, 3, 4, 2, 1))
            noise_patches = noise_patches.reshape((-1, channels * patch_size))
        elif self.noise_type in ('mn_block', 'mn_array'):
            noise_patches = noise_patches.transpose((0, 2, 3, 1))
            noise_patches = noise_patches.reshape((-1, channels * patch_size))
            # 直接对patch
            # noise_patches = noise_patches.reshape((patches.shape[0], -1))
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = relu(hiddens)
        # 计算beta
        beta = compute_beta_direct(hiddens, patches)
        beta = beta.T
        return beta

    def forward_decomp(self, inputX, beta):
        assert inputX.ndim == 4
        batchSize = int(round(float(inputX.shape[0]) / 10))
        splits = int(np.ceil(float(inputX.shape[0]) / batchSize))
        patches = []
        for _ in xrange(splits):
            patchestmp = im2col_catch(inputX[:batchSize], self.filter_size, stride=self.stride, pad=self.pad)
            inputX = inputX[batchSize:]
            # 归一化
            patchestmp = norm2d(patchestmp)
            # patchestmp = whiten2d(patchestmp, self.mean1, self.P1)
            patchestmp = np.dot(patchestmp, beta)
            patches = np.concatenate([patches, patchestmp], axis=0) if len(patches) != 0 else patchestmp
        return patches

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        oshape_ = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                               (self.n_hidden, channels, self.filter_size, self.filter_size),
                                               pad=self.pad_, stride=self.stride_, ignore_border=False)
        self.orows_, self.ocols_ = oshape_[-2:]
        oshape = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                              (self.n_hidden, channels, self.filter_size, self.filter_size),
                                              pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        # 将输入按照通道分为多个组,每个组学习一个beta
        self.filters = []
        self.cccps = []
        output = []
        splits = int(np.ceil(float(channels) / self.cross_size))
        for num in xrange(splits):
            # 取部分通道
            partX = inputX[:, :self.cross_size, :, :]
            inputX = inputX[:, self.cross_size:, :, :]
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmin')
            # 学习beta
            beta = self._get_beta(partX)
            myUtils.visual.save_beta_mch(beta, self.cross_size, dir_name, 'beta')
            # 前向计算
            partX = self.forward_decomp(partX, beta)
            partX = partX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmraw')
            # 归一化
            # partX = norm4d(partX)
            # myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            partX = relu(partX)
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_type=self.cccp_noise_type,
                             noise_args=self.cccp_noise_args)
            if self.add_cccp: partX = cccp.get_train_output_for(partX)
            # 池化
            partX = pool_op(partX, self.pool_size, self.pool_type, self.mode, self.pool_args)
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmpool')
            # 组合最终结果
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            self.filters.append(copy(beta))
            self.cccps.append(deepcopy(cccp))
            print num,
            gc.collect()
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        output = []
        splits = int(np.ceil(float(channels) / self.cross_size))
        for num in xrange(splits):
            # 取部分通道
            partX = inputX[:, :self.cross_size, :, :]
            inputX = inputX[:, self.cross_size:, :, :]
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elminte')
            # 前向计算
            partX = self.forward_decomp(partX, self.filters[num])
            partX = partX.reshape((batches, self.orows, self.ocols, -1)).transpose((0, 3, 1, 2))
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrawte')
            # 归一化
            # partX = norm4d(partX)
            # myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            partX = relu(partX)
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            if self.add_cccp: partX = self.cccps[num].get_test_output_for(partX)
            # 池化
            partX = pool_op(partX, self.pool_size, self.pool_type, self.mode, self.pool_args)
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmpoolte')
            # 组合最终结果
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            print num,
            gc.collect()
        return output


def get_indexed(inputX, idx):
    assert inputX.ndim == 4
    batches, rows, cols, patch_size = inputX.shape
    if idx.ndim == 2:  # 对应于block和neib
        idx_rows, idx_cols = idx.shape
        inputX = inputX.reshape((batches, -1, patch_size))
        idx = idx.reshape(-1)
        inputX = inputX[:, idx, :]
        return inputX.reshape((batches, idx_rows, idx_cols, patch_size))
    elif idx.ndim == 1:  # 对应于rand
        inputX = inputX.reshape((batches, -1, patch_size))
        inputX = inputX[:, idx, :]
        return inputX[:, None, :, :]
    else:
        raise NotImplementedError


def join_result(result, idx_list):
    assert result.ndim == 3
    idx_list = map(lambda x: x.reshape(-1), idx_list)
    all_idx = np.concatenate(idx_list)
    sort_idx = np.argsort(all_idx)
    result = result[:, sort_idx, :]
    return result


# 分块取索引,part_size表示分割出块的多少
def get_block_idx(part_size, orows, ocols):
    nr, nc = part_size
    blockr = int(np.ceil(float(orows) / nr))
    blockc = int(np.ceil(float(ocols) / nc))
    idx = []
    for row in xrange(nr):
        row_bias = row * blockr
        for col in xrange(nc):
            col_bias = col * blockc
            base = np.arange(blockc) if col_bias + blockc < ocols else np.arange(ocols - col_bias)
            block_row = blockr if row_bias + blockr < orows else orows - row_bias
            one_block = []
            for br in xrange(block_row):
                one_row = base + orows * br + col_bias + row_bias * orows
                one_block = np.concatenate([one_block, one_row[None, :]], axis=0) \
                    if len(one_block) != 0 else one_row[None, :]
            idx.append(copy(one_block))
    return idx


# 类似池化的方式取邻域索引,part_size表示分割出邻域的多少
def get_neib_idx(part_size, orows, ocols):
    nr, nc = part_size
    idx = []
    for i in xrange(nr):
        row_idx = np.arange(i, orows, nr)
        for j in xrange(nc):
            col_idx = np.arange(j, ocols, nc)
            one_neib = []
            for row_step in row_idx:
                one_row = col_idx + row_step * orows
                one_neib = np.concatenate([one_neib, one_row[None, :]], axis=0) \
                    if len(one_neib) != 0 else one_row[None, :]
            idx.append(copy(one_neib))
    return idx


def get_rand_idx(n_rand, orows, ocols):
    size = orows * ocols
    split_size = int(round(float(size) / n_rand))
    all_idx = np.random.permutation(size)
    split_range = [split_size + split_size * i for i in xrange(n_rand - 1)]
    split_idx = np.split(all_idx, split_range)
    return split_idx


class ELMAEMBetaLayer(Layer):
    def __init__(self, C, n_hidden, filter_size, pad, stride, noise_type, noise_args,
                 part_size, idx_type, pool_type, pool_size, mode, pool_args,
                 cccp_out, cccp_noise_type, cccp_noise_args, add_cccp):
        self.C = C
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.noise_type = noise_type
        self.noise_args = noise_args
        assert noise_type != 'mn_block_mch'
        self.part_size = part_size
        assert (idx_type == 'rand') ^ (noise_type == 'mn_block')
        if idx_type == 'block':
            self.get_idx = get_block_idx
        elif idx_type == 'neib':
            self.get_idx = get_neib_idx
        elif idx_type == 'rand':
            self.get_idx = get_rand_idx
        else:
            raise NameError
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.pool_args = pool_args
        self.cccp_out = cccp_out
        self.cccp_noise_type = cccp_noise_type
        self.cccp_noise_args = cccp_noise_args
        self.add_cccp = add_cccp

    def _get_beta(self, part_patches, W, b, bias_scale=10):
        assert part_patches.ndim == 4
        batches, rows, cols, patch_size = part_patches.shape
        part_patches = part_patches.reshape(-1, patch_size)
        # 生成随机正交滤波器
        # W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 在patches上加噪
        noise_patches = np.copy(part_patches)
        if self.noise_type in ('mn_block', 'mn_array'):  # 4d
            noise_patches = noise_patches.reshape((batches, rows, cols, patch_size))
            noise_patches = noise_patches.transpose((0, 3, 1, 2))
        noise_patches = add_noise_decomp(noise_patches, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block', 'mn_array'):
            noise_patches = noise_patches.transpose((0, 2, 3, 1))
            noise_patches = noise_patches.reshape((-1, patch_size))
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = relu(hiddens)
        # 计算beta
        beta = compute_beta_direct(hiddens, part_patches)
        beta = beta.T
        return beta

    def _train_forward(self, oneChannel):
        batches = oneChannel.shape[0]
        patch_size = self.filter_size ** 2
        patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride, ignore_border=False)
        del oneChannel
        ##########################
        patches = norm2d(patches)
        # patches = whiten2d(patches, self.mean1, self.P1)
        ##########################
        patches = patches.reshape((batches, self.orows, self.ocols, patch_size))
        W, b = normal_random(input_unit=patch_size, hidden_unit=self.n_hidden)
        self.idx = self.get_idx(self.part_size, self.orows, self.ocols)
        filters = []
        output = []
        for num in xrange(len(self.idx)):
            part_patches = get_indexed(patches, self.idx[num])
            beta = self._get_beta(part_patches, W, b)
            myUtils.visual.save_beta(beta, dir_name, 'beta')
            part_patches = part_patches.reshape(-1, patch_size)
            part_patches = np.dot(part_patches, beta)
            part_patches = part_patches.reshape((batches, -1, self.n_hidden))
            output = np.concatenate([output, part_patches], axis=1) if len(output) != 0 else part_patches
            filters.append(copy(beta))
        output = join_result(output, self.idx)
        output = output.reshape((batches, self.orows, self.ocols, self.n_hidden)).transpose((0, 3, 1, 2))
        return output, filters

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        oshape = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                              (self.n_hidden, channels, self.filter_size, self.filter_size),
                                              pad=self.pad, stride=self.stride, ignore_border=False)
        self.orows, self.ocols = oshape[-2:]
        self.filters = []
        self.cccps = []
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            myUtils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elmin')
            oneChannelOut, oneChannelFilter = self._train_forward(oneChannel)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmraw')
            # 归一化
            # oneChannelOut = norm4d(oneChannelOut)
            # myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            oneChannelOut = relu(oneChannelOut)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_type=self.cccp_noise_type,
                             noise_args=self.cccp_noise_args)
            if self.add_cccp: oneChannelOut = cccp.get_train_output_for(oneChannelOut)
            # 池化
            oneChannelOut = pool_op(oneChannelOut, self.pool_size, self.pool_type, self.mode, self.pool_args)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmpool')
            # 组合最终结果
            output = np.concatenate([output, oneChannelOut], axis=1) if len(output) != 0 else oneChannelOut
            self.filters.append(deepcopy(oneChannelFilter))
            self.cccps.append(deepcopy(cccp))
            print ch,
            gc.collect()
        return output

    def _test_forward(self, oneChannel, oneChannelFilter):
        batches = oneChannel.shape[0]
        patch_size = self.filter_size ** 2
        patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride, ignore_border=False)
        del oneChannel
        ##########################
        patches = norm2d(patches)
        # patches = whiten2d(patches, self.mean1, self.P1)
        ##########################
        patches = patches.reshape((batches, self.orows, self.ocols, patch_size))
        output = []
        for num in xrange(len(self.idx)):
            part_patches = get_indexed(patches, self.idx[num])
            part_patches = part_patches.reshape(-1, patch_size)
            part_patches = np.dot(part_patches, oneChannelFilter[num])
            part_patches = part_patches.reshape((batches, -1, self.n_hidden))
            output = np.concatenate([output, part_patches], axis=1) if len(output) != 0 else part_patches
        output = join_result(output, self.idx)
        output = output.reshape((batches, self.orows, self.ocols, self.n_hidden)).transpose((0, 3, 1, 2))
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            myUtils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elminte')
            oneChannelOut = self._test_forward(oneChannel, self.filters[ch])
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmrawte')
            # 归一化
            # oneChannelOut = norm4d(oneChannelOut)
            # myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            oneChannelOut = relu(oneChannelOut)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            if self.add_cccp: oneChannelOut = self.cccps[ch].get_test_output_for(oneChannelOut)
            # 池化
            oneChannelOut = pool_op(oneChannelOut, self.pool_size, self.pool_type, self.mode, self.pool_args)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmpoolte')
            # 组合最终结果
            output = np.concatenate([output, oneChannelOut], axis=1) if len(output) != 0 else oneChannelOut
            print ch,
            gc.collect()
        return output


class ELMAECrossAllMBetaLayer(Layer):
    def __init__(self, C, n_hidden, filter_size, pad, stride, noise_level, part_size, idx_type,
                 pool_type, pool_size, mode, cccp_out, cccp_noise_level):
        self.C = C
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.noise_level = noise_level
        self.part_size = part_size
        if idx_type == 'block':
            self.get_idx = get_block_idx
        elif idx_type == 'neib':
            self.get_idx = get_neib_idx
        elif idx_type == 'rand':
            self.get_idx = get_rand_idx
        else:
            raise NameError
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.cccp_out = cccp_out
        self.cccp_noise_level = cccp_noise_level

    def _get_beta(self, patches, W, b, bias_scale=25):
        # 生成随机正交滤波器
        # W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 在patches上加噪
        noise_patches = np.copy(patches)
        noise_patches = add_noise_decomp(noise_patches, add_mn, self.noise_level)
        # noisePatch = add_mn_row(patches, p=0.25)
        # noisePatch = add_sp(patches, p=0.25)
        # noisePatch = add_gs(patches, p=0.25)
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = relu(hiddens)
        # 计算beta
        beta = compute_beta_direct(hiddens, patches)
        beta = beta.T
        return beta

    def _train_forward(self, inputX):
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        patches = im2col_catch(inputX, self.filter_size, pad=self.pad, stride=self.stride)
        del inputX
        patches = patches.reshape((batches, orows * ocols, channels * self.filter_size ** 2))
        W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        self.idx = self.get_idx(self.part_size, orows, ocols)
        self.filters = []
        output = []
        for num in xrange(len(self.idx)):
            one_part = get_indexed(patches, self.idx[num])
            ##########################
            one_part = norm2d(one_part)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            beta = self._get_beta(one_part, W, b)
            myUtils.visual.save_beta(beta, dir_name, 'beta')
            one_part = np.dot(one_part, beta)
            one_part = one_part.reshape((batches, -1, self.n_hidden))
            output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
            self.filters.append(copy(beta))
        output = join_result(output, self.idx, orows, ocols)
        return output

    def get_train_output_for(self, inputX):
        inputX = self._train_forward(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmraw')
        # 池化
        inputX = pool_op(inputX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmpool')
        # 归一化
        # patches = norm4d(patches)
        # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnorm')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrelu')
        # 添加cccp层组合
        self.cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_level=self.cccp_noise_level)
        inputX = self.cccp.get_train_output_for(inputX)
        return inputX

    def _test_forward(self, inputX):
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        patches = im2col_catch(inputX, self.filter_size, pad=self.pad, stride=self.stride)
        del inputX
        patches = patches.reshape((batches, orows * ocols, channels * self.filter_size ** 2))
        output = []
        for num in xrange(len(self.idx)):
            one_part = get_indexed(patches, self.idx[num])
            ##########################
            one_part = norm2d(one_part)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            one_part = np.dot(one_part, self.filters[num])
            one_part = one_part.reshape((batches, -1, self.n_hidden))
            output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
        output = join_result(output, self.idx, orows, ocols)
        return output

    def get_test_output_for(self, inputX):
        inputX = self._test_forward(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrawte')
        # 池化
        inputX = pool_op(inputX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmpoolte')
        # 归一化
        # patches = norm4d(patches)
        # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnormte')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmrelute')
        # 添加cccp层组合
        inputX = self.cccp.get_test_output_for(inputX)
        return inputX


class ELMAECrossPartMBetaLayer(Layer):
    def __init__(self, C, n_hidden, filter_size, pad, stride, noise_level, part_size, idx_type, cross_size,
                 pool_type, pool_size, mode, cccp_out, cccp_noise_level):
        self.C = C
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.noise_level = noise_level
        self.part_size = part_size
        self.cross_size = cross_size
        if idx_type == 'block':
            self.get_idx = get_block_idx
        elif idx_type == 'neib':
            self.get_idx = get_neib_idx
        elif idx_type == 'rand':
            self.get_idx = get_rand_idx
        else:
            raise NameError
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.cccp_out = cccp_out
        self.cccp_noise_level = cccp_noise_level

    def _get_beta(self, patches, W, b, bias_scale=25):
        # 生成随机正交滤波器
        # W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 在patches上加噪
        noise_patches = np.copy(patches)
        noise_patches = add_noise_decomp(noise_patches, add_mn, self.noise_level)
        # noisePatch = add_mn_row(patches, p=0.25)
        # noisePatch = add_sp(patches, p=0.25)
        # noisePatch = add_gs(patches, p=0.25)
        hiddens = np.dot(noise_patches, W)
        del noise_patches
        hmax, hmin = np.max(hiddens, axis=0), np.min(hiddens, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        hiddens += b * scale
        hiddens = relu(hiddens)
        # 计算beta
        beta = compute_beta_direct(hiddens, patches)
        beta = beta.T
        return beta

    def _train_forward(self, partX):
        batches, channels, rows, cols = partX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        patches = im2col_catch(partX, self.filter_size, pad=self.pad, stride=self.stride)
        del partX
        patches = patches.reshape((batches, orows * ocols, channels * self.filter_size ** 2))
        W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        self.idx = self.get_idx(self.part_size, orows, ocols)
        filters = []
        output = []
        for num in xrange(len(self.idx)):
            one_part = get_indexed(patches, self.idx[num])
            ##########################
            one_part = norm2d(one_part)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            beta = self._get_beta(one_part, W, b)
            myUtils.visual.save_beta(beta, dir_name, 'beta')
            one_part = np.dot(one_part, beta)
            one_part = one_part.reshape((batches, -1, self.n_hidden))
            output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
            filters.append(copy(beta))
        output = join_result(output, self.idx, orows, ocols)
        return output, filters

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        # 将输入按照通道分为多个组,每个组学习一个beta
        self.filters = []
        self.cccps = []
        output = []
        splits = int(np.ceil(float(channels) / self.cross_size))
        for num in xrange(splits):
            # 取部分通道
            partX = inputX[:, :self.cross_size, :, :]
            inputX = inputX[:, self.cross_size:, :, :]
            partX, partXFilter = self._train_forward(partX)
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmraw')
            # 池化
            partX = pool_op(partX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmpool')
            # 归一化
            # patches = norm4d(patches)
            # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            partX = relu(partX)
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_level=self.cccp_noise_level)
            partX = cccp.get_train_output_for(partX)
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            self.filters.append(deepcopy(partXFilter))
            self.cccps.append(deepcopy(cccp))
            print num,
            gc.collect()
        return output

    def _test_forward(self, partX, partXFilter):
        batches, channels, rows, cols = partX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        patches = im2col_catch(partX, self.filter_size, pad=self.pad, stride=self.stride)
        del partX
        patches = patches.reshape((batches, orows * ocols, channels * self.filter_size ** 2))
        output = []
        for num in xrange(len(self.idx)):
            one_part = get_indexed(patches, self.idx[num])
            ##########################
            one_part = norm2d(one_part)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            one_part = np.dot(one_part, partXFilter[num])
            one_part = one_part.reshape((batches, -1, self.n_hidden))
            output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
        output = join_result(output, self.idx, orows, ocols)
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        output = []
        splits = int(np.ceil(float(channels) / self.cross_size))
        for num in xrange(splits):
            # 取部分通道
            partX = inputX[:, :self.cross_size, :, :]
            inputX = inputX[:, self.cross_size:, :, :]
            partX = self._test_forward(partX, self.filters[num])
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrawte')
            # 池化
            partX = pool_op(partX, self.pool_size, pool_type=self.pool_type, mode=self.mode)
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmpoolte')
            # 归一化
            # patches = norm4d(patches)
            # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            partX = relu(partX)
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            partX = self.cccps[num].get_test_output_for(partX)
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            print num,
            gc.collect()
        return output


class PoolLayer(Layer):
    def __init__(self, pool_size, pool_type, mode='max'):
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.mode = mode

    def get_train_output_for(self, inputX):
        output = pool_op(inputX, self.pool_size, self.pool_type, self.mode)
        return output

    def get_test_output_for(self, inputX):
        output = pool_op(inputX, self.pool_size, self.pool_type, self.mode)
        return output


class BNLayer(Layer):
    def get_train_output_for(self, inputX, regularization=0.1):
        inputX, self.mean, self.norm = norm4dglobal(inputX, regularization)
        return inputX

    def get_test_output_for(self, inputX, regularization=0.1):
        return norm4dglobal(inputX, self.mean, self.norm, regularization)


class GCNLayer(Layer):
    def get_train_output_for(self, inputX, regularization=0.1):
        return norm4d(inputX, regularization)

    def get_test_output_for(self, inputX, regularization=0.1):
        return norm4d(inputX, regularization)


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
            pool_list.append(copy(pool))
        pooled = np.concatenate(pool_list, axis=2)
        return pooled.reshape((-1, pooled.shape[2]))

    def get_test_output_for(self, inputX):
        return self.get_train_output_for(inputX)


class CCCPLayer(Layer):
    def __init__(self, C, n_out, noise_type, noise_args):
        self.C = C
        self.n_out = n_out
        self.noise_type = noise_type
        self.noise_args = noise_args

    def _get_beta(self, inputX, bias_scale=25):
        assert inputX.ndim == 4
        batches, rows, cols, n_in = inputX.shape
        W, b = normal_random(input_unit=n_in, hidden_unit=self.n_out)
        W = orthonormalize(W)
        # 在转化的矩阵上加噪
        noiseX = np.copy(inputX)
        inputX = inputX.reshape((-1, n_in))
        if self.noise_type in ('mn_block', 'mn_block_mch', 'mn_array'):  # 4d
            noiseX = noiseX.transpose((0, 3, 1, 2))
        noiseX = add_noise_decomp(noiseX, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block', 'mn_block_mch', 'mn_array'):
            noiseX = noiseX.transpose((0, 2, 3, 1))
        noiseX = noiseX.reshape((-1, n_in))
        H = np.dot(noiseX, W)
        del noiseX
        hmax, hmin = np.max(H, axis=0), np.min(H, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        H += b * scale
        H = relu(H)
        beta = compute_beta_direct(H, inputX)
        beta = beta.T
        return beta

    def get_train_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpin')
        batches, n_in, rows, cols = inputX.shape
        ##################
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        inputX = norm2d(inputX)
        ##################
        inputX = inputX.reshape((batches, rows, cols, -1))
        self.beta = self._get_beta(inputX)
        inputX = inputX.reshape((-1, n_in))
        # 前向计算
        inputX = np.dot(inputX, self.beta)
        inputX = inputX.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpraw')
        # 归一化
        # inputX = norm4d(inputX)
        # myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnorm')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelu')
        return inputX

    def get_test_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpinte')
        batches, n_in, rows, cols = inputX.shape
        ##################
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        inputX = norm2d(inputX)
        ##################
        # 前向计算
        inputX = np.dot(inputX, self.beta)
        inputX = inputX.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprawte')
        # 归一化
        # inputX = norm4d(inputX)
        # myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnormte')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelute')
        return inputX


class CCCPMBetaLayer(Layer):
    def __init__(self, C, n_out, noise_type, noise_args, part_size, idx_type):
        self.C = C
        self.n_out = n_out
        self.noise_type = noise_type
        self.noise_args = noise_args
        self.part_size = part_size
        if idx_type == 'block':
            self.get_idx = get_block_idx
        elif idx_type == 'neib':
            self.get_idx = get_neib_idx
        elif idx_type == 'rand':
            self.get_idx = get_rand_idx
        else:
            raise NameError

    def _get_beta(self, partX, W, b, bias_scale=25):
        assert partX.ndim == 4
        batches, rows, cols, n_in = partX.shape
        # W, b = normal_random(input_unit=self.n_in, hidden_unit=self.n_out)
        W = orthonormalize(W)
        # 在转化的矩阵上加噪
        noiseX = np.copy(partX)
        partX = partX.reshape((-1, n_in))
        if self.noise_type in ('mn_block', 'mn_block_mch', 'mn_array'):  # 4d
            noiseX = noiseX.transpose((0, 3, 1, 2))
        noiseX = add_noise_decomp(noiseX, self.noise_type, self.noise_args)
        if self.noise_type in ('mn_block', 'mn_block_mch', 'mn_array'):
            noiseX = noiseX.transpose((0, 2, 3, 1))
        noiseX = noiseX.reshape((-1, n_in))
        H = np.dot(noiseX, W)
        del noiseX
        hmax, hmin = np.max(H, axis=0), np.min(H, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        H += b * scale
        H = relu(H)
        beta = compute_beta_direct(H, partX)
        beta = beta.T
        return beta

    def _train_forward(self, inputX):
        batches, n_in, rows, cols = inputX.shape
        ##################
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        inputX = norm2d(inputX)
        inputX = inputX.reshape((batches, rows, cols, -1))
        ##################
        W, b = normal_random(input_unit=n_in, hidden_unit=self.n_out)
        self.idx = self.get_idx(self.part_size, rows, cols)
        self.filters = []
        output = []
        for num in xrange(len(self.idx)):
            partX = get_indexed(inputX, self.idx[num])
            beta = self._get_beta(partX, W, b)
            partX = partX.reshape((-1, n_in))
            partX = np.dot(partX, beta)
            partX = partX.reshape((batches, -1, self.n_out))
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            self.filters.append(copy(beta))
        output = join_result(output, self.idx)
        output = output.reshape((batches, rows, cols, self.n_out)).transpose((0, 3, 1, 2))
        return output

    def get_train_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpin')
        inputX = self._train_forward(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpraw')
        # 归一化
        # inputX = norm4d(inputX)
        # myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnorm')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelu')
        return inputX

    def _test_forward(self, inputX):
        batches, n_in, rows, cols = inputX.shape
        ##################
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        inputX = norm2d(inputX)
        inputX = inputX.reshape((batches, rows, cols, -1))
        ##################
        output = []
        for num in xrange(len(self.idx)):
            partX = get_indexed(inputX, self.idx[num])
            partX = partX.reshape((-1, n_in))
            partX = np.dot(partX, self.filters[num])
            partX = partX.reshape((batches, -1, self.n_out))
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
        output = join_result(output, self.idx)
        output = output.reshape((batches, rows, cols, self.n_out)).transpose((0, 3, 1, 2))
        return output

    def get_test_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpinte')
        inputX = self._test_forward(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprawte')
        # 归一化
        # inputX = norm4d(inputX)
        # myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnormte')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelute')
        return inputX


class CVInner(object):
    def get_train_acc(self, inputX, inputy):
        raise NotImplementedError

    def get_test_acc(self, inputX, inputy):
        raise NotImplementedError


class CVOuter(object):
    def train_cv(self, inputX, inputy):
        raise NotImplementedError

    def test_cv(self, inputX, inputy):
        raise NotImplementedError


def accuracy(ypred, ytrue):
    if ypred.ndim == 2:
        ypred = np.argmax(ypred, axis=1)
    if ytrue.ndim == 2:
        ytrue = np.argmax(ytrue, axis=1)
    return np.mean(ypred == ytrue)


class Classifier_ELM(Layer):
    def __init__(self, C, n_times):
        self.C = C
        self.n_times = n_times

    def get_train_output_for(self, inputX, inputy=None):
        n_hidden = int(self.n_times * inputX.shape[1])
        self.W, self.b = normal_random(input_unit=inputX.shape[1], hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        self.b = orthonormalize(self.b)
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        self.beta = compute_beta_rand(H, inputy, self.C)
        out = np.dot(H, self.beta)
        return out

    def get_test_output_for(self, inputX):
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        out = np.dot(H, self.beta)
        return out


class Classifier_ELMcv(CVInner):
    def __init__(self, C_range, n_times):
        self.C_range = C_range
        self.n_times = n_times

    def get_train_acc(self, inputX, inputy):
        n_hidden = int(self.n_times * inputX.shape[1])
        print 'hiddens =', n_hidden
        self.W, self.b = normal_random(input_unit=inputX.shape[1], hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        self.b = orthonormalize(self.b)
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        rows, cols = H.shape
        K = np.dot(H, H.T) if rows <= cols else np.dot(H.T, H)
        self.beta_list = []
        optacc = 0.
        optC = None
        for C in self.C_range:
            Crand = abs(np.random.uniform(0.1, 1.1)) * C
            beta = np.dot(H.T, solve(np.eye(rows) / Crand + K, inputy)) if rows <= cols \
                else solve(np.eye(cols) / Crand + K, np.dot(H.T, inputy))
            out = np.dot(H, beta)
            acc = accuracy(out, inputy)
            self.beta_list.append(copy(beta))
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        return optC, optacc

    def get_test_acc(self, inputX, inputy):
        H = np.dot(inputX, self.W) + self.b
        del inputX
        H = relu(H)
        optacc = 0.
        optC = None
        for beta, C in zip(self.beta_list, self.C_range):
            out = np.dot(H, beta)
            acc = accuracy(out, inputy)
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        return optC, optacc


class Classifier_ELMtimescv(CVOuter):
    def __init__(self, n_rep, C_range, times_range):
        self.C_range = C_range
        self.n_rep = n_rep
        self.times_range = times_range
        self.clf_list = []

    def train_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        for n_times in self.times_range:
            print 'times', n_times, ':'
            for j in xrange(self.n_rep):
                print 'repeat', j
                clf = Classifier_ELMcv(self.C_range, n_times)
                C, acc = clf.get_train_acc(inputX, inputy)
                self.clf_list.append(deepcopy(clf))
                if acc > optacc:
                    optacc = acc
                    optC = C
            print 'train opt', optC, optacc

    def test_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        for clf in self.clf_list:
            print 'times', clf.n_times, ':'
            C, acc = clf.get_test_acc(inputX, inputy)
            if acc > optacc:
                optacc = acc
                optC = C
            print 'test opt', optC, optacc


def addtrans_decomp(X, Y=None):
    if Y is None: Y = X
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        Xtmp = X[start:end, :] + Y[:, start:end].T
        result = np.concatenate([result, Xtmp], axis=0) if len(result) != 0 else Xtmp
    return result


def kernel(Xtr, Xte=None, kernel_type='rbf', kernel_args=(1.,)):
    rows_tr = Xtr.shape[0]
    if not isinstance(kernel_args, (tuple, list)): kernel_args = (kernel_args,)
    if kernel_type == 'rbf':
        if Xte is None:
            H = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            # omega = H + H.T - 2 * np.dot(Xtr, Xtr.T)
            omega = addtrans_decomp(H) - 2 * np.dot(Xtr, Xtr.T)
            del H, Xtr
            omega = np.exp(-omega / kernel_args[0])
        else:
            rows_te = Xte.shape[0]
            Htr = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_te, axis=1)
            Hte = np.repeat(np.sum(Xte ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            # omega = Htr + Hte.T - 2 * np.dot(Xtr, Xte.T)
            omega = addtrans_decomp(Htr, Hte) - 2 * np.dot(Xtr, Xte.T)
            del Htr, Hte, Xtr, Xte
            omega = np.exp(-omega / kernel_args[0])
    elif kernel_type == 'lin':
        if Xte is None:
            omega = np.dot(Xtr, Xtr.T)
        else:
            omega = np.dot(Xtr, Xte.T)
    elif kernel_type == 'poly':
        if Xte is None:
            omega = (np.dot(Xtr, Xtr.T) + kernel_args[0]) ** kernel_args[1]
        else:
            omega = (np.dot(Xtr, Xte.T) + kernel_args[0]) ** kernel_args[1]
    elif kernel_type == 'wav':
        if Xte is None:
            H = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            omega = H + H.T - 2 * np.dot(Xtr, Xtr.T)
            H1 = np.repeat(np.sum(Xtr, axis=1, keepdims=True), rows_tr, axis=1)
            omega1 = H1 - H1.T
            omega = np.cos(omega1 / kernel_args[0]) * np.exp(-omega / kernel_args[1])
        else:
            rows_te = Xte.shape[0]
            Htr = np.repeat(np.sum(Xtr ** 2, axis=1, keepdims=True), rows_te, axis=1)
            Hte = np.repeat(np.sum(Xte ** 2, axis=1, keepdims=True), rows_tr, axis=1)
            omega = Htr + Hte.T - 2 * np.dot(Xtr, Xte.T)
            Htr1 = np.repeat(np.sum(Xtr, axis=1, keepdims=True), rows_te, axis=1)
            Hte1 = np.repeat(np.sum(Xte, axis=1, keepdims=True), rows_tr, axis=1)
            omega1 = Htr1 - Hte1.T
            omega = np.cos(omega1 / kernel_args[0]) * np.exp(-omega / kernel_args[1])
    else:
        raise NotImplemented
    return omega


class Classifier_KELM(Layer):
    def __init__(self, C, kernel_type, kernel_args):
        self.C = C
        self.kernel_type = kernel_type
        self.kernel_args = kernel_args

    def get_train_output_for(self, inputX, inputy=None):
        self.trainX = inputX
        omega = kernel(inputX, self.kernel_type, self.kernel_args)
        rows = omega.shape[0]
        Crand = abs(np.random.uniform(0.1, 1.1)) * self.C
        self.beta = solve(np.eye(rows) / Crand + omega, inputy)
        out = np.dot(omega, self.beta)
        return out

    def get_test_output_for(self, inputX):
        omega = kernel(self.trainX, inputX, self.kernel_type, self.kernel_args)
        del inputX
        out = np.dot(omega.T, self.beta)
        return out


class Classifier_KELMcv(CVOuter):
    def __init__(self, C_range, kernel_type, kernel_args_list):
        self.C_range = C_range
        self.kernel_type = kernel_type
        self.kernel_args_list = kernel_args_list

    def train_cv(self, inputX, inputy):
        self.trainX = inputX
        self.beta_list = []
        optacc = 0.
        optC = None
        optarg = None
        for kernel_args in self.kernel_args_list:
            omega = kernel(inputX, None, self.kernel_type, kernel_args)
            rows = omega.shape[0]
            for C in self.C_range:
                Crand = abs(np.random.uniform(0.1, 1.1)) * C
                beta = solve(np.eye(rows) / Crand + omega, inputy)
                out = np.dot(omega, beta)
                acc = accuracy(out, inputy)
                self.beta_list.append(copy(beta))
                print '\t', kernel_args, C, acc
                if acc > optacc:
                    optacc = acc
                    optC = C
                    optarg = kernel_args
            del omega
            gc.collect()
        print 'train opt', optarg, optC, optacc

    def test_cv(self, inputX, inputy):
        optacc = 0.
        optC = None
        optarg = None
        num = 0
        for kernel_args in self.kernel_args_list:
            omega = kernel(self.trainX, inputX, self.kernel_type, kernel_args)
            for C in self.C_range:
                out = np.dot(omega.T, self.beta_list[num])
                acc = accuracy(out, inputy)
                print '\t', kernel_args, C, acc
                num += 1
                if acc > optacc:
                    optacc = acc
                    optC = C
                    optarg = kernel_args
            del omega
            gc.collect()
        print 'test opt', optarg, optC, optacc


class LRFELMAE(object):
    def __init__(self, C):
        self.C = C

    def _build(self):
        net = OrderedDict()
        # net['gcn0'] = GCNLayer()
        # layer1
        # net['layer1'] = ELMAELayer(C=self.C, n_hidden=32, filter_size=6, pad=0, stride=1, pad_=0, stride_=1,
        #                            noise_type='mn_block', noise_args={'percent': 0.5, 'block_list': ((5, 5),)},
        #                            pool_type='pool', pool_size=(2, 2), mode='max',
        #                            cccp_out=32, cccp_noise_type='mn_block',
        #                            cccp_noise_args={'percent': 0.5, 'block_list': ((5, 5),)})
        net['layer1'] = ELMAECrossAllLayer(C=self.C, n_hidden=32, filter_size=6, pad=0, stride=1,
                                           pad_=0, stride_=2, noise_type='mn_array',
                                           noise_args={'percent': 0.3,
                                                       'block_list': ((5, 5), (9, 9), (13, 13),),
                                                       'mode': 'channel'},
                                           pool_type='fp', pool_size=2.6, mode='max', pool_args={'overlap': False,},
                                           add_cccp=False, cccp_out=32, cccp_noise_type='mn_array',
                                           cccp_noise_args={'percent': 0.3,
                                                            'block_list': ((5, 5), (9, 9), (13, 131),),
                                                            'mode': 'channel'})
        # net['layer1'] = ELMAEMBetaLayer(C=self.C, n_hidden=64, filter_size=6, pad=0, stride=1,
        #                                 part_size=(4, 4), idx_type='neib', noise_level=0.15,
        #                                 pool_type='pool', pool_size=(2, 2), mode='max',
        #                                 cccp_out=32, cccp_noise_level=0.)
        # net['gcn11'] = GCNLayer()
        # net['cccp1'] = CCCPLayer(C=self.C, n_out=64, noise_level=0.2)
        # net['gcn12'] = GCNLayer()
        # layer2
        net['layer2'] = ELMAELayer(C=self.C, n_hidden=32, filter_size=4, pad=0, stride=1,
                                   pad_=0, stride_=1, noise_type='mn_array',
                                   noise_args={'percent': 0.3,
                                               'block_list': ((3, 3), (5, 5),),
                                               'mode': 'channel'},
                                   pool_type='fp', pool_size=2.6, mode='max', pool_args={'overlap': False,},
                                   add_cccp=False, cccp_out=32, cccp_noise_type='mn_array',
                                   cccp_noise_args={'percent': 0.3,
                                                    'block_list': ((3, 3), (5, 5),),
                                                    'mode': 'channel'})
        # net['layer2'] = ELMAECrossPartLayer(C=self.C, n_hidden=32, filter_size=6, pad=0, stride=1,
        #                                     pad_=0, stride_=1, cross_size=4, noise_type='mn_block',
        #                                     noise_args={'percent': 0.5, 'block_list': ((5, 5),)},
        #                                     pool_type='fp', pool_size=2.4, mode='avg',
        #                                     cccp_out=32, cccp_noise_type='mn_block',
        #                                     cccp_noise_args={'percent': 0.5, 'block_list': ((5, 5),)})
        # net['layer2'] = ELMAEMBetaLayer(C=self.C, n_hidden=64, filter_size=6, pad=0, stride=1,
        #                                 part_size=(4, 4), idx_type='neib', noise_level=0.15,
        #                                 pool_type='pool', pool_size=(2, 2), mode='max',
        #                                 cccp_out=32, cccp_noise_level=0.)
        # net['gcn21'] = GCNLayer()
        # net['cccp2'] = CCCPLayer(C=self.C, n_out=289, noise_level=0.2)
        # net['gcn22'] = GCNLayer()
        # layer3
        # net['layer3'] = ELMAELayer(C=self.C, n_hidden=32, filter_size=6, pad=0, stride=1, pad_=0, stride_=1,
        #                            noise_type='mn_block', noise_args={'percent': 0.5, 'block_list': ((5, 5),)},
        #                            pool_type='fp', pool_size=2.4, mode='avg',
        #                            cccp_out=32, cccp_noise_type='mn_block',
        #                            cccp_noise_args={'percent': 0.5, 'block_list': ((5, 5),)})
        # net['layer3'] = ELMAECrossPartLayer(C=self.C, n_hidden=32, filter_size=6, pad=0, stride=1,
        #                                     pad_=0, stride_=1, cross_size=4, noise_type='mn_block',
        #                                     noise_args={'percent': 0.5, 'block_list': ((5, 5),)},
        #                                     pool_type='fp', pool_size=2.4, mode='avg',
        #                                     cccp_out=32, cccp_noise_type='mn_block',
        #                                     cccp_noise_args={'percent': 0.5, 'block_list': ((5, 5),)})
        # net['layer3'] = ELMAEMBetaLayer(C=self.C, n_hidden=64, filter_size=6, pad=0, stride=1,
        #                                 part_size=(4, 4), idx_type='neib', noise_level=0.15,
        #                                 pool_type='pool', pool_size=(2, 2), mode='max',
        #                                 cccp_out=32, cccp_noise_level=0.)
        # net['gcn21'] = GCNLayer()
        # net['cccp2'] = CCCPLayer(C=self.C, n_out=289, noise_level=0.2)
        # net['gcn22'] = GCNLayer()
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
        # self.classifier = Classifier_ELMtimescv(n_rep=3, C_range=10 ** np.arange(-1., 3., 1.), times_range=[12, ])
        self.classifier = Classifier_KELMcv(C_range=10 ** np.arange(2, 7., 1.), kernel_type='rbf',
                                            kernel_args_list=10 ** np.arange(2., 7., 1.))
        return self.classifier.train_cv(netout, inputy)

    def test(self, inputX, inputy):
        netout = self._get_test_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        return self.classifier.test_cv(netout, inputy)


def main():
    # tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    # tr_X = myUtils.pre.norm4d_per_sample(tr_X)
    # te_X = myUtils.pre.norm4d_per_sample(te_X)
    # tr_X, te_X, tr_y, te_y = myUtils.pre.cifarWhiten('cifar10')
    # tr_y = myUtils.load.one_hot(tr_y, 10)
    # te_y = myUtils.load.one_hot(te_y, 10)
    tr_X, te_X, tr_y, te_y = myUtils.load.cifar(onehot=True)
    tr_X = myUtils.pre.norm4d_per_sample(tr_X, scale=55.)
    te_X = myUtils.pre.norm4d_per_sample(te_X, scale=55.)
    model = LRFELMAE(C=None)
    model.train(tr_X, tr_y)
    model.test(te_X, te_y)


if __name__ == '__main__':
    main()
