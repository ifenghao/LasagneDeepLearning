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
from copy import deepcopy
import myUtils

compute_beta_val_times = 4
dir_name = 'val'


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
    assert X.ndim == 2
    noise_dict = {'mn': add_mn, 'gs': add_gs, 'sp': add_sp,
                  'gs_part': add_gs_part, 'mn_gs': add_mn_gs,
                  'mn_block': add_mn_block}
    if noise_type not in noise_dict.keys():
        raise NotImplemented
    noise_fn = noise_dict[noise_type]
    if noise_type == 'mn_block':
        X = X.reshape((-1, int(np.sqrt(X.shape[-1])), int(np.sqrt(X.shape[-1]))))
    size = X.shape[0]
    batchSize = size / 10
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    for start, end in zip(startRange, endRange):
        X[start:end] = noise_fn(X[start:end], **args)
    if noise_type == 'mn_block':
        X = X.reshape((X.shape[0], -1))
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
    assert X.ndim == 3
    row_bond, col_bond = X.shape[-2:]
    n_blocks = len(block_list)
    interval = 1. / n_blocks
    pequal = percent / sum(map(lambda x: np.prod(x), block_list))
    uniform = np.random.uniform(low=0., high=1., size=X.shape)
    start = 0.
    for block_row_size, block_col_size in block_list:
        index = np.where((uniform > start) * (uniform <= start + pequal))
        index_mask = block_mask(index, block_row_size, block_col_size)
        out_of_bond = np.where((index_mask[-2] >= row_bond) + (index_mask[-1] >= col_bond))
        index_mask = map(lambda x: np.delete(x, out_of_bond), index_mask)
        X[index_mask] = 0.
        start += interval
    return X


def poolfn(pool_size, ignore_border, stride, pad, mode):
    xt = T.tensor4()
    poolx = pool_2d(xt, pool_size, ignore_border=ignore_border, st=stride, padding=pad, mode=mode)
    pool = theano.function([xt], poolx, allow_input_downcast=True)
    return pool


def pool_cpu(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
    if mode == 'avg': mode = 'average_exc_pad'
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


def pool_op(xnp, psize, pool_type, mode):
    if pool_type not in ('pool', 'fp'):
        raise NotImplemented
    if mode not in ('max', 'sum', 'avg', 'l2'):
        raise NotImplemented
    if pool_type == 'pool':
        if mode in ('max', 'sum', 'avg'):
            xnp = pool_cpu(xnp, psize, mode=mode)
        else:
            xnp = pool_l2_cpu(xnp, psize)
    else:
        if mode in ('max', 'sum', 'avg'):
            xnp = fp_cpu(xnp, psize, mode=mode)
        else:
            xnp = fp_l2_cpu(xnp, psize)
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
    def __init__(self, C, n_hidden, filter_size, pad, stride, pad_, stride_, noise_level,
                 pool_type, pool_size, mode, cccp_out, cccp_noise_level):
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
        self.cccp_out = cccp_out
        self.cccp_noise_level = cccp_noise_level

    def _get_beta(self, oneChannel, bias_scale=25):
        assert oneChannel.ndim == 4 and oneChannel.shape[1] == 1  # ELMAE的输入通道数必须为1,即只有一张特征图
        # 生成随机正交滤波器
        W, b = normal_random(input_unit=self.filter_size ** 2, hidden_unit=self.n_hidden)
        W = orthonormalize(W)  # 正交后的幅度在-1~+1之间
        # 卷积前向输出,和取patch时一致
        patches = im2col(oneChannel, self.filter_size, stride=self.stride_, pad=self.pad_)
        del oneChannel
        ##########################
        patches = norm2d(patches)
        # patches, self.mean1, self.P1 = whiten2d(patches)
        ##########################
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

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        self.filters = []
        self.cccps = []
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            myUtils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elmin')
            beta = self._get_beta(oneChannel)
            myUtils.visual.save_beta(beta, dir_name, 'beta')
            patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride)
            del oneChannel
            ##########################
            patches = norm2d(patches)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            patches = np.dot(patches, beta)
            patches = patches.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmraw')
            # 池化
            patches = pool_op(patches, self.pool_size, pool_type=self.pool_type, mode=self.mode)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmpool')
            # 归一化
            # patches = norm4d(patches)
            # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            patches = relu(patches)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_level=self.cccp_noise_level)
            # patches = cccp.get_train_output_for(patches)
            output = np.concatenate([output, patches], axis=1) if len(output) != 0 else patches
            self.filters.append(beta)
            self.cccps.append(deepcopy(cccp))
            print ch,
            gc.collect()
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            myUtils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elmin')
            patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride)
            del oneChannel
            ##########################
            patches = norm2d(patches)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            patches = np.dot(patches, self.filters[ch])
            patches = patches.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrawte')
            # 池化
            patches = pool_op(patches, self.pool_size, pool_type=self.pool_type, mode=self.mode)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmpoolte')
            # 归一化
            # patches = norm4d(patches)
            # myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            patches = relu(patches)
            myUtils.visual.save_map(patches[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            # patches = self.cccps[ch].get_test_output_for(patches)
            output = np.concatenate([output, patches], axis=1) if len(output) != 0 else patches
            print ch,
            gc.collect()
        return output


def im2col_catch(inputX, fsize, stride, pad):
    assert inputX.ndim == 4
    patches = []
    for ch in xrange(inputX.shape[1]):
        patches1ch = im2col(inputX[:, [0], :, :], fsize, stride, pad)
        inputX = inputX[:, 1:, :, :]
        patches = np.concatenate([patches, patches1ch], axis=1) if len(patches) != 0 else patches1ch
    return patches


def forward_decomp(inputX, beta, fsize, stride, pad):
    assert inputX.ndim == 4
    batchSize = int(round(float(inputX.shape[0]) / 10))
    splits = int(np.ceil(float(inputX.shape[0]) / batchSize))
    patches = []
    for _ in xrange(splits):
        patchestmp = im2col_catch(inputX[:batchSize], fsize, stride, pad)
        inputX = inputX[batchSize:]
        # 归一化
        patchestmp = norm2d(patchestmp)
        # patches, self.mean1, self.P1 = whiten2d(patches)
        patchestmp = np.dot(patchestmp, beta)
        patches = np.concatenate([patches, patchestmp], axis=0) if len(patches) != 0 else patchestmp
    return patches


class ELMAECrossAllLayer(Layer):
    def __init__(self, C, n_hidden, filter_size, pad, stride, pad_, stride_, noise_level,
                 pool_type, pool_size, mode, cccp_out, cccp_noise_level):
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
        self.cccp_out = cccp_out
        self.cccp_noise_level = cccp_noise_level

    def _get_beta(self, inputX, bias_scale=25):
        assert inputX.ndim == 4
        # 生成随机正交滤波器
        W, b = normal_random(input_unit=inputX.shape[1] * self.filter_size ** 2, hidden_unit=self.n_hidden)
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

    def get_train_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elmin')
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        # 学习beta
        self.beta = self._get_beta(inputX)
        myUtils.visual.save_beta_mch(self.beta, channels, dir_name, 'beta')
        inputX = forward_decomp(inputX, self.beta, self.filter_size, stride=self.stride, pad=self.pad)
        inputX = inputX.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
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
        # inputX = self.cccp.get_train_output_for(inputX)
        return inputX

    def get_test_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'elminte')
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        inputX = forward_decomp(inputX, self.beta, self.filter_size, stride=self.stride, pad=self.pad)
        inputX = inputX.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
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
        # inputX = self.cccp.get_test_output_for(inputX)
        return inputX


class ELMAECrossPartLayer(Layer):
    def __init__(self, C, n_hidden, filter_size, pad, stride, pad_, stride_, noise_level, cross_size,
                 pool_type, pool_size, mode, cccp_out, cccp_noise_level):
        self.C = C
        self.n_hidden = n_hidden
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.stride_ = stride_
        self.pad_ = pad_
        self.noise_level = noise_level
        self.cross_size = cross_size
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.mode = mode
        self.cccp_out = cccp_out
        self.cccp_noise_level = cccp_noise_level

    def _get_beta(self, partX, bias_scale=25):
        assert partX.ndim == 4
        # 生成随机正交滤波器
        W, b = normal_random(input_unit=partX.shape[1] * self.filter_size ** 2, hidden_unit=self.n_hidden)
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

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
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
            # 前向输出
            partX = forward_decomp(partX, beta, self.filter_size, stride=self.stride, pad=self.pad)
            partX = partX.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
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
            # 组合最终结果
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            self.filters.append(beta)
            self.cccps.append(deepcopy(cccp))
            print num,
            gc.collect()
        return output

    def get_test_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        output = []
        splits = int(np.ceil(float(channels) / self.cross_size))
        for num in xrange(splits):
            # 取部分通道
            partX = inputX[:, :self.cross_size, :, :]
            inputX = inputX[:, self.cross_size:, :, :]
            myUtils.visual.save_map(partX[[10, 100, 1000]], dir_name, 'elminte')
            # 前向输出
            partX = forward_decomp(partX, self.filters[num], self.filter_size, stride=self.stride, pad=self.pad)
            partX = partX.reshape((batches, orows, ocols, -1)).transpose((0, 3, 1, 2))
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
            # 组合最终结果
            output = np.concatenate([output, partX], axis=1) if len(output) != 0 else partX
            print num,
            gc.collect()
        return output


def get_indexed(inputX, idx):
    assert inputX.ndim == 3
    inputX = inputX[:, idx, :]
    return inputX.reshape((-1, inputX.shape[-1]))


def join_result(result3d, idx_list, orows, ocols):
    assert result3d.ndim == 3
    all_idx = np.concatenate(idx_list)
    sort_idx = np.argsort(all_idx)
    result3d = get_indexed(result3d, sort_idx)
    result3d = result3d.reshape((-1, orows, ocols, result3d.shape[-1])).transpose((0, 3, 1, 2))
    return result3d


# 分块取索引
def get_block_idx(part_size, orows, ocols):
    blockr, blockc = part_size
    nr = int(np.ceil(float(orows) / blockr))
    nc = int(np.ceil(float(ocols) / blockc))
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
                one_block = np.concatenate([one_block, one_row]) if len(one_block) != 0 else one_row
            idx.append(one_block)
    return idx


# 类似池化的方式取邻域索引
def get_neib_idx(part_size, orows, ocols):
    neibr, neibc = part_size
    idx = []
    for i in xrange(neibr):
        row_idx = np.arange(i, orows, neibr)
        for j in xrange(neibc):
            col_idx = np.arange(j, ocols, neibc)
            one_neib = []
            for row_step in row_idx:
                one_row = col_idx + row_step * orows
                one_neib = np.concatenate([one_neib, one_row]) if len(one_neib) != 0 else one_row
            idx.append(one_neib)
    return idx


def get_rand_idx(n_rand, orows, ocols):
    size = orows * ocols
    split_size = int(round(float(size) / n_rand))
    all_idx = np.random.permutation(size)
    split_range = [split_size + split_size * i for i in xrange(n_rand - 1)]
    split_idx = np.split(all_idx, split_range)
    return split_idx


class ELMAEMBetaLayer(Layer):
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

    def _train_forward(self, oneChannel):
        batches, channels, rows, cols = oneChannel.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride)
        del oneChannel
        patches = patches.reshape((batches, orows * ocols, self.filter_size ** 2))
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
            filters.append(beta)
        output = join_result(output, self.idx, orows, ocols)
        return output, filters

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        self.filters = []
        self.cccps = []
        output = []
        for ch in xrange(channels):
            oneChannel = inputX[:, [0], :, :]
            inputX = inputX[:, 1:, :, :]
            myUtils.visual.save_map(oneChannel[[10, 100, 1000]], dir_name, 'elmin')
            oneChannelOut, oneChannelFilter = self._train_forward(oneChannel)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmraw')
            # 池化
            oneChannelOut = pool_op(oneChannelOut, self.pool_size, pool_type=self.pool_type, mode=self.mode)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmpool')
            # 归一化
            # oneChannelOut = norm4d(oneChannelOut)
            # myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmnorm')
            # 激活
            oneChannelOut = relu(oneChannelOut)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmrelu')
            # 添加cccp层组合
            cccp = CCCPLayer(C=self.C, n_out=self.cccp_out, noise_level=self.cccp_noise_level)
            oneChannelOut = cccp.get_train_output_for(oneChannelOut)
            output = np.concatenate([output, oneChannelOut], axis=1) if len(output) != 0 else oneChannelOut
            self.filters.append(deepcopy(oneChannelFilter))
            self.cccps.append(deepcopy(cccp))
            print ch,
            gc.collect()
        return output

    def _test_forward(self, oneChannel, oneChannelFilter):
        batches, channels, rows, cols = oneChannel.shape
        _, _, orows, ocols = myUtils.basic.conv_out_shape((batches, channels, rows, cols),
                                                          (self.n_hidden, channels, self.filter_size, self.filter_size),
                                                          pad=self.pad, stride=self.stride)
        patches = im2col(oneChannel, self.filter_size, pad=self.pad, stride=self.stride)
        del oneChannel
        patches = patches.reshape((batches, orows * ocols, self.filter_size ** 2))
        output = []
        for num in xrange(len(self.idx)):
            one_part = get_indexed(patches, self.idx[num])
            ##########################
            one_part = norm2d(one_part)
            # patches = whiten2d(patches, self.mean1, self.P1)
            ##########################
            one_part = np.dot(one_part, oneChannelFilter[num])
            one_part = one_part.reshape((batches, -1, self.n_hidden))
            output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
        output = join_result(output, self.idx, orows, ocols)
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
            # 池化
            oneChannelOut = pool_op(oneChannelOut, self.pool_size, pool_type=self.pool_type, mode=self.mode)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmpoolte')
            # 归一化
            # oneChannelOut = norm4d(oneChannelOut)
            # myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmnormte')
            # 激活
            oneChannelOut = relu(oneChannelOut)
            myUtils.visual.save_map(oneChannelOut[[10, 100, 1000]], dir_name, 'elmrelute')
            # 添加cccp层组合
            oneChannelOut = self.cccps[ch].get_test_output_for(oneChannelOut)
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
            self.filters.append(beta)
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
            filters.append(beta)
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

    def _get_beta(self, inputX, bias_scale=25):
        assert inputX.ndim == 2
        W, b = normal_random(input_unit=inputX.shape[1], hidden_unit=self.n_out)
        W = orthonormalize(W)
        # 在转化的矩阵上加噪
        noiseX = np.copy(inputX)
        noiseX = add_noise_decomp(noiseX, add_mn, self.noise_level)
        # noiseX = add_sp(inputX, p=self.noise_level)
        # noiseX = add_gs(inputX, scale=self.noise_level)
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
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        ##################
        inputX = norm2d(inputX)
        ##################
        self.beta = self._get_beta(inputX)
        inputX = np.dot(inputX, self.beta)
        inputX = inputX.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpraw')
        # 归一化
        inputX = norm4d(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnorm')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelu')
        return inputX

    def get_test_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpinte')
        batches, n_in, rows, cols = inputX.shape
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        ##################
        inputX = norm2d(inputX)
        ##################
        inputX = np.dot(inputX, self.beta)
        inputX = inputX.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprawte')
        # 归一化
        inputX = norm4d(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnormte')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelute')
        return inputX


class CCCPMBetaLayer(Layer):
    def __init__(self, C, n_out, noise_level, part_size, idx_type):
        self.C = C
        self.n_out = n_out
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

    def _get_beta(self, inputX, W, b, bias_scale=25):
        assert inputX.ndim == 2
        # W, b = normal_random(input_unit=channels, hidden_unit=self.n_out)
        W = orthonormalize(W)
        # 在转化的矩阵上加噪
        noiseX = np.copy(inputX)
        noiseX = add_noise_decomp(noiseX, add_mn, self.noise_level)
        # noiseX = add_sp(inputX, p=self.noise_level)
        # noiseX = add_gs(inputX, scale=self.noise_level)
        H = np.dot(noiseX, W)
        del noiseX
        hmax, hmin = np.max(H, axis=0), np.min(H, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        H += b * scale
        H = relu(H)
        beta = compute_beta_direct(H, inputX)
        beta = beta.T
        return beta

    def _train_forward(self, inputX):
        batches, n_in, rows, cols = inputX.shape
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        ##################
        inputX = norm2d(inputX)
        ##################
        inputX = inputX.reshape((batches, -1, n_in))
        W, b = normal_random(input_unit=n_in, hidden_unit=self.n_out)
        self.idx = self.get_idx(self.part_size, rows, cols)
        self.filters = []
        output = []
        for num in xrange(len(self.idx)):
            one_part = get_indexed(inputX, self.idx[num])
            beta = self._get_beta(one_part, W, b)
            one_part = np.dot(one_part, beta)
            one_part = one_part.reshape((batches, -1, self.n_out))
            output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
            self.filters.append(beta)
        output = join_result(output, self.idx, rows, cols)
        return output

    def get_train_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpin')
        inputX = self._train_forward(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpraw')
        # 归一化
        inputX = norm4d(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnorm')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelu')
        return inputX

    def _test_forward(self, inputX):
        batches, n_in, rows, cols = inputX.shape
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, n_in))
        ##################
        inputX = norm2d(inputX)
        ##################
        inputX = inputX.reshape((batches, -1, n_in))
        output = []
        for num in xrange(len(self.idx)):
            one_part = get_indexed(inputX, self.idx[num])
            one_part = np.dot(one_part, self.filters[num])
            one_part = one_part.reshape((batches, -1, self.n_out))
            output = np.concatenate([output, one_part], axis=1) if len(output) != 0 else one_part
        output = join_result(output, self.idx, rows, cols)
        return output

    def get_test_output_for(self, inputX):
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpinte')
        inputX = self._test_forward(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprawte')
        # 归一化
        inputX = norm4d(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccpnormte')
        # 激活
        inputX = relu(inputX)
        myUtils.visual.save_map(inputX[[10, 100, 1000]], dir_name, 'cccprelute')
        return inputX


class Classifier(Layer):
    def __init__(self, C, n_times):
        self.C = C
        self.n_times = n_times

    def get_train_output_for(self, inputX, inputy=None, bias_scale=25):
        n_hidden = int(self.n_times * inputX.shape[1])
        self.W, self.b = normal_random(input_unit=inputX.shape[1], hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        H = np.dot(inputX, self.W)
        del inputX
        hmax, hmin = np.max(H, axis=0), np.min(H, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        self.b = self.b * scale
        H += self.b
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


class Classifier_Crange(object):
    def __init__(self, C_range, n_times):
        self.C_range = C_range
        self.n_times = n_times

    def get_train_acc(self, inputX, inputy=None, bias_scale=25):
        n_hidden = int(self.n_times * inputX.shape[1])
        print 'hiddens =', n_hidden
        self.W, self.b = normal_random(input_unit=inputX.shape[1], hidden_unit=n_hidden)
        self.W = orthonormalize(self.W)
        H = np.dot(inputX, self.W)
        del inputX
        hmax, hmin = np.max(H, axis=0), np.min(H, axis=0)
        scale = (hmax - hmin) / (2 * bias_scale)
        self.b = self.b * scale
        H += self.b
        H = relu(H)
        self.beta_list = []
        optacc = 0.
        optC = None
        for C in self.C_range:
            beta = compute_beta_rand(H, inputy, C)
            self.beta_list.append(beta)
            out = np.dot(H, beta)
            acc = accuracy(out, inputy)
            print '\t', C, acc
            if acc > optacc:
                optacc = acc
                optC = C
        return optC, optacc

    def get_test_acc(self, inputX, inputy=None):
        H = np.dot(inputX, self.W) + self.b
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
        optC = None
        for n_times in self.times_range:
            print 'times', n_times, ':'
            for j in xrange(self.n_rep):
                print 'repeat', j
                clf = Classifier_Crange(self.C_range, n_times)
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


class LRFELMAE(object):
    def __init__(self, C):
        self.C = C

    def _build(self):
        net = OrderedDict()
        # net['gcn0'] = GCNLayer()
        # layer1
        # net['layer1'] = ELMAELayer(C=self.C, n_hidden=64, filter_size=6, pad=0, stride=1, pad_=None, stride_=1,
        #                            noise_level=0.15, pool_type='pool', pool_size=(2, 2), mode='max',
        #                            cccp_out=32, cccp_noise_level=0.)
        net['layer1'] = ELMAECrossAllLayer(C=self.C, n_hidden=32, filter_size=8, pad=0, stride=1,
                                           pad_=None, stride_=4, noise_level=0.15,
                                           pool_type='pool', pool_size=(2, 2), mode='max',
                                           cccp_out=64, cccp_noise_level=0.)
        # net['layer1'] = ELMAEMBetaLayer(C=self.C, n_hidden=64, filter_size=6, pad=0, stride=1,
        #                                 part_size=(4, 4), idx_type='neib', noise_level=0.15,
        #                                 pool_type='pool', pool_size=(2, 2), mode='max',
        #                                 cccp_out=32, cccp_noise_level=0.)
        # net['gcn11'] = GCNLayer()
        # net['cccp1'] = CCCPLayer(C=self.C, n_out=64, noise_level=0.2)
        # net['gcn12'] = GCNLayer()
        # layer2
        net['layer2'] = ELMAELayer(C=self.C, n_hidden=32, filter_size=6, pad=0, stride=1,
                                   pad_=None, stride_=1, noise_level=0.15,
                                   pool_type='pool', pool_size=(2, 2), mode='max',
                                   cccp_out=16, cccp_noise_level=0.)
        # net['layer2'] = ELMAECrossPartLayer(C=self.C, n_hidden=128, filter_size=8, pad=0, stride=1,
        #                                     pad_=None, stride_=1, noise_level=0.15, cross_size=4,
        #                                     pool_type='pool', pool_size=(2, 2), mode='max',
        #                                     cccp_out=32, cccp_noise_level=0.)
        # net['layer2'] = ELMAEMBetaLayer(C=self.C, n_hidden=64, filter_size=6, pad=0, stride=1,
        #                                 part_size=(4, 4), idx_type='neib', noise_level=0.15,
        #                                 pool_type='pool', pool_size=(2, 2), mode='max',
        #                                 cccp_out=32, cccp_noise_level=0.)
        # net['gcn21'] = GCNLayer()
        # net['cccp2'] = CCCPLayer(C=self.C, n_out=289, noise_level=0.2)
        # net['gcn22'] = GCNLayer()
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
        self.classifier = Classifier_cv(n_rep=2, C_range=10 ** np.arange(-4., 2., 1.), times_range=[5, 7])
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
