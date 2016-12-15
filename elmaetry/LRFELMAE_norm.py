# coding:utf-8
'''
基于ELM的局部感知自编码器
仿照多通道的卷积层，将上一层所有通道卷积后的特征图求和，得到下一层的一张特征图
在ELMAE的过程中使用zca_whiten，而不是对求得的特征图
对求得的特征图可以使用BN中均值方差归一化
要对所有的通道同时白化，而不能各通道独立白化
要把训练的白化矩阵用在测试数据上，不能各自独立白化
--------
bug：所有ELMAE前向输出的时候没有激活
'''
import gc
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from numpy.linalg import solve
from scipy.linalg import orth, svd
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
from lasagne.utils import floatX
from collections import OrderedDict
import myUtils


def compute_beta(Hmat, Tmat, C):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = np.dot(Hmat.T, solve(np.eye(rows) / C + np.dot(Hmat, Hmat.T), Tmat))
    else:
        beta = solve(np.eye(cols) / C + np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
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


def dotfn():
    xt = T.fmatrix()
    yt = T.fmatrix()
    dotxy = T.dot(xt, yt)
    dot = theano.function([xt, yt], dotxy, allow_input_downcast=True)
    return dot


dot = dotfn()


def dot_decomp(xnp, ynp):
    size = xnp.shape[0]
    batchSize = 4096
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(dot(xnp[start:end], ynp))
    return np.concatenate(result, axis=0)


# def dot_self(xnp, ynp):
#     subsize = 100000
#     maxdim = np.max(xnp.shape)
#     parts = maxdim // subsize + 1  # 分块矩阵的分块数
#     index = [subsize * i for i in range(1, parts)]
#     xparts = np.split(xnp, index, axis=1)
#     yparts = np.split(ynp, index, axis=0)
#     partsum = []
#     for x, y in zip(xparts, yparts):
#         partsum.append(dot(x, y))
#     return sum(partsum)


def dotbiasactfn():
    xt = T.fmatrix()
    yt = T.fmatrix()
    bt = T.fvector()
    dotbiasxy = T.dot(xt, yt) + bt.dimshuffle('x', 0)
    dotbiasactxy = T.tanh(dotbiasxy)
    dotbiasact = theano.function([xt, yt, bt], dotbiasactxy, allow_input_downcast=True)
    return dotbiasact


dotbiasact = dotbiasactfn()


def dotbiasact_decomp(xnp, ynp, bnp):
    size = xnp.shape[0]
    batchSize = 4096
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(dotbiasact(xnp[start:end], ynp, bnp))
    return np.concatenate(result, axis=0)


# def convactfn(pad, stride):
#     xt = T.ftensor4()
#     ft = T.ftensor4()
#     convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=stride, filter_flip=False)
#     convxf = T.tanh(convxf)
#     conv2d = theano.function([xt, ft], convxf, allow_input_downcast=True)
#     return conv2d
#
#
# def convact_decomp(xnp, fnp, pad=0, stride=(1, 1)):
#     convact = convactfn(pad, stride)
#     size = xnp.shape[0]
#     batchSize = 2048
#     startRange = range(0, size - batchSize + 1, batchSize)
#     endRange = range(batchSize, size + 1, batchSize)
#     if size % batchSize != 0:
#         startRange.append(size - size % batchSize)
#         endRange.append(size)
#     result = []
#     for start, end in zip(startRange, endRange):
#         result.append(convact(xnp[start:end], fnp))
#     return np.concatenate(result, axis=0)
#
#
# def convbiasactfn(pad, stride):
#     xt = T.ftensor4()
#     ft = T.ftensor4()
#     bt = T.fvector()
#     convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=stride, filter_flip=False) \
#              + bt.dimshuffle('x', 0, 'x', 'x')
#     convxf = T.tanh(convxf)
#     conv2d = theano.function([xt, ft, bt], convxf, allow_input_downcast=True)
#     return conv2d
#
#
# def convbiasact_decomp(xnp, fnp, bnp, pad=0, stride=(1, 1)):
#     convbiasact = convbiasactfn(pad, stride)
#     size = xnp.shape[0]
#     batchSize = 2048
#     startRange = range(0, size - batchSize + 1, batchSize)
#     endRange = range(batchSize, size + 1, batchSize)
#     if size % batchSize != 0:
#         startRange.append(size - size % batchSize)
#         endRange.append(size)
#     result = []
#     for start, end in zip(startRange, endRange):
#         result.append(convbiasact(xnp[start:end], fnp, bnp))
#     return np.concatenate(result, axis=0)


def poolfn(pool_size, ignore_border, stride, pad, mode):
    xt = T.tensor4()
    poolx = pool_2d(xt, pool_size, ignore_border=ignore_border, st=stride, padding=pad, mode=mode)
    pool = theano.function([xt], poolx, allow_input_downcast=True)
    return pool


def pool_decomp(xnp, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
    pool = poolfn(pool_size, ignore_border, stride, pad, mode)
    size = xnp.shape[0]
    batchSize = 16384
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(pool(xnp[start:end]))
    return np.concatenate(result, axis=0)


class ZCAWhiten(object):
    def __init__(self):
        self.mean = None
        self.whiten = None

    def _fit(self, X, regularization=1e-5):
        # X : 2d shape [n_samples, n_features]
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.dot(X.T, X) / (X.shape[0] - 1)
        U, S, _ = svd(cov)
        S = np.diag(1. / np.sqrt(S + regularization))
        self.whiten = np.dot(np.dot(U, S), U.T)

    def apply(self, X, fit=True):
        # X : 2d shape [n_samples, n_features]
        if fit:
            self._fit(X)
        return np.dot(X - self.mean, self.whiten)


def norm4d(X, regularization=1e-5):
    mean = np.mean(X, axis=(2, 3), keepdims=True)
    var = np.var(X, axis=(2, 3), keepdims=True)
    return (X - mean) / np.sqrt(var + regularization)


def norm2d(X, regularization=1e-5):
    # 每个样本对自己均值方差归一化
    mean = np.mean(X, axis=1, keepdims=True)
    var = np.var(X, axis=1, keepdims=True)
    return (X - mean) / np.sqrt(var + regularization)


class ELMAE(object):
    def __init__(self, C, hidden_unit, filter_size, stride):
        self.C = C
        self.hidden_unit = hidden_unit
        self.filter_size = filter_size
        self.pad = filter_size // 2
        self.stride = stride
        self.zca_whiten = ZCAWhiten()

    def _make_patches(self, inputX, fit=True, addpad=True):
        assert inputX.ndim == 4 and inputX.shape[1] == 1  # 只对一张特征图取patch
        if addpad:
            padding = ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad))
            inputX = np.pad(inputX, padding, mode='constant', constant_values=0)
        # 取图像的patch
        im2col = myUtils.basic.Im2ColOp(self.filter_size, self.stride)
        patches = im2col.transform(inputX)
        patches = patches.reshape((-1, self.filter_size ** 2))
        # 归一化并白化
        # patches = norm2d(patches)
        # patches = self.zca_whiten.apply(patches, fit=fit)
        return patches

    def forward(self, inputX, train=True):
        assert inputX.ndim == 4 and inputX.shape[1] == 1  # ELMAE的输入通道数必须为1，即只有一张特征图
        batches, channels, rows, cols = inputX.shape
        if train:
            patches = self._make_patches(inputX, fit=True, addpad=False)
            # 使用聚类获取每一类中不同的patch

            # 生成随机正交滤波器
            filters = init.GlorotNormal().sample((self.filter_size ** 2, self.hidden_unit))
            filters = orthonormalize(filters)
            bias = init.Normal().sample(self.hidden_unit)
            bias = orthonormalize(bias)
            # 卷积前向输出，和取patch时一致
            hiddens = dotbiasact_decomp(patches, filters, bias)
            # 计算beta
            self.beta = compute_beta(hiddens, patches, self.C).T
        patches = self._make_patches(inputX, fit=False, addpad=True)
        out = dot_decomp(patches, self.beta)
        out = out.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        return out


class Layer(object):
    def get_output_for(self, inputX, train=True):
        raise NotImplementedError


class InputLayer(Layer):
    def get_output_for(self, inputX, train=None):
        return inputX


class ELMAELayer(Layer):
    def __init__(self, C, hidden_unit, filter_size, stride):
        self.C = C
        self.hidden_unit = hidden_unit
        self.filter_size = filter_size
        self.stride = stride
        self.elmae_list = []

    def get_output_for(self, inputX, train=True):
        batches, channels, rows, cols = inputX.shape
        output = 0.
        for ch in xrange(channels):
            oneChannel = inputX[:, ch, :, :].reshape((batches, 1, rows, cols))
            if train:
                elmae = ELMAE(self.C, self.hidden_unit, self.filter_size, self.stride)
                self.elmae_list.append(elmae)
            else:
                elmae = self.elmae_list[ch]
            output += elmae.forward(oneChannel, train=train)
        output = np.tanh(output)  # 先将输出相加，再激活（同一般卷积）
        return output


class PoolLayer(Layer):
    def __init__(self, pool_size, ignore_border=False, stride=None, pad=(0, 0), mode='max'):
        self.pool_size = pool_size
        self.ignore_border = ignore_border
        self.stride = stride
        self.pad = pad
        self.mode = mode

    def get_output_for(self, inputX, train=None):
        output = pool_decomp(inputX, self.pool_size, self.ignore_border, self.stride, self.pad, self.mode)
        return output


class BNLayer(Layer):
    def __init__(self):
        self.mean = None
        self.var = None

    def get_output_for(self, inputX, train=True, regularization=1e-5):
        # 每个特征在整个训练集上归一化
        if train:
            self.mean = np.mean(inputX, axis=(0, 2, 3), keepdims=True)
            self.var = np.var(inputX, axis=(0, 2, 3), keepdims=True)
        return (inputX - self.mean) / np.sqrt(self.var + regularization)


class ZCAWhitenLayer(Layer):
    def __init__(self):
        self.zca_whiten = ZCAWhiten()

    def get_output_for(self, inputX, train=True):
        # 对所有通道进行白化
        inputX = inputX.reshape((inputX.shape[0], -1))
        return self.zca_whiten.apply(inputX, train).reshape(inputX.shape)


class Classifier(Layer):
    def __init__(self, C, hidden_unit):
        self.C = C
        self.hidden_unit = hidden_unit

    def get_output_for(self, inputX, inputy=None, train=True):
        if train:
            self.W = init.GlorotNormal().sample((inputX.shape[1], self.hidden_unit))
            self.b = init.Normal().sample(self.hidden_unit)
        H = dotbiasact_decomp(inputX, self.W, self.b)
        if train:
            self.beta = compute_beta(H, inputy, self.C)
        out = dot_decomp(H, self.beta)
        return out


class LRFELMAE(object):
    def __init__(self, C):
        self.C = C
        self.whitens = {}
        self.paramsAE = {}
        self.paramsC = {}

    def _build(self):
        net = OrderedDict()
        net['input'] = InputLayer()
        # layer1
        net['layer1'] = ELMAELayer(C=self.C, hidden_unit=32, filter_size=7, stride=1)
        # net['bn1'] = BNLayer()
        # net['zca1'] = ZCAWhitenLayer()
        net['pool1'] = PoolLayer(pool_size=(2, 2))
        # layer2
        net['layer2'] = ELMAELayer(C=self.C, hidden_unit=64, filter_size=7, stride=1)
        # net['bn2'] = BNLayer()
        # net['zca2'] = ZCAWhitenLayer()
        net['pool2'] = PoolLayer(pool_size=(2, 2))
        return net

    def _get_output_for(self, net, inputX, train):
        out = None
        for name, layer in net.iteritems():
            print 'add ' + name
            if name == 'input':
                out = layer.get_output_for(inputX, train=None)
            else:
                out = layer.get_output_for(out, train=train)
        return out

    def train(self, inputX, inputy):
        self.net = self._build()
        netout = self._get_output_for(self.net, inputX, train=True)
        netout = netout.reshape((netout.shape[0], -1))
        self.classifier = Classifier(self.C, hidden_unit=netout.shape[1] * 5)
        predout = self.classifier.get_output_for(netout, inputy, train=True)
        ypred = np.argmax(predout, axis=1)
        ytrue = np.argmax(inputy, axis=1)
        return np.mean(ypred == ytrue)

    def score(self, inputX, inputy):
        netout = self._get_output_for(self.net, inputX, train=False)
        netout = netout.reshape((netout.shape[0], -1))
        predout = self.classifier.get_output_for(netout, inputy, train=False)
        ypred = np.argmax(predout, axis=1)
        ytrue = np.argmax(inputy, axis=1)
        return np.mean(ypred == ytrue)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X = myUtils.pre.norm4d_per_sample_channel(tr_X)
    te_X = myUtils.pre.norm4d_per_sample_channel(te_X)
    zca_whiten = ZCAWhiten()
    tr_X = zca_whiten.apply(tr_X, fit=True)
    te_X = zca_whiten.apply(te_X, fit=False)
    model = LRFELMAE(C=10)
    print model.train(tr_X, tr_y)
    print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
