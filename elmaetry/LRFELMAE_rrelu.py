# coding:utf-8
'''
基于ELM的局部感知自编码器
仿照多通道的卷积层，将上一层所有通道卷积后的特征图求和，得到下一层的一张特征图
使用rrelu效果一般
'''
import gc
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.neighbours import images2neibs
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from numpy.linalg import solve
from scipy.linalg import orth, svd
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
from lasagne.utils import floatX
from lasagne.theano_extensions.padding import pad as lasagnepad
from lasagne.random import get_rng
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


def im2col(inputX, fsize, stride, pad):
    assert inputX.ndim == 4
    X = T.tensor4()
    Xpad = lasagnepad(X, pad, batch_ndim=2)
    neibs = images2neibs(Xpad, (fsize, fsize), (stride, stride), 'ignore_borders')
    im2colfn = theano.function([X], neibs, allow_input_downcast=True)
    return im2colfn(inputX)


def rrelu(tensorin, shape, lower=0.3, upper=0.8, train=True):
    if not train or upper == lower:
        tensorout = T.nnet.relu(tensorin, (upper + lower) / 2.0)
    else:
        shape = list(shape)
        shared_axes = (0,) + tuple(range(2, len(shape)))
        for ax in shared_axes:
            shape[ax] = 1
        srng = RandomStreams(get_rng().randint(1, 2147462579))
        rnd = srng.uniform(tuple(shape), low=lower, high=upper, dtype=theano.config.floatX)
        rnd = T.addbroadcast(rnd, *shared_axes)
        tensorout = T.nnet.relu(tensorin, rnd)
    return tensorout


def dotfn():
    xt = T.fmatrix()
    yt = T.fmatrix()
    dotxy = T.dot(xt, yt)
    dot = theano.function([xt, yt], dotxy, allow_input_downcast=True)
    return dot


dot = dotfn()


def dot_decomp(xnp, ynp):
    size = xnp.shape[0]
    batchSize = 8192
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
    dotbiasxy = T.nnet.relu(dotbiasxy)
    dotbiasact = theano.function([xt, yt, bt], dotbiasxy, allow_input_downcast=True)
    return dotbiasact


dotbiasact = dotbiasactfn()


def dotbiasact_decomp(xnp, ynp, bnp):
    size = xnp.shape[0]
    batchSize = 8192
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(dotbiasact(xnp[start:end], ynp, bnp))
    return np.concatenate(result, axis=0)


def dotbiasrrelu_decomp(xnp, ynp, bnp, lower=0.3, upper=0.8, train=True):
    xt = T.fmatrix()
    yt = T.fmatrix()
    bt = T.fvector()
    dotbiasxy = T.dot(xt, yt) + bt.dimshuffle('x', 0)
    dotbiasxy = rrelu(dotbiasxy, [xnp.shape[0], ynp.shape[1]], lower, upper, train)
    dotbiasrrelu = theano.function([xt, yt, bt], dotbiasxy, allow_input_downcast=True)
    size = xnp.shape[0]
    batchSize = 8192
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(dotbiasrrelu(xnp[start:end], ynp, bnp))
    return np.concatenate(result, axis=0)


def convactfn(pad, stride):
    xt = T.ftensor4()
    ft = T.ftensor4()
    convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=stride, filter_flip=False)
    convxf = T.nnet.relu(convxf)
    conv2d = theano.function([xt, ft], convxf, allow_input_downcast=True)
    return conv2d


def convact_decomp(xnp, fnp, pad=0, stride=1):
    convact = convactfn(pad, (stride, stride))
    size = xnp.shape[0]
    batchSize = 8192
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(convact(xnp[start:end], fnp))
    return np.concatenate(result, axis=0)


def convrrelu_decomp(xnp, ynp, pad=0, stride=1, lower=0.3, upper=0.8, train=True):
    xt = T.ftensor4()
    ft = T.ftensor4()
    convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=(stride, stride), filter_flip=False)
    shape = myUtils.basic.convOutputShape(xnp.shape, ynp.shape, pad, (stride, stride))
    convxf = rrelu(convxf, shape, lower, upper, train)
    dotbiasrrelu = theano.function([xt, ft], convxf, allow_input_downcast=True)
    size = xnp.shape[0]
    batchSize = 8192
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(dotbiasrrelu(xnp[start:end], ynp))
    return np.concatenate(result, axis=0)


def convbiasactfn(pad, stride):
    xt = T.ftensor4()
    ft = T.ftensor4()
    bt = T.fvector()
    convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=stride, filter_flip=False) \
             + bt.dimshuffle('x', 0, 'x', 'x')
    convxf = T.nnet.relu(convxf)
    conv2d = theano.function([xt, ft, bt], convxf, allow_input_downcast=True)
    return conv2d


def convbiasact_decomp(xnp, fnp, bnp, pad=0, stride=1):
    convbiasact = convbiasactfn(pad, (stride, stride))
    size = xnp.shape[0]
    batchSize = 8192
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(convbiasact(xnp[start:end], fnp, bnp))
    return np.concatenate(result, axis=0)


def convbiasrrelu_decomp(xnp, ynp, bnp, pad=0, stride=1, lower=0.3, upper=0.8, train=True):
    xt = T.ftensor4()
    ft = T.ftensor4()
    bt = T.fvector()
    convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=(stride, stride), filter_flip=False) \
             + bt.dimshuffle('x', 0, 'x', 'x')
    shape = myUtils.basic.convOutputShape(xnp.shape, ynp.shape, pad, (stride, stride))
    convxf = rrelu(convxf, shape, lower, upper, train)
    dotbiasrrelu = theano.function([xt, ft, bt], convxf, allow_input_downcast=True)
    size = xnp.shape[0]
    batchSize = 8192
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(dotbiasrrelu(xnp[start:end], ynp, bnp))
    return np.concatenate(result, axis=0)


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


class Layer(object):
    def get_train_output_for(self, inputX):
        raise NotImplementedError

    def get_test_output_for(self, inputX):
        raise NotImplementedError


class ELMAELayer(Layer):
    def __init__(self, C, hidden_unit, filter_size, stride):
        self.C = C
        self.hidden_unit = hidden_unit
        self.filter_size = filter_size
        self.stride = stride

    def _get_beta(self, inputX, ch, addpad):
        # assert inputX.ndim == 4 and inputX.shape[1] == 1  # ELMAE的输入通道数必须为1，即只有一张特征图
        batches, channels, rows, cols = inputX.shape
        oneChannel = inputX[:, ch, :, :].reshape((batches, 1, rows, cols))
        # 使用聚类获取每一类中不同的patch

        # 生成随机正交滤波器
        filters = init.GlorotNormal().sample((self.hidden_unit, self.filter_size ** 2))
        filters = orthonormalize(filters)
        filters = filters.reshape((self.hidden_unit, 1, self.filter_size, self.filter_size))
        bias = init.Normal().sample(self.hidden_unit)
        bias = orthonormalize(bias)
        # 卷积前向输出，和取patch时一致
        pad = self.filter_size // 2 if addpad else 0
        stride = self.filter_size // 2 + 1
        hiddens = convbiasrrelu_decomp(oneChannel, filters, bias, pad=pad, stride=stride, train=True)
        hiddens = hiddens.transpose((0, 2, 3, 1)).reshape((-1, self.hidden_unit))
        # 随机跨通道取图像patch
        batchindex = np.arange(batches)
        channelindex = np.random.randint(channels, size=batches)
        randChannel = inputX[batchindex, channelindex, :, :].reshape((batches, 1, rows, cols))
        patches = im2col(randChannel, self.filter_size, stride=stride, pad=pad)
        # 计算beta
        beta = compute_beta(hiddens, patches, self.C)
        beta = beta.reshape((self.hidden_unit, 1, self.filter_size, self.filter_size))
        return beta

    def get_train_output_for(self, inputX):
        batches, channels, rows, cols = inputX.shape
        betalist = []
        # batchindex = np.arange(batches)
        for ch in xrange(channels):
            # channelindex = np.random.randint(channels, size=batches)
            # oneChannel = inputX[:, ch, :, :].reshape((batches, 1, rows, cols))
            # oneChannel = inputX[batchindex, channelindex, :, :].reshape((batches, 1, rows, cols))
            betalist.append(self._get_beta(inputX, ch, addpad=False))
        self.filters = np.concatenate(betalist, axis=1)
        output = convrrelu_decomp(inputX, self.filters, pad=self.filter_size // 2, stride=self.stride, train=True)
        return output

    def get_test_output_for(self, inputX):
        output = convrrelu_decomp(inputX, self.filters, pad=self.filter_size // 2, stride=self.stride, train=False)
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


class ZCAWhitenLayer(Layer):  # 计算过慢不使用
    def __init__(self):
        self.zca_whiten = ZCAWhiten()

    def get_train_output_for(self, inputX):
        # 对所有通道进行白化
        origin_shape = inputX.shape
        inputX = inputX.reshape((origin_shape[0], -1))
        return self.zca_whiten.apply(inputX, True).reshape(origin_shape)

    def get_test_output_for(self, inputX):
        # 对所有通道进行白化
        origin_shape = inputX.shape
        inputX = inputX.reshape((origin_shape[0], -1))
        return self.zca_whiten.apply(inputX, False).reshape(origin_shape)


class Classifier(Layer):
    def __init__(self, C, hidden_unit):
        self.C = C
        self.hidden_unit = hidden_unit

    def get_train_output_for(self, inputX, inputy=None):
        self.W = init.GlorotNormal().sample((inputX.shape[1], self.hidden_unit))
        self.b = init.Normal().sample(self.hidden_unit)
        H = dotbiasrrelu_decomp(inputX, self.W, self.b, train=True)
        self.beta = compute_beta(H, inputy, self.C)
        out = dot_decomp(H, self.beta)
        return out

    def get_test_output_for(self, inputX):
        H = dotbiasrrelu_decomp(inputX, self.W, self.b, train=False)
        out = dot_decomp(H, self.beta)
        return out


class LRFELMAE(object):
    def __init__(self, C):
        self.C = C

    def _build(self):
        net = OrderedDict()
        # layer1
        net['layer1'] = ELMAELayer(C=self.C, hidden_unit=32, filter_size=7, stride=1)
        net['bn1'] = BNLayer()
        # net['zca1'] = ZCAWhitenLayer()
        net['pool1'] = PoolLayer(pool_size=(2, 2))
        # layer2
        net['layer2'] = ELMAELayer(C=self.C, hidden_unit=64, filter_size=7, stride=1)
        net['bn2'] = BNLayer()
        # net['zca2'] = ZCAWhitenLayer()
        net['pool2'] = PoolLayer(pool_size=(2, 2))
        return net

    def _get_train_output(self, net, inputX):
        out = inputX
        for name, layer in net.iteritems():
            out = layer.get_train_output(out)
            print 'add ' + name
        return out

    def _get_test_output(self, net, inputX):
        out = inputX
        for name, layer in net.iteritems():
            out = layer.get_test_output(out)
        return out

    def train(self, inputX, inputy):
        self.net = self._build()
        netout = self._get_train_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        self.classifier = Classifier(self.C, hidden_unit=netout.shape[1] * 5)
        predout = self.classifier.get_train_output_for(netout, inputy)
        ypred = np.argmax(predout, axis=1)
        ytrue = np.argmax(inputy, axis=1)
        return np.mean(ypred == ytrue)

    def cv(self, tr_X, va_X, tr_y, va_y):
        self.net = self._build()
        netout = self._get_train_output(self.net, tr_X)
        netout = netout.reshape((netout.shape[0], -1))
        features = netout.shape[1]
        times = [1, 1.5, 2, 4, 8]
        hiddenunit = map(lambda x: x * features, times)
        Clist = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        vaopt = 0.
        for unit in hiddenunit:
            for C in Clist:
                classifier = Classifier(C, hidden_unit=unit)
                predout = classifier.get_train_output_for(netout, tr_y)
                ypred = np.argmax(predout, axis=1)
                ytrue = np.argmax(tr_y, axis=1)
                tr_acc = np.mean(ypred == ytrue)
                netoutva = self._get_test_output(self.net, va_X)
                netoutva = netoutva.reshape((netoutva.shape[0], -1))
                predout = classifier.get_test_output_for(netoutva)
                ypred = np.argmax(predout, axis=1)
                ytrue = np.argmax(va_y, axis=1)
                va_acc = np.mean(ypred == ytrue)
                print unit, C, tr_acc, va_acc
                if va_acc > vaopt:
                    vaopt = va_acc
                    self.classifier = classifier
                    print 'opt'

    def score(self, inputX, inputy):
        netout = self._get_test_output(self.net, inputX)
        netout = netout.reshape((netout.shape[0], -1))
        predout = self.classifier.get_test_output_for(netout)
        ypred = np.argmax(predout, axis=1)
        ytrue = np.argmax(inputy, axis=1)
        return np.mean(ypred == ytrue)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X = myUtils.pre.norm4d_per_sample_channel(tr_X)
    te_X = myUtils.pre.norm4d_per_sample_channel(te_X)
    # tr_X, va_X = tr_X[:-10000], tr_X[-10000:]
    # tr_y, va_y = tr_y[:-10000], tr_y[-10000:]
    # zca_whiten = ZCAWhiten()
    # tr_X = zca_whiten.apply(tr_X, fit=True)
    # te_X = zca_whiten.apply(te_X, fit=False)
    model = LRFELMAE(C=10)
    # print model.cv(tr_X, va_X, tr_y, va_y)
    print model.train(tr_X, tr_y)
    print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
