# coding:utf-8
# import numpy as np
# import theano
# import theano.tensor as T
# from theano.tensor.signal.pool import pool_2d
# from theano.sandbox.neighbours import images2neibs
# from numpy.linalg import solve
# from scipy.linalg import orth, svd
# from lasagne import layers
# from lasagne import init
# from lasagne import nonlinearities
# from lasagne import objectives
# from lasagne import updates
# from lasagne import regularization
# from lasagne.utils import floatX
# from lasagne.theano_extensions.padding import pad as lasagnepad
# from collections import OrderedDict
# import myUtils
#
#
# def compute_beta(Hmat, Tmat, C):
#     rows, cols = Hmat.shape
#     if rows <= cols:
#         beta = np.dot(Hmat.T, solve(np.eye(rows) / C + np.dot(Hmat, Hmat.T), Tmat))
#     else:
#         beta = solve(np.eye(cols) / C + np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
#     return beta
#
#
# def orthonormalize(filters):
#     ndim = filters.ndim
#     if ndim != 2:
#         filters = np.expand_dims(filters, axis=0)
#     rows, cols = filters.shape
#     if rows >= cols:
#         orthonormal = orth(filters)
#     else:
#         orthonormal = orth(filters.T).T
#     if ndim != 2:
#         orthonormal = np.squeeze(orthonormal, axis=0)
#     return orthonormal
#
#
# def im2col(inputX, fsize, stride, pad):
#     assert inputX.ndim == 4
#     X = T.tensor4()
#     Xpad = lasagnepad(X, pad, batch_ndim=2)
#     neibs = images2neibs(Xpad, (fsize, fsize), (stride, stride), 'ignore_borders')
#     im2colfn = theano.function([X], neibs, allow_input_downcast=True)
#     return im2colfn(inputX)
#
# def convbiasactfn(pad, stride):
#     xt = T.ftensor4()
#     ft = T.ftensor4()
#     bt = T.fvector()
#     convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=stride, filter_flip=False) \
#              + bt.dimshuffle('x', 0, 'x', 'x')
#     convxf = T.nnet.relu(convxf)
#     conv2d = theano.function([xt, ft, bt], convxf, allow_input_downcast=True)
#     return conv2d
#
#
# def convbiasact_decomp(xnp, fnp, bnp, pad=0, stride=1):
#     convbiasact = convbiasactfn(pad, (stride, stride))
#     size = xnp.shape[0]
#     batchSize = 8192
#     startRange = range(0, size - batchSize + 1, batchSize)
#     endRange = range(batchSize, size + 1, batchSize)
#     if size % batchSize != 0:
#         startRange.append(size - size % batchSize)
#         endRange.append(size)
#     result = []
#     for start, end in zip(startRange, endRange):
#         result.append(convbiasact(xnp[start:end], fnp, bnp))
#     return np.concatenate(result, axis=0)
#
#
# def get_beta(inputX, ch):
#     hidden_unit=32
#     filter_size=7
#     # assert inputX.ndim == 4 and inputX.shape[1] == 1  # ELMAE的输入通道数必须为1，即只有一张特征图
#     batches, channels, rows, cols = inputX.shape
#     oneChannel = inputX[:, ch, :, :].reshape((batches, 1, rows, cols))
#     # 使用聚类获取每一类中不同的patch
#
#     # 生成随机正交滤波器
#     filters = init.GlorotNormal().sample((hidden_unit, filter_size ** 2))
#     filters = orthonormalize(filters)
#     filters = filters.reshape((hidden_unit, 1, filter_size, filter_size))
#     bias = init.Normal().sample(hidden_unit)
#     bias = orthonormalize(bias)
#     # 卷积前向输出，和取patch时一致
#     pad = 0
#     stride = 4
#     hiddens = convbiasact_decomp(oneChannel, filters, bias, pad=pad, stride=stride)
#     hiddens = hiddens.transpose((0, 2, 3, 1)).reshape((-1, hidden_unit))
#     # 随机跨通道取图像patch
#     # batchindex = np.arange(batches)
#     # channelindex = np.random.randint(channels, size=batches)
#     # randChannel = inputX[batchindex, channelindex, :, :].reshape((batches, 1, rows, cols))
#     patches = im2col(oneChannel, filter_size, stride=stride, pad=pad)
#     # 计算beta
#     beta1 = compute_beta(hiddens, patches, 1)
#
#     # 卷积前向输出，和取patch时一致
#     pad = 0
#     stride = 1
#     hiddens = convbiasact_decomp(oneChannel, filters, bias, pad=pad, stride=stride)
#     hiddens = hiddens.transpose((0, 2, 3, 1)).reshape((-1, hidden_unit))
#     # 随机跨通道取图像patch
#     # batchindex = np.arange(batches)
#     # channelindex = np.random.randint(channels, size=batches)
#     # randChannel = inputX[batchindex, channelindex, :, :].reshape((batches, 1, rows, cols))
#     patches = im2col(oneChannel, filter_size, stride=stride, pad=pad)
#     # 计算beta
#     beta2 = compute_beta(hiddens, patches, 1)
#
#     return beta1,beta2

'''
基于ELM的局部感知自编码器
仿照多通道的卷积层，将上一层所有通道卷积后的特征图求和，得到下一层的一张特征图
'''
import gc
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from numpy.linalg import solve
from scipy.linalg import orth
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
from lasagne.utils import floatX
from lasagne.theano_extensions.padding import pad as lasagnepad
from theano.sandbox.neighbours import images2neibs
import myUtils


def compute_beta(Hmat, Tmat, C):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = Hmat.T.dot(solve(np.eye(rows) / C + Hmat.dot(Hmat.T), Tmat))
    else:
        beta = solve(np.eye(cols) / C + Hmat.T.dot(Hmat), Hmat.T.dot(Tmat))
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


def dotfn():
    xt = T.fmatrix()
    yt = T.fmatrix()
    dotxy = T.dot(xt, yt)
    dotact = theano.function([xt, yt], dotxy, allow_input_downcast=True)
    return dotact


dot = dotfn()


def dot_decomp(xnp, ynp):
    size = xnp.shape[0]
    batchSize = 2048
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(dot(xnp[start:end], ynp))
    return np.concatenate(result, axis=0)


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
    batchSize = 2048
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(dotbiasact(xnp[start:end], ynp, bnp))
    return np.concatenate(result, axis=0)


def convactfn(pad, stride):
    xt = T.ftensor4()
    ft = T.ftensor4()
    convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=stride, filter_flip=False)
    convxf = T.tanh(convxf)
    conv2d = theano.function([xt, ft], convxf, allow_input_downcast=True)
    return conv2d


def convact_decomp(xnp, fnp, pad=0, stride=(1, 1)):
    convact = convactfn(pad, stride)
    size = xnp.shape[0]
    batchSize = 2048
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(convact(xnp[start:end], fnp))
    return np.concatenate(result, axis=0)


def convbiasactfn(pad, stride):
    xt = T.ftensor4()
    ft = T.ftensor4()
    bt = T.fvector()
    convxf = T.nnet.conv2d(xt, ft, border_mode=pad, subsample=stride, filter_flip=False) \
             + bt.dimshuffle('x', 0, 'x', 'x')
    convxf = T.tanh(convxf)
    conv2d = theano.function([xt, ft, bt], convxf, allow_input_downcast=True)
    return conv2d


def convbiasact_decomp(xnp, fnp, bnp, pad=0, stride=(1, 1)):
    convbiasact = convbiasactfn(pad, stride)
    size = xnp.shape[0]
    batchSize = 2048
    startRange = range(0, size - batchSize + 1, batchSize)
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    result = []
    for start, end in zip(startRange, endRange):
        result.append(convbiasact(xnp[start:end], fnp, bnp))
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


class LRFELMAE(object):
    def __init__(self, C):
        self.C = C
        self.paramsAE = {}
        self.paramsC = {}

    def _buildAE(self, inputX):
        batches, _, _, _ = inputX.shape
        layer1, filters1, layer2, filters2 = self._addAELayer(inputX, 32, filtersize=(7, 7),
                                                              convpad=3, convstride=(1, 1))
        layer1 = pool_decomp(layer1, (2, 2))
        layer2 = pool_decomp(layer2, (2, 2))
        self.paramsAE['filters1'] = filters1
        self.paramsAE['filters2'] = filters2
        # output
        layer1 = layer1.reshape((batches, -1))
        layer2 = layer2.reshape((batches, -1))
        return layer1, layer2

    def _addAELayer(self, inputX, hiddenunit, filtersize, convpad, convstride):
        # 对每个通道的特征图分别训练卷积核
        batches, channels, rows, cols = inputX.shape
        filters1 = []
        filters2 = []
        for ch in xrange(channels):
            oneChannel = inputX[:, ch, :, :].reshape((batches, 1, rows, cols))
            # 使用AE训练卷积核
            # 在取图像的patch时不做pad， 且stride可以取较大一些减少patch数量（论文中只随机取10个）
            beta1, beta2 = self._ELMAE(oneChannel, hiddenunit, filtersize)
            beta1 = beta1.reshape((hiddenunit, 1) + filtersize)
            filters1.append(np.copy(beta1))
            beta2 = beta2.reshape((hiddenunit, 1) + filtersize)
            filters2.append(np.copy(beta2))
        # 所有通道卷积核串联
        filters1 = np.concatenate(filters1, axis=1)
        filters2 = np.concatenate(filters2, axis=1)
        # 使用训练出来的卷积核进行前向传播（注意：将上一层所有通道卷积后的特征图求和）
        # 在前向传播时使用pad，且stride取（1,1）保持大小不变
        convout1 = convact_decomp(inputX, filters1, convpad, convstride)
        convout2 = convact_decomp(inputX, filters2, convpad, convstride)
        print 'add AE layer'
        return convout1, filters1, convout2, filters2

    def _ELMAE(self, inputX, hiddenunit, filtersize):
        assert inputX.shape[1] == 1  # ELMAE的输入通道数必须为1，即只有一张特征图
        # 生成随机正交滤波器
        filters = init.GlorotNormal().sample((hiddenunit, np.prod(filtersize)))
        filters = orthonormalize(filters)
        filters = filters.reshape((hiddenunit, 1) + filtersize)
        bias = init.Normal().sample(hiddenunit)
        bias = orthonormalize(bias)
        # 卷积前向输出，一定要pad=0和取patch时一致
        stride = 4
        convout = convbiasact_decomp(inputX, filters, bias, pad=0, stride=(stride, stride))
        # 将卷积的结果4维矩阵改变为2维
        hiddens = convout.transpose((0, 2, 3, 1)).reshape((-1, hiddenunit))
        # 取图像的patch
        patches = im2col(inputX, filtersize[0], stride=stride, pad=0)
        # 计算beta
        beta1 = compute_beta(hiddens, patches, self.C)

        stride = 1
        convout = convbiasact_decomp(inputX, filters, bias, pad=0, stride=(stride, stride))
        # 将卷积的结果4维矩阵改变为2维
        hiddens = convout.transpose((0, 2, 3, 1)).reshape((-1, hiddenunit))
        # 取图像的patch
        patches = im2col(inputX, filtersize[0], stride=stride, pad=0)
        # 计算beta
        beta2 = compute_beta(hiddens, patches, self.C)
        return beta1, beta2

    def train(self, inputX, inputy):
        layerout1, layerout2 = self._buildAE(inputX)
        rows, cols = layerout1.shape
        classifierunit = cols * 5
        W = init.GlorotNormal().sample((cols, classifierunit))
        b = init.Normal().sample(classifierunit)
        H1 = dotbiasact_decomp(layerout1, W, b)
        beta1 = compute_beta(H1, inputy, self.C)
        out1 = dot_decomp(H1, beta1)

        H2 = dotbiasact_decomp(layerout2, W, b)
        beta2 = compute_beta(H2, inputy, self.C)
        out2 = dot_decomp(H2, beta2)

        self.paramsC['W'] = W
        self.paramsC['b'] = b
        self.paramsC['beta1'] = beta1
        self.paramsC['beta2'] = beta2
        ypred1 = np.argmax(out1, axis=1)
        ypred2 = np.argmax(out2, axis=1)
        ytrue = np.argmax(inputy, axis=1)
        return np.mean(ypred1 == ytrue), np.mean(ypred2 == ytrue)

    def score(self, inputX, inputy):
        batches, _, _, _ = inputX.shape
        layer1 = convact_decomp(inputX, self.paramsAE['filters1'], pad=3, stride=(1, 1))
        layer1 = pool_decomp(layer1, (2, 2))
        layer1 = layer1.reshape((batches, -1))
        H = dotbiasact_decomp(layer1, self.paramsC['W'], self.paramsC['b'])
        out1 = dot_decomp(H, self.paramsC['beta1'])

        layer2 = convact_decomp(inputX, self.paramsAE['filters2'], pad=3, stride=(1, 1))
        layer2 = pool_decomp(layer2, (2, 2))
        layer2 = layer2.reshape((batches, -1))
        H = dotbiasact_decomp(layer2, self.paramsC['W'], self.paramsC['b'])
        out2 = dot_decomp(H, self.paramsC['beta2'])

        # accuracy
        predict1 = np.argmax(out1, axis=1)
        predict2 = np.argmax(out2, axis=1)
        true = np.argmax(inputy, axis=1)
        return np.mean(predict1 == true), np.mean(predict2 == true)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = LRFELMAE(C=10)
    print model.train(tr_X, tr_y)
    print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
