# coding:utf-8
'''
基于ELM的局部感知自编码器
仿照多通道的卷积层，将上一层所有通道卷积后的特征图求和，得到下一层的一张特征图
--------
bug:验证了Im2ColOp的错误，以及使用卷积和点乘的结果的一致性，但是有一定误差
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
        layer, filters = self._addAELayer(inputX, 32, filtersize=(7, 7), aestride=(4, 4),
                                          convpad=3, convstride=(1, 1))
        self.paramsAE['l1_filters'] = filters
        # layer, filters = self._addAELayer(layer, 32, filtersize=(5, 5), aestride=(3, 3),
        #                                   convpad=2, convstride=(1, 1))
        # self.paramsAE['l2_filters'] = filters
        layer = pool_decomp(layer, (2, 2))
        layer, filters = self._addAELayer(layer, 64, filtersize=(7, 7), aestride=(4, 4),
                                          convpad=3, convstride=(1, 1))
        self.paramsAE['l3_filters'] = filters
        # layer, filters = self._addAELayer(layer, 64, filtersize=(5, 5), aestride=(3, 3),
        #                                   convpad=2, convstride=(1, 1))
        # self.paramsAE['l4_filters'] = filters
        layer = pool_decomp(layer, (2, 2))
        # output
        layer = layer.reshape((batches, -1))
        return layer

    def _addAELayer(self, inputX, hiddenunit, filtersize, aestride, convpad, convstride):
        # 对每个通道的特征图分别训练卷积核
        batches, channels, rows, cols = inputX.shape
        output = 0.
        filters=[]
        for ch in xrange(channels):
            oneChannel = inputX[:, ch, :, :].reshape((batches, 1, rows, cols))
            # 使用AE训练卷积核
            # 在取图像的patch时不做pad， 且stride可以取较大一些减少patch数量（论文中只随机取10个）
            beta = self._ELMAE(oneChannel, hiddenunit, filtersize, aestride)
            filters.append(beta)
            im2col = myUtils.basic.Im2ColOp(psize=filtersize[0], stride=convstride[0])
            patches = im2col.transform(oneChannel)  # 在取图像的patch时不做pad
            patches = patches.reshape((-1, np.prod(filtersize)))
            output += dot_decomp(patches, beta).reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        output=np.tanh(output)
        print 'add AE layer'
        return output, filters

    def _ELMAE(self, inputX, hiddenunit, filtersize, stride):
        assert inputX.shape[1] == 1  # ELMAE的输入通道数必须为1，即只有一张特征图
        # 取图像的patch
        im2col = myUtils.basic.Im2ColOp(psize=filtersize[0], stride=stride[0])
        patches = im2col.transform(inputX)  # 在取图像的patch时不做pad
        patches = patches.reshape((-1, np.prod(filtersize)))
        # 生成随机正交滤波器
        filters = init.GlorotNormal().sample((np.prod(filtersize),hiddenunit))
        filters = orthonormalize(filters)
        bias = init.Normal().sample(hiddenunit)
        bias = orthonormalize(bias)
        # 卷积前向输出，一定要pad=0和取patch时一致
        hiddens = dotbiasact_decomp(inputX, filters, bias)
        # 计算beta
        beta = compute_beta(hiddens, patches, self.C)
        del hiddens, patches
        return beta

    def train(self, inputX, inputy):
        layerout = self._buildAE(inputX)
        rows, cols = layerout.shape
        classifierunit = cols * 5
        W = init.GlorotNormal().sample((cols, classifierunit))
        b = init.Normal().sample(classifierunit)
        H = dotbiasact_decomp(layerout, W, b)
        del layerout
        beta = compute_beta(H, inputy, self.C)
        out = dot_decomp(H, beta)
        del H
        self.paramsC['W'] = W
        self.paramsC['b'] = b
        self.paramsC['beta'] = beta
        ypred = np.argmax(out, axis=1)
        ytrue = np.argmax(inputy, axis=1)
        return np.mean(ypred == ytrue)

    def score(self, inputX, inputy):
        batches, _, _, _ = inputX.shape
        layer = convact_decomp(inputX, self.paramsAE['l1_filters'], pad=3, stride=(1, 1))
        # layer = convact_decomp(layer, self.paramsAE['l2_filters'], pad=2, stride=(1, 1))
        layer = pool_decomp(layer, (2, 2))
        layer = convact_decomp(layer, self.paramsAE['l3_filters'], pad=3, stride=(1, 1))
        # layer = convact_decomp(layer, self.paramsAE['l4_filters'], pad=2, stride=(1, 1))
        layer = pool_decomp(layer, (2, 2))
        # classifier
        layer = layer.reshape((batches, -1))
        H = dotbiasact_decomp(layer, self.paramsC['W'], self.paramsC['b'])
        del layer
        out = dot_decomp(H, self.paramsC['beta'])
        del H
        # accuracy
        predict = np.argmax(out, axis=1)
        true = np.argmax(inputy, axis=1)
        return np.mean(predict == true)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = LRFELMAE(C=10)
    print model.train(tr_X, tr_y)
    print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
