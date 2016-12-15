# coding:utf-8
'''
基于ELM的局部感知自编码器
增加inception结构，效果没有明显改进
增加的cccp层反而使得性能变差
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


def cccp(inputX, beta):
    batches, inchannels, rows, cols = inputX.shape
    inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, inchannels))
    cccpout = dot_decomp(inputX, beta)
    cccpout = cccpout.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
    return cccpout


class LRFELMAE(object):
    def __init__(self, C):
        self.C = C
        self.paramsAE = {}
        self.paramsC = {}

    def _buildAE(self, inputX):
        batches, _, _, _ = inputX.shape
        # layer1
        conv7, beta7 = self._addAELayer(inputX, 12, filtersize=(7, 7), aestride=(4, 4),
                                        convpad=3, convstride=(1, 1))
        self.paramsAE['l1_beta7'] = beta7
        conv5, beta5 = self._addAELayer(inputX, 16, filtersize=(5, 5), aestride=(3, 3),
                                        convpad=2, convstride=(1, 1))
        self.paramsAE['l1_beta5'] = beta5
        conv3, beta3 = self._addAELayer(inputX, 20, filtersize=(3, 3), aestride=(2, 2),
                                        convpad=1, convstride=(1, 1))
        self.paramsAE['l1_beta3'] = beta3
        layer = np.concatenate((conv7, conv5, conv3), axis=1)  # 将特征串联
        layer, betacccp = self._addCCCPLayer(layer, 16)
        self.paramsAE['l1_betacccp'] = betacccp
        del conv7, conv5, conv3
        _, _, rows, cols = layer.shape  # 将结果的channel合并到batch中
        layer = layer.reshape((-1, 1) + (rows, cols))
        layer = pool_decomp(layer, (2, 2))
        # 垃圾回收减少内存
        gc.collect()
        # layer2
        conv7, beta7 = self._addAELayer(layer, 8, filtersize=(7, 7), aestride=(4, 4),
                                        convpad=3, convstride=(1, 1))
        self.paramsAE['l2_beta7'] = beta7
        conv5, beta5 = self._addAELayer(layer, 10, filtersize=(5, 5), aestride=(3, 3),
                                        convpad=2, convstride=(1, 1))
        self.paramsAE['l2_beta5'] = beta5
        conv3, beta3 = self._addAELayer(layer, 12, filtersize=(3, 3), aestride=(2, 2),
                                        convpad=1, convstride=(1, 1))
        self.paramsAE['l2_beta3'] = beta3
        layer = np.concatenate((conv7, conv5, conv3), axis=1)  # 将特征串联
        layer, betacccp = self._addCCCPLayer(layer, 10)
        self.paramsAE['l2_betacccp'] = betacccp
        del conv7, conv5, conv3
        _, _, rows, cols = layer.shape  # 将结果的channel合并到batch中
        layer = layer.reshape((-1, 1) + (rows, cols))
        layer = pool_decomp(layer, (2, 2))
        # 垃圾回收减少内存
        gc.collect()
        # output
        layer = layer.reshape((batches, -1))
        return layer

    def _addAELayer(self, inputX, hiddenunit, filtersize, aestride, convpad, convstride):
        # 使用AE训练卷积核
        # 在取图像的patch时不做pad， 且stride可以取较大一些减少patch数量（论文中只随机取10个）
        beta = self._ELMAE(inputX, hiddenunit, filtersize, aestride)
        beta = beta.reshape((hiddenunit, 1) + filtersize)
        # 使用训练出来的卷积核进行前向传播
        # 在前向传播时使用pad，且stride取（1,1）保持大小不变
        convout = convact_decomp(inputX, beta, convpad, convstride)
        print 'add AE layer'
        return convout, beta

    def _ELMAE(self, inputX, hiddenunit, filtersize, stride):
        assert inputX.shape[1] == 1  # ELMAE的输入通道数必须为1，即只有一张特征图
        # 生成随机正交滤波器
        filters = init.GlorotNormal().sample((hiddenunit, np.prod(filtersize)))
        filters = orthonormalize(filters)
        filters = filters.reshape((hiddenunit, 1) + filtersize)
        bias = init.Normal().sample(hiddenunit)
        bias = orthonormalize(bias)
        # 卷积前向输出，一定要pad=0和取patch时一致
        convout = convbiasact_decomp(inputX, filters, bias, pad=0, stride=stride)
        del filters, bias
        # # 形状不改变的最大池化（改进点）
        # convout = pool_cpu(convout, (3, 3), stride=(1, 1), pad=(1, 1), ignore_border=True)
        # 将卷积的结果4维矩阵改变为2维
        hiddens = convout.transpose((0, 2, 3, 1)).reshape((-1, hiddenunit))
        del convout
        # 取图像的patch
        im2col = myUtils.basic.Im2ColOp(psize=filtersize[0], stride=stride[0])
        patches = im2col.transform(inputX)  # 在取图像的patch时不做pad
        patches = patches.reshape((-1, np.prod(filtersize)))
        # 计算beta
        beta = compute_beta(hiddens, patches, self.C)
        del hiddens, patches
        return beta

    def _addCCCPLayer(self, inputX, outchannels):
        batches, inchannels, rows, cols = inputX.shape
        inputX = inputX.transpose((0, 2, 3, 1)).reshape((-1, inchannels))
        W = init.GlorotNormal().sample((inchannels, outchannels))
        b = init.Normal().sample(outchannels)
        # AE前向计算隐层
        H = dotbiasact_decomp(inputX, W, b)
        del W, b
        # AE使用ELM计算输出矩阵
        beta = compute_beta(H, inputX, self.C).T
        # 前向输出CCCP结果
        cccpout = dot_decomp(inputX, beta)
        del inputX
        cccpout = cccpout.reshape((batches, rows, cols, -1)).transpose((0, 3, 1, 2))
        print 'add cccp layer'
        return cccpout, beta

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
        # layer1
        conv7 = convact_decomp(inputX, self.paramsAE['l1_beta7'], pad=3, stride=(1, 1))
        conv5 = convact_decomp(inputX, self.paramsAE['l1_beta5'], pad=2, stride=(1, 1))
        conv3 = convact_decomp(inputX, self.paramsAE['l1_beta3'], pad=1, stride=(1, 1))
        del inputX
        H1 = np.concatenate((conv7, conv5, conv3), axis=1)
        del conv7, conv5, conv3
        H1 = cccp(H1, self.paramsAE['l1_betacccp'])
        _, _, rows, cols = H1.shape  # 将结果的channel合并到batch中
        H1 = H1.reshape((-1, 1) + (rows, cols))
        H1 = pool_decomp(H1, (2, 2))
        # layer2
        conv7 = convact_decomp(H1, self.paramsAE['l2_beta7'], pad=3, stride=(1, 1))
        conv5 = convact_decomp(H1, self.paramsAE['l2_beta5'], pad=2, stride=(1, 1))
        conv3 = convact_decomp(H1, self.paramsAE['l2_beta3'], pad=1, stride=(1, 1))
        del H1
        H2 = np.concatenate((conv7, conv5, conv3), axis=1)
        del conv7, conv5, conv3
        H2 = cccp(H2, self.paramsAE['l2_betacccp'])
        _, _, rows, cols = H2.shape  # 将结果的channel合并到batch中
        H2 = H2.reshape((-1, 1) + (rows, cols))
        H2 = pool_decomp(H2, (2, 2))
        # classifier
        H2 = H2.reshape((batches, -1))
        H = dotbiasact_decomp(H2, self.paramsC['W'], self.paramsC['b'])
        del H2
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
