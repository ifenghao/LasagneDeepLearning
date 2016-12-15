# coding:utf-8
'''
取中间层作为输出，使用ELM计算输出矩阵
结果：中间层取的越靠前，训练准确率变化不大，但是测试准确率越低，如何得到网络的最合适的深度
cccp6: train:99.782 test:88.41
cccp5: train:99.6 test:87.49
conv3: train:99.24 test:87.13
'''
import time
import cPickle
import os
from numpy.linalg import solve
import numpy as np
import theano
import theano.tensor as T
from lasagne import layers
from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer, NonlinearityLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
from lasagne.utils import floatX
import myUtils


def buildmodel(x):
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32), input_var=x)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp1'] = ConvLayer(
        net['conv1'], num_filters=160, filter_size=1, flip_filters=False)
    net['cccp2'] = ConvLayer(
        net['cccp1'], num_filters=96, filter_size=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['cccp2'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['drop3'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv2'] = ConvLayer(net['drop3'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp3'] = ConvLayer(
        net['conv2'], num_filters=192, filter_size=1, flip_filters=False)
    net['cccp4'] = ConvLayer(
        net['cccp3'], num_filters=192, filter_size=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['cccp4'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['drop6'] = DropoutLayer(net['pool2'], p=0.5)
    net['conv3'] = ConvLayer(net['drop6'],
                             num_filters=192,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['cccp5'] = ConvLayer(
        net['conv3'], num_filters=192, filter_size=1, flip_filters=False)
    net['cccp6'] = ConvLayer(
        net['cccp5'], num_filters=10, filter_size=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['cccp6'],
                             pool_size=8,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['out'] = NonlinearityLayer(FlattenLayer(net['pool3']), nonlinearity=nonlinearities.softmax)
    net['dense'] = layers.DenseLayer(net['cccp6'], 10, b=None, nonlinearity=nonlinearities.softmax)
    return net


class NINELM(object):
    def __init__(self, C):
        self.C = C
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.fnet = buildmodel(self.X)
        netout = layers.get_output(self.fnet['dense'])
        internalout = layers.get_output(self.fnet['pool2'])
        internalout = T.flatten(internalout, outdim=2)
        self.internalfn = theano.function([self.X], internalout, allow_input_downcast=True)
        self.sharedBeta = self.fnet['dense'].get_params()[0]
        crossentropy = objectives.categorical_crossentropy(netout, self.Y)
        cost = T.mean(crossentropy)
        eqs = myUtils.basic.eqs(netout, self.Y)
        self.scorefn = theano.function([self.X, self.Y], [cost, eqs], allow_input_downcast=True)

    def initBeta(self, inputX, inputY):
        # assert inputX.shape[0] >= self.Hunits
        H0 = self.internalfn(inputX)
        Tmat = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
        Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1.
        rows, cols = H0.shape  # 只允许行大于列
        self.K = H0.T.dot(H0) + np.eye(cols) / self.C  # 只在这里加入一次惩罚项
        self.beta = solve(self.K, H0.T.dot(Tmat))
        self.sharedBeta.set_value(floatX(self.beta))

    def trainBeta(self, inputX, inputY):
        # assert inputX.shape[0] >= self.Hunits
        H = self.internalfn(inputX)
        Tmat = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
        Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1.
        self.K = self.K + H.T.dot(H)  # 惩罚项np.eye(cols) / self.C在这里不再加入
        self.beta = self.beta + solve(self.K, H.T.dot(Tmat - H.dot(self.beta)))
        self.sharedBeta.set_value(floatX(self.beta))

    def score(self, testX, testY):
        return self.scorefn(testX, testY)

    def loadParams(self, filename):
        valueList = cPickle.load(open(filename, 'r'))
        layers.set_all_param_values(self.fnet['out'], valueList)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.pre.cifarWhiten('cifar10')
    tr_y = myUtils.load.one_hot(tr_y, 10)
    te_y = myUtils.load.one_hot(te_y, 10)
    model = NINELM(C=1)
    model.loadParams('/home/zhufenghao/pretrained_models/cifar10_nin.pkl')
    model.trainBeta(tr_X[:60000], tr_y[:60000])
    trcost, traccrarcy = model.score(tr_X, tr_y)
    tecost, teaccrarcy = model.score(te_X, te_y)
    print 'train ', trcost, traccrarcy, 'test ', tecost, teaccrarcy


def main2():
    tr_X, te_X, tr_y, te_y = myUtils.pre.cifarWhiten('cifar10')
    tr_y = myUtils.load.one_hot(tr_y, 10)
    te_y = myUtils.load.one_hot(te_y, 10)
    model = NINELM(C=10)
    model.loadParams('/home/zhufenghao/pretrained_models/cifar10_nin.pkl')
    minibatch = 2000
    model.initBeta(tr_X[:minibatch], te_y[:minibatch])
    for batch in myUtils.basic.miniBatchGen(tr_X[minibatch:], tr_y[minibatch:], minibatch, shuffle=False):
        Xb, yb = batch
        model.trainBeta(Xb, yb)
    trcost = 0.
    treqs = 0
    count = 0
    for batch in myUtils.basic.miniBatchGen(tr_X, tr_y, minibatch, shuffle=False):
        Xb, yb = batch
        cost, eqs = model.score(Xb, yb)
        print cost, eqs, '\r',
        trcost += cost
        treqs += eqs
        count += 1
    print 'train cost', trcost / count, ' accuracy ', float(treqs) / len(tr_X)
    tecost = 0.
    teeqs = 0
    count = 0
    for batch in myUtils.basic.miniBatchGen(te_X, te_y, minibatch, shuffle=False):
        Xb, yb = batch
        cost, eqs = model.score(Xb, yb)
        print cost, eqs, '\r',
        tecost += cost
        teeqs += eqs
        count += 1
    print 'test cost', tecost / count, ' accuracy ', float(teeqs) / len(te_X)


if __name__ == '__main__':
    main2()
