# coding:utf-8
'''
使用lasagne框架
使用了一个LRF层，和CNN作对比
BP更新LRF层
'''
import numpy as np
import theano
import theano.tensor as T
from numpy.linalg import solve
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
from lasagne.utils import floatX
import myUtils


class CNN(object):
    def __init__(self, C, lr):
        self.C = C
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.net = self._forward()
        params = layers.get_all_params(self.net['out'], trainable=True)
        netout = layers.get_output(self.net['out'])
        reg = regularization.regularize_network_params(self.net['out'], regularization.l2)
        reg /= layers.helper.count_params(self.net['out'])
        self.predictfn = theano.function([self.X], netout, allow_input_downcast=True)
        accrarcy = myUtils.basic.accuracy(netout, self.Y)
        self.scorefn = theano.function([self.X, self.Y], accrarcy, allow_input_downcast=True)
        crossentropy = objectives.categorical_crossentropy(netout, self.Y)
        cost = T.mean(crossentropy) + C * reg
        updatesDict = updates.nesterov_momentum(cost, params, lr, 0.9)
        self.trainfn = theano.function([self.X, self.Y], [cost, accrarcy],
                                       updates=updatesDict, allow_input_downcast=True)

    def _forward(self):
        net = {}
        net['input'] = layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.X)
        net['conv'] = layers.Conv2DLayer(net['input'], 10, (5, 5), W=init.Orthogonal())
        net['pool'] = layers.MaxPool2DLayer(net['conv'], (3, 3), stride=(1, 1), pad=(1, 1))
        net['flatten'] = layers.FlattenLayer(net['pool'])
        net['out'] = layers.DenseLayer(net['flatten'], 10, b=None, nonlinearity=nonlinearities.softmax)
        return net

    def train(self, inputX, inputY):
        return self.trainfn(inputX, inputY)

    def predict(self, inputX):
        return self.predictfn(inputX)

    def score(self, testX, testY):
        return self.scorefn(testX, testY)


class LRFELM(object):
    def __init__(self, C, lr):
        self.C = C
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.net = self._forward()
        params = layers.get_all_params(self.net['flatten'], trainable=True)
        netout = layers.get_output(self.net['out'])
        flattenout = layers.get_output(self.net['flatten'])
        reg = regularization.regularize_network_params(self.net['flatten'], regularization.l2)
        reg /= layers.helper.count_params(self.net['flatten'])
        self.flattenfn = theano.function([self.X], flattenout, allow_input_downcast=True)
        self.predictfn = theano.function([self.X], netout, allow_input_downcast=True)
        accrarcy = myUtils.basic.accuracy(netout, self.Y)
        self.scorefn = theano.function([self.X, self.Y], accrarcy, allow_input_downcast=True)
        self.sharedBeta = self.net['out'].get_params()[0]
        crossentropy = objectives.categorical_crossentropy(netout, self.Y)
        cost = T.mean(crossentropy) + C * reg
        updatesDict = updates.nesterov_momentum(cost, params, lr, 0.9)
        # 训练随机参数
        self.trainfn = theano.function([self.X, self.Y], [cost, accrarcy],
                                       updates=updatesDict, allow_input_downcast=True)

    def _forward(self):
        net = {}
        net['input'] = layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.X)
        net['conv'] = layers.Conv2DLayer(net['input'], 10, (5, 5), W=init.Orthogonal())
        net['pool'] = layers.MaxPool2DLayer(net['conv'], (3, 3), stride=(1, 1), pad=(1, 1))
        net['flatten'] = layers.FlattenLayer(net['pool'])
        net['out'] = layers.DenseLayer(net['flatten'], 10, b=None, nonlinearity=nonlinearities.softmax)
        return net

    def _computeBeta(self, Hmat, Tmat):
        rows, cols = Hmat.shape
        if rows <= cols:
            beta = Hmat.T.dot(solve(np.eye(rows) / self.C + Hmat.dot(Hmat.T), Tmat))
        else:
            beta = solve(np.eye(cols) / self.C + Hmat.T.dot(Hmat), Hmat.T.dot(Tmat))
        return beta

    def trainBeta(self, inputX, inputY):
        Hmat = self.flattenfn(inputX)
        Tmat = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
        Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1.
        beta = self._computeBeta(Hmat, Tmat)
        self.sharedBeta.set_value(floatX(beta))

    def train(self, inputX, inputY):
        return self.trainfn(inputX, inputY)

    def predict(self, inputX):
        return self.predictfn(inputX)

    def score(self, testX, testY):
        return self.scorefn(testX, testY)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = LRFELM(C=1, lr=0.01)
    for i in range(100):
        model.trainBeta(tr_X[:30000], tr_y[:30000])
        print model.train(tr_X[:30000], tr_y[:30000])
        print model.score(tr_X[:30000], tr_y[:30000])
        print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
