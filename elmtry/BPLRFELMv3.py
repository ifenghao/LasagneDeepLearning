# coding:utf-8
'''
使用lasagne框架
使用了3个LRF层，和CNN作对比
BP更新所有的LRF层
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
        crossentropy = objectives.categorical_crossentropy(netout, self.Y)
        cost = T.mean(crossentropy) + C * reg
        updatesDict = updates.nesterov_momentum(cost, params, lr, 0.9)
        eqs = myUtils.basic.eqs(netout, self.Y)
        self.trainfn = theano.function([self.X, self.Y], [cost, eqs],
                                       updates=updatesDict, allow_input_downcast=True)
        self.scorefn = theano.function([self.X, self.Y], [cost, eqs], allow_input_downcast=True)

    def _forward(self):
        net = {}
        net['input'] = layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.X)
        net['conv1'] = layers.Conv2DLayer(net['input'], 32, (3, 3), W=init.Orthogonal(), pad=1)
        net['pool1'] = layers.MaxPool2DLayer(net['conv1'], (2, 2), stride=(2, 2))
        net['conv2'] = layers.Conv2DLayer(net['pool1'], 64, (3, 3), W=init.Orthogonal(), pad=1)
        net['pool2'] = layers.MaxPool2DLayer(net['conv2'], (2, 2), stride=(2, 2))
        net['conv3'] = layers.Conv2DLayer(net['pool2'], 128, (3, 3), W=init.Orthogonal(), pad=1)
        net['conv4'] = layers.Conv2DLayer(net['conv3'], 128, (3, 3), W=init.Orthogonal(), pad=1)
        net['pool3'] = layers.MaxPool2DLayer(net['conv4'], (2, 2), stride=(2, 2))
        net['flatten'] = layers.FlattenLayer(net['pool3'])
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
        self.sharedBeta = self.net['out'].get_params()[0]
        crossentropy = objectives.categorical_crossentropy(netout, self.Y)
        cost = T.mean(crossentropy) + C * reg
        updatesDict = updates.nesterov_momentum(cost, params, lr, 0.9)
        eqs = myUtils.basic.eqs(netout, self.Y)
        self.trainfn = theano.function([self.X, self.Y], [cost, eqs],
                                       updates=updatesDict, allow_input_downcast=True)
        self.scorefn = theano.function([self.X, self.Y], [cost, eqs], allow_input_downcast=True)

    def _forward(self):
        net = {}
        net['input'] = layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.X)
        net['conv1'] = layers.Conv2DLayer(net['input'], 32, (3, 3), W=init.Orthogonal(), pad=1)
        net['pool1'] = layers.MaxPool2DLayer(net['conv1'], (2, 2), stride=(2, 2))
        net['conv2'] = layers.Conv2DLayer(net['pool1'], 64, (3, 3), W=init.Orthogonal(), pad=1)
        net['pool2'] = layers.MaxPool2DLayer(net['conv2'], (2, 2), stride=(2, 2))
        net['conv3'] = layers.Conv2DLayer(net['pool2'], 128, (3, 3), W=init.Orthogonal(), pad=1)
        net['conv4'] = layers.Conv2DLayer(net['conv3'], 128, (3, 3), W=init.Orthogonal(), pad=1)
        net['pool3'] = layers.MaxPool2DLayer(net['conv4'], (2, 2), stride=(2, 2))
        net['flatten'] = layers.FlattenLayer(net['pool3'])
        net['out'] = layers.DenseLayer(net['flatten'], 10, b=None, nonlinearity=nonlinearities.softmax)
        return net

    # def _computeBeta(self, Hmat, Tmat):
    #     rows, cols = Hmat.shape
    #     if rows <= cols:
    #         beta = Hmat.T.dot(solve(np.eye(rows) / self.C + Hmat.dot(Hmat.T), Tmat))
    #     else:
    #         beta = solve(np.eye(cols) / self.C + Hmat.T.dot(Hmat), Hmat.T.dot(Tmat))
    #     return beta
    #
    # def trainBeta(self, inputX, inputY):
    #     Hmat = self.flattenfn(inputX)
    #     Tmat = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
    #     Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1.
    #     beta = self._computeBeta(Hmat, Tmat)
    #     self.sharedBeta.set_value(floatX(beta))

    def initBeta(self, inputX, inputY):
        Hmat0 = self.flattenfn(inputX)
        assert Hmat0.shape[0] >= Hmat0.shape[1]
        Tmat0 = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
        Tmat0[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1.
        rows, cols = Hmat0.shape  # 只允许行大于列
        self.K = Hmat0.T.dot(Hmat0) + np.eye(cols) / self.C  # 只在这里加入一次惩罚项
        self.beta = solve(self.K, Hmat0.T.dot(Tmat0))
        self.sharedBeta.set_value(floatX(self.beta))

    def trainBeta(self, inputX, inputY):
        Hmat = self.flattenfn(inputX)
        Tmat = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
        Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1.
        self.K = self.K + Hmat.T.dot(Hmat)  # 惩罚项np.eye(cols) / self.C在这里不再加入
        self.beta = self.beta + solve(self.K, Hmat.T.dot(Tmat - Hmat.dot(self.beta)))
        self.sharedBeta.set_value(floatX(self.beta))

    def train(self, inputX, inputY):
        return self.trainfn(inputX, inputY)

    def predict(self, inputX):
        return self.predictfn(inputX)

    def score(self, testX, testY):
        return self.scorefn(testX, testY)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = CNN(C=1, lr=0.01)
    minibatch = 1000
    for i in range(100):
        trcost = 0.
        treqs = 0
        count = 0
        for batch in myUtils.basic.miniBatchGen(tr_X, tr_y, minibatch, shuffle=False):
            Xb, yb = batch
            cost, eqs = model.train(Xb, yb)
            print cost, eqs, '\r',
            trcost += cost
            treqs += eqs
            count += 1
        print i, 'train cost', trcost / count, ' accuracy ', float(treqs) / len(tr_X)
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


def main2():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = LRFELM(C=1, lr=0.01)
    minibatch = 128*9
    for i in range(100):
        model.initBeta(tr_X[:minibatch], te_y[:minibatch])
        trcost = 0.
        treqs = 0
        count = 0
        for batch in myUtils.basic.miniBatchGen(tr_X[minibatch:], tr_y[minibatch:], minibatch, shuffle=False):
            Xb, yb = batch
            cost, eqs = model.train(Xb, yb)
            print cost, eqs, '\r',
            model.trainBeta(Xb, yb)
            trcost += cost
            treqs += eqs
            count += 1
        print i, 'train cost', trcost / count, ' accuracy ', float(treqs) / len(tr_X)
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
    main()
