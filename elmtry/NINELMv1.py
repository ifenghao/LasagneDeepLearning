# coding:utf-8
'''
使用单个NIN结构
cccp1层权重随机不变，反向计算cccp2层，BP更新conv层
采用了在线学习算法
结果：信息反向传播时缺失，求出的beta误差很大，导致损失函数不下降
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


class NIN(object):
    def __init__(self, C, lr):
        self.C = C
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.fnet = self._forward()
        params = layers.get_all_params(self.fnet['out'], trainable=True)
        netout = layers.get_output(self.fnet['out'])
        reg = regularization.regularize_network_params(self.fnet['out'], regularization.l2)
        reg /= layers.helper.count_params(self.fnet['out'])
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
        net['conv'] = layers.Conv2DLayer(net['input'], 32, (5, 5), W=init.Orthogonal())
        net['cccp1'] = layers.Conv2DLayer(net['conv'], 64, (1, 1), W=init.Orthogonal())
        net['cccp2'] = layers.Conv2DLayer(net['cccp1'], 10, (1, 1), W=init.Orthogonal())
        net['pool'] = layers.MaxPool2DLayer(net['cccp2'], (2, 2), stride=(2, 2), pad=(0, 0))
        net['gp'] = layers.GlobalPoolLayer(net['pool'])
        net['out'] = layers.NonlinearityLayer(net['gp'], nonlinearity=nonlinearities.softmax)
        return net

    def train(self, inputX, inputY):
        return self.trainfn(inputX, inputY)

    def predict(self, inputX):
        return self.predictfn(inputX)

    def score(self, testX, testY):
        return self.scorefn(testX, testY)


class BPNIN(object):
    def __init__(self, C, lr):
        self.C = C
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.fnet = self._forward()
        self.bnet = self._backward()
        params = layers.get_all_params(self.fnet['conv'], trainable=True)
        netout = layers.get_output(self.fnet['out'])
        cccp1out = layers.get_output(self.fnet['cccp1'])
        fshape = cccp1out.shape
        cccp1out = cccp1out.transpose((0, 2, 3, 1)).reshape((fshape[0] * fshape[2] * fshape[3], fshape[1]))
        uppoolout = layers.get_output(self.bnet['uppool'])
        bshape = uppoolout.shape
        uppoolout = uppoolout.transpose((0, 2, 3, 1)).reshape((bshape[0] * bshape[2] * bshape[3], bshape[1]))
        reg = regularization.regularize_network_params(self.fnet['conv'], regularization.l2)
        reg /= layers.helper.count_params(self.fnet['conv'])
        self.cccp1fn = theano.function([self.X], cccp1out, allow_input_downcast=True)
        self.predictfn = theano.function([self.X], netout, allow_input_downcast=True)
        self.backfn = theano.function([self.Y], uppoolout, allow_input_downcast=True)
        self.sharedBeta = self.fnet['cccp2'].get_params()[0]
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
        net['conv'] = layers.Conv2DLayer(net['input'], 32, (5, 5), W=init.Orthogonal())
        net['cccp1'] = layers.Conv2DLayer(net['conv'], 64, (1, 1), W=init.Orthogonal())
        net['cccp2'] = layers.Conv2DLayer(net['cccp1'], 10, (1, 1), W=init.Orthogonal())
        net['pool'] = layers.MaxPool2DLayer(net['cccp2'], (2, 2), stride=(2, 2), pad=(0, 0))
        net['gp'] = layers.GlobalPoolLayer(net['pool'])
        net['out'] = layers.NonlinearityLayer(net['gp'], nonlinearity=nonlinearities.softmax)
        return net

    def _backward(self):
        net = {}
        net['input'] = layers.InputLayer(shape=(None, 10), input_var=self.Y)
        net['input'] = layers.ReshapeLayer(net['input'], (-1, 10, 1, 1))
        net['uppool'] = layers.Upscale2DLayer(net['input'], 24)
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
    #     Hmat = self.cccp1fn(inputX)
    #     Tmat = self.backfn(inputY)
    #     beta = self._computeBeta(Hmat, Tmat)
    #     beta = beta.transpose((1, 0))
    #     beta = beta.reshape((beta.shape[0], beta.shape[1], 1, 1))
    #     self.sharedBeta.set_value(floatX(beta))

    def initBeta(self, inputX, inputY):
        H0 = self.cccp1fn(inputX)
        assert H0.shape[0] > H0.shape[1]
        Tmat = self.backfn(inputY)
        rows, cols = H0.shape  # 只允许行大于列
        self.K = H0.T.dot(H0) + np.eye(cols) / self.C  # 只在这里加入一次惩罚项
        self.beta = solve(self.K, H0.T.dot(Tmat))
        betatmp = self.beta.transpose((1, 0))
        betatmp = betatmp.reshape((betatmp.shape[0], betatmp.shape[1], 1, 1))
        self.sharedBeta.set_value(floatX(betatmp))

    def trainBeta(self, inputX, inputY):
        H = self.cccp1fn(inputX)
        assert H.shape[0] > H.shape[1]
        Tmat = self.backfn(inputY)
        self.K = self.K + H.T.dot(H)  # 惩罚项np.eye(cols) / self.C在这里不再加入
        self.beta = self.beta + solve(self.K, H.T.dot(Tmat - H.dot(self.beta)))
        betatmp = self.beta.transpose((1, 0))
        betatmp = betatmp.reshape((betatmp.shape[0], betatmp.shape[1], 1, 1))
        self.sharedBeta.set_value(floatX(betatmp))

    def train(self, inputX, inputY):
        return self.trainfn(inputX, inputY)

    def predict(self, inputX):
        return self.predictfn(inputX)

    def score(self, testX, testY):
        return self.scorefn(testX, testY)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = NIN(C=1, lr=0.01)
    minibatch = 1000
    for i in range(100):
        trcost = 0.
        treqs = 0
        count = 0
        for batch in myUtils.basic.miniBatchGen(tr_X[minibatch:], tr_y[minibatch:], minibatch, shuffle=False):
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
    model = BPNIN(C=1, lr=0.01)
    minibatch = 1000
    for i in range(100):
        model.initBeta(tr_X[:minibatch], te_y[:minibatch])
        trcost = 0.
        treqs = 0
        count = 0
        for batch in myUtils.basic.miniBatchGen(tr_X[minibatch:], tr_y[minibatch:], minibatch, shuffle=False):
            Xb, yb = batch
            model.trainBeta(Xb, yb)
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


if __name__ == '__main__':
    main2()
