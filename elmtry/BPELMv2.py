# coding:utf-8
'''
使用了lasagne框架
用一个求逆层
使用带有正则项的在线更新beta算法
'''
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy, softmax
from numpy.linalg import solve, pinv
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
from lasagne.utils import floatX
import myUtils


class MLP(object):
    def __init__(self, C, lr):
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.net = self._forward()
        params = layers.get_all_params(self.net['out'])
        netout = layers.get_output(self.net['out'])
        reg = regularization.regularize_network_params(self.net['out'], regularization.l2)
        reg /= layers.helper.count_params(self.net['out'])
        # 最后概率输出
        self.predictfn = theano.function([self.X], netout, allow_input_downcast=True)
        crossentropy = objectives.categorical_crossentropy(netout, self.Y)
        cost = T.mean(crossentropy) + C * reg
        updatesDict = updates.nesterov_momentum(cost, params, lr, 0.9)
        accrarcy = myUtils.basic.accuracy(netout, self.Y)
        # 训练随机参数
        self.trainfn = theano.function([self.X, self.Y], [cost, accrarcy],
                                       updates=updatesDict, allow_input_downcast=True)
        self.scorefn = theano.function([self.X, self.Y], [cost, accrarcy], allow_input_downcast=True)

    def _forward(self):
        net = {}
        net['input'] = layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.X)
        net['dense1'] = layers.DenseLayer(net['input'], num_units=800,
                                          W=init.GlorotUniform(), b=init.Constant(0.),
                                          nonlinearity=nonlinearities.rectify)
        net['dense2'] = layers.DenseLayer(net['dense1'], num_units=512,
                                          W=init.GlorotUniform(), b=init.Constant(0.),
                                          nonlinearity=nonlinearities.rectify)
        net['out'] = layers.DenseLayer(net['dense2'], num_units=10,
                                       W=init.GlorotUniform(), b=None,
                                       nonlinearity=nonlinearities.softmax)
        return net

    def train(self, inputX, inputY):
        return self.trainfn(inputX, inputY)

    def predict(self, inputX):
        return self.predictfn(inputX)

    def score(self, testX, testY):
        return self.scorefn(testX, testY)


class BPELM(object):
    def __init__(self, C, lr):
        self.C = C
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.net = self._forward()
        params = layers.get_all_params(self.net['dense2'])
        netout = layers.get_output(self.net['out'])
        forwardout = layers.get_output(self.net['dense2'])
        reg = regularization.regularize_network_params(self.net['dense2'], regularization.l2)
        reg /= layers.helper.count_params(self.net['dense2'])
        self.forwardfn = theano.function([self.X], forwardout, allow_input_downcast=True)
        self.predictfn = theano.function([self.X], netout, allow_input_downcast=True)
        self.sharedBeta = self.net['out'].get_params()[0]
        crossentropy = objectives.categorical_crossentropy(netout, self.Y)
        cost = T.mean(crossentropy) + C * reg
        updatesDict = updates.nesterov_momentum(cost, params, lr, 0.9)
        accrarcy = myUtils.basic.accuracy(netout, self.Y)
        # 训练随机参数
        self.trainfn = theano.function([self.X, self.Y], [cost, accrarcy],
                                       updates=updatesDict, allow_input_downcast=True)
        self.scorefn = theano.function([self.X, self.Y], [cost, accrarcy], allow_input_downcast=True)

    def _forward(self):
        net = {}
        net['input'] = layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.X)
        net['dense1'] = layers.DenseLayer(net['input'], num_units=800,
                                          W=init.GlorotUniform(), b=init.Constant(0.),
                                          nonlinearity=nonlinearities.rectify)
        net['dense2'] = layers.DenseLayer(net['dense1'], num_units=512,
                                          W=init.GlorotUniform(), b=init.Constant(0.),
                                          nonlinearity=nonlinearities.rectify)
        net['out'] = layers.DenseLayer(net['dense2'], num_units=10,
                                       W=init.GlorotUniform(), b=None,
                                       nonlinearity=nonlinearities.softmax)
        return net

    def _computeBeta(self, Hmat, Tmat):
        rows, cols = Hmat.shape
        if rows <= cols:
            beta = Hmat.T.dot(solve(np.eye(rows) / self.C + Hmat.dot(Hmat.T), Tmat))
        else:
            beta = solve(np.eye(cols) / self.C + Hmat.T.dot(Hmat), Hmat.T.dot(Tmat))
        return beta

    def trainBeta(self, inputX, inputY):
        Hmat = self.forwardfn(inputX)
        Tmat = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
        Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1
        beta = self._computeBeta(Hmat, Tmat)
        self.sharedBeta.set_value(floatX(beta))

    def train(self, inputX, inputY):
        return self.trainfn(inputX, inputY)

    def predict(self, inputX):
        return self.predictfn(inputX)

    def score(self, testX, testY):
        return self.scorefn(testX, testY)


class BPELMREG(object):
    def __init__(self, C, lr):
        self.C = C
        self.Hunits = 512
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.net = self._forward()
        params = layers.get_all_params(self.net['dense2'])
        netout = layers.get_output(self.net['out'])
        forwardout = layers.get_output(self.net['dense2'])
        reg = regularization.regularize_network_params(self.net['dense2'], regularization.l2)
        reg /= layers.helper.count_params(self.net['dense2'])
        self.forwardfn = theano.function([self.X], forwardout, allow_input_downcast=True)
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
        net['dense1'] = layers.DenseLayer(net['input'], num_units=800,
                                          W=init.GlorotUniform(), b=init.Constant(0.),
                                          nonlinearity=nonlinearities.rectify)
        net['dense2'] = layers.DenseLayer(net['dense1'], num_units=self.Hunits,
                                          W=init.GlorotUniform(), b=init.Constant(0.),
                                          nonlinearity=nonlinearities.rectify)
        net['out'] = layers.DenseLayer(net['dense2'], num_units=10,
                                       W=init.GlorotUniform(), b=None,
                                       nonlinearity=nonlinearities.softmax)
        return net

    # def initBeta(self, inputX, inputY):
    #     assert inputX.shape[0] >= self.Hunits
    #     H0 = self.forwardfn(inputX)
    #     rows, cols = H0.shape
    #     if rows <= cols:
    #         self.P = pinv(np.eye(rows) / self.C + H0.dot(H0.T))
    #         self.beta = H0.T.dot(solve(np.eye(rows) / self.C + H0.dot(H0.T), inputY))
    #     else:
    #         self.P = pinv(np.eye(cols) / self.C + H0.T.dot(H0))
    #         self.beta = solve(np.eye(cols) / self.C + H0.T.dot(H0), H0.T.dot(inputY))
    #     self.sharedBeta.set_value(floatX(self.beta))
    #
    # def trainBeta(self, inputX, inputY):
    #     H = self.forwardfn(inputX)
    #     Tmat = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
    #     Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1.
    #     size1 = self.P.shape[0]
    #     Ptmp = self.P - self.P.dot(solve(np.eye(size1) + self.P / self.C, self.P / self.C))
    #     size2 = H.shape[0]
    #     self.P = Ptmp - Ptmp.dot(H.T).dot(solve(np.eye(size2) + H.dot(Ptmp).dot(H.T), H.dot(Ptmp)))
    #     self.beta = self.beta + self.P.dot(H.T).dot(Tmat - H.dot(self.beta)) - self.P.dot(self.beta) / self.C
    #     self.sharedBeta.set_value(floatX(self.beta))

    def initBeta(self, inputX, inputY):
        assert inputX.shape[0] >= self.Hunits
        H0 = self.forwardfn(inputX)
        Tmat = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
        Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1.
        rows, cols = H0.shape  # 只允许行大于列
        self.K = H0.T.dot(H0)+np.eye(cols) / self.C# 只在这里加入一次惩罚项
        self.beta = solve(self.K, H0.T.dot(Tmat))
        self.sharedBeta.set_value(floatX(self.beta))

    def trainBeta(self, inputX, inputY):
        assert inputX.shape[0] >= self.Hunits
        H = self.forwardfn(inputX)
        Tmat = np.full_like(inputY, -1., dtype=theano.config.floatX, order='A')
        Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 1.
        self.K = self.K + H.T.dot(H)# 惩罚项np.eye(cols) / self.C在这里不再加入
        self.beta = self.beta + solve(self.K, H.T.dot(Tmat - H.dot(self.beta)))
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
    model = BPELM(C=10, lr=0.001)
    for i in range(100):
        model.trainBeta(tr_X[:60000], tr_y[:60000])
        trcost, traccrarcy = model.train(tr_X[:60000], tr_y[:60000])
        tecost, teaccrarcy = model.score(te_X, te_y)
        print i, 'train ', trcost, traccrarcy, 'test ', tecost, teaccrarcy


def main2():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = BPELMREG(C=10, lr=0.001)
    minibatch = 1000
    for i in range(100):
        model.initBeta(tr_X[:minibatch], te_y[:minibatch])
        trcost = 0.
        treqs = 0
        count = 0
        for batch in myUtils.basic.miniBatchGen(tr_X[minibatch:], tr_y[minibatch:], minibatch, shuffle=False):
            Xb, yb = batch
            model.trainBeta(Xb, yb)
        # 全部训练完beta后再BP训练
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


if __name__ == '__main__':
    main2()
