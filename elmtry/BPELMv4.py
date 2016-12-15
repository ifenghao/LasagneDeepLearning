# coding:utf-8
'''
先按照一般的BP训练3层网络，收敛后将输出矩阵重新使用ELM方法计算
结果：使用ELM计算后的网络效果改善，但是继续训练则网络恶化
与随机初始化输入矩阵的3层ELM网络比较，使用BP训练过的输入矩阵是否会有改善
结果：使用BP训练过的输入矩阵效果更好或者差不多
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
from copy import deepcopy


class BPELM(object):
    def __init__(self, C, lr):
        self.C = C
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.net = self._forward()
        params = layers.get_all_params(self.net['out'])
        netout = layers.get_output(self.net['out'])
        forwardout = layers.get_output(self.net['dense'])
        reg = regularization.regularize_network_params(self.net['out'], regularization.l2)
        reg /= layers.helper.count_params(self.net['out'])
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
        net['dense'] = layers.DenseLayer(net['input'], num_units=2048,
                                         W=init.GlorotUniform(), b=init.Constant(0.),
                                         nonlinearity=nonlinearities.rectify)
        net['out'] = layers.DenseLayer(net['dense'], num_units=10,
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


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = BPELM(C=1, lr=0.001)
    for i in range(100):
        trcost, traccrarcy = model.train(tr_X[:60000], tr_y[:60000])
        tecost, teaccrarcy = model.score(te_X, te_y)
        print i, 'train ', trcost, traccrarcy, 'test ', tecost, teaccrarcy
    print 'stop BP and recompute beta'
    model.trainBeta(tr_X[:60000], tr_y[:60000])
    trcost, traccrarcy = model.score(tr_X, tr_y)
    tecost, teaccrarcy = model.score(te_X, te_y)
    print 'train ', trcost, traccrarcy, 'test ', tecost, teaccrarcy
    print 'continue BP'
    for i in range(100):
        trcost, traccrarcy = model.train(tr_X[:60000], tr_y[:60000])
        tecost, teaccrarcy = model.score(te_X, te_y)
        print i, 'train ', trcost, traccrarcy, 'test ', tecost, teaccrarcy


def main2():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model1 = BPELM(C=1, lr=0.001)
    model2 = deepcopy(model1)
    print 'ELM result'
    model1.trainBeta(tr_X[:60000], tr_y[:60000])
    trcost, traccrarcy = model1.score(tr_X, tr_y)
    tecost, teaccrarcy = model1.score(te_X, te_y)
    print 'train ', trcost, traccrarcy, 'test ', tecost, teaccrarcy
    print 'BP and ELM result'
    for i in range(100):
        trcost, traccrarcy = model2.train(tr_X[:60000], tr_y[:60000])
        tecost, teaccrarcy = model2.score(te_X, te_y)
        print i, 'train ', trcost, traccrarcy, 'test ', tecost, teaccrarcy


if __name__ == '__main__':
    main2()
