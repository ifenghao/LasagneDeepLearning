# coding:utf-8
'''
用两个求逆层
不应该用正向传播到最后的结果去反向计算，而应该用真实的标签作为输入反向计算
但是全连接的反向计算难以良好的恢复输入，因此放弃全连接反向计算
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

    def _forward(self):
        net = {}
        net['input'] = layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.X)
        net['dense1'] = layers.DenseLayer(net['input'], num_units=1024,
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
        Ypredict = self.predict(testX)
        ypredict = np.argmax(Ypredict, axis=1)
        ytrue = np.argmax(testY, axis=1)
        return np.mean(ypredict == ytrue)


class BPELM(object):
    def __init__(self, C, lr):
        self.C = C
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.net = self._forward()
        params = layers.get_all_params(self.net['dense1'])
        netout = layers.get_output(self.net['out'])
        dense2out = layers.get_output(self.net['dense2'])
        dense1out = layers.get_output(self.net['dense1'])
        outinvout = layers.get_output(self.net['outinv'])
        reg = regularization.regularize_network_params(self.net['dense1'], regularization.l2)
        reg /= layers.helper.count_params(self.net['dense1'])
        self.dense2fn = theano.function([self.X], dense2out, allow_input_downcast=True)
        self.dense1fn = theano.function([self.X], dense1out, allow_input_downcast=True)
        self.outinvfn = theano.function([self.X], outinvout, allow_input_downcast=True)
        self.predictfn = theano.function([self.X], netout, allow_input_downcast=True)
        self.sharedBeta3 = self.net['out'].get_params()[0]
        self.sharedBeta2 = self.net['dense2'].get_params()[0]
        crossentropy = objectives.categorical_crossentropy(netout, self.Y)
        cost = T.mean(crossentropy) + C * reg
        updatesDict = updates.nesterov_momentum(cost, params, lr, 0.9)
        accrarcy = myUtils.basic.accuracy(netout, self.Y)
        # 训练随机参数
        self.trainfn = theano.function([self.X, self.Y], [cost, accrarcy],
                                       updates=updatesDict, allow_input_downcast=True)

    def _forward(self):
        net = {}
        net['input'] = layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.X)
        net['dense1'] = layers.DenseLayer(net['input'], num_units=1024,
                                          W=init.GlorotUniform(), b=init.Constant(0.),
                                          nonlinearity=nonlinearities.rectify)
        net['dense2'] = layers.DenseLayer(net['dense1'], num_units=512,
                                          W=init.GlorotUniform(), b=None,
                                          nonlinearity=nonlinearities.rectify)
        net['out'] = layers.DenseLayer(net['dense2'], num_units=10,
                                          W=init.GlorotUniform(), b=None,
                                          nonlinearity=nonlinearities.softmax)
        net['outinv'] = layers.InverseLayer(net['out'], net['out'])
        return net

    def _computeBeta(self, Hmat, Tmat):
        rows, cols = Hmat.shape
        if rows <= cols:
            beta = Hmat.T.dot(solve(np.eye(rows) / self.C + Hmat.dot(Hmat.T), Tmat))
        else:
            beta = solve(np.eye(cols) / self.C + Hmat.T.dot(Hmat), Hmat.T.dot(Tmat))
        return beta

    def trainBeta3(self, inputX, inputY):
        Hmat = self.dense2fn(inputX)
        tinyfloat = np.finfo(theano.config.floatX).tiny
        Tmat = np.full_like(inputY, np.log(tinyfloat), dtype=theano.config.floatX, order='A')
        Tmat[np.arange(len(inputY)), np.argmax(inputY, axis=1)] = 0
        beta = self._computeBeta(Hmat, Tmat)
        self.sharedBeta3.set_value(floatX(beta))

    def trainBeta2(self, inputX):
        Hmat = self.dense1fn(inputX)
        Tmat = self.outinvfn(inputX)
        beta = self._computeBeta(Hmat, Tmat)
        self.sharedBeta2.set_value(floatX(beta))

    def train(self, inputX, inputY):
        return self.trainfn(inputX, inputY)

    def predict(self, inputX):
        return self.predictfn(inputX)

    def score(self, testX, testY):
        Ypredict = self.predict(testX)
        ypredict = np.argmax(Ypredict, axis=1)
        ytrue = np.argmax(testY, axis=1)
        return np.mean(ypredict == ytrue)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = BPELM(C=10, lr=0.01)
    for i in range(100):
        model.trainBeta3(tr_X[:60000], tr_y[:60000])
        model.trainBeta2(tr_X[:60000])
        print model.train(tr_X[:60000], tr_y[:60000])
        print model.score(tr_X[:60000], tr_y[:60000])
        print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
