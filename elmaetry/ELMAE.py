# coding:utf-8
'''
基于ELM的自编码器
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


def add_ones(X):
    rows, cols = X.shape
    X = np.hstack((X, np.ones((rows, 1))))
    return X


def getBeta(Hmat, Tmat, C):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = Hmat.T.dot(solve(np.eye(rows) / C + Hmat.dot(Hmat.T), Tmat))
    else:
        beta = solve(np.eye(cols) / C + Hmat.T.dot(Hmat), Hmat.T.dot(Tmat))
    return beta


class ELMAE(object):
    def __init__(self, C):
        self.C = C
        self.X = T.fmatrix()
        self.Y = T.fmatrix()
        self.dotact = self._dotactfn()
        self.params = {}

    def _dotactfn(self):
        dotxy = T.dot(self.X, self.Y)
        dotxy = T.tanh(dotxy)
        dotfn = theano.function([self.X, self.Y], dotxy, allow_input_downcast=True)
        return dotfn

    def _forward(self, inputX, hidden_units):
        rows, cols = inputX.shape
        layer = layers.InputLayer(shape=(rows, cols), input_var=self.X)
        layer = layers.DenseLayer(layer, num_units=hidden_units,
                                  W=init.GlorotUniform(), b=init.Uniform(),
                                  nonlinearity=nonlinearities.tanh)
        Hout = layers.get_output(layer)
        forwardfn = theano.function([self.X], Hout, allow_input_downcast=True)
        return forwardfn(inputX)

    def _random_preturb(self, inputX, hidden_units):
        rows, cols = inputX.shape
        W = init.GlorotUniform().sample((cols, hidden_units))
        b = init.Uniform().sample(hidden_units)
        layer = layers.InputLayer(shape=(rows, cols), input_var=self.X)
        layer = layers.DenseLayer(layer, num_units=hidden_units, W=W, b=b,
                                  nonlinearity=nonlinearities.tanh)
        Hout = layers.get_output(layer)
        forwardfn = theano.function([self.X], Hout, allow_input_downcast=True)
        return forwardfn(inputX), W, b

    def train(self, inputX, inputy):
        # 1
        Hsub1 = self._forward(inputX, 1024)
        beta1 = getBeta(Hsub1, inputX, self.C)
        H1 = self.dotact(inputX, beta1.T)
        self.params['beta1'] = beta1
        # 2
        Hsub2 = self._forward(H1, 256)
        beta2 = getBeta(Hsub2, H1, self.C)
        H2 = self.dotact(H1, beta2.T)
        self.params['beta2'] = beta2
        # 3
        Hsub3 = self._forward(H2, 128)
        beta3 = getBeta(Hsub3, H2, self.C)
        H3 = self.dotact(H2, beta3.T)
        self.params['beta3'] = beta3
        # classifier
        H, randomW, randomb = self._random_preturb(H3, 2048)
        beta = getBeta(H, inputy, self.C)
        out = H.dot(beta)
        self.params['beta'] = beta
        self.params['randomW'] = randomW
        self.params['randomb'] = randomb
        return out

    def score(self, inputX, inputy):
        H1 = self.dotact(inputX, self.params['beta1'].T)
        H2 = self.dotact(H1, self.params['beta2'].T)
        H3 = self.dotact(H2, self.params['beta3'].T)
        H = self.dotact(H3, self.params['randomW']) + self.params['randomb']
        out = H.dot(self.params['beta'])
        predict = np.argmax(out, axis=1)
        true = np.argmax(inputy, axis=1)
        return np.mean(predict == true)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    tr_X, te_X = tr_X.reshape((-1, 784)), te_X.reshape((-1, 784))
    model = ELMAE(C=1)
    predict = model.train(tr_X, tr_y)
    predict = np.argmax(predict, axis=1)
    true = np.argmax(tr_y, axis=1)
    print np.mean(predict == true)
    print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
