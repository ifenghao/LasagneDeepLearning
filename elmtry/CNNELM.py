# coding:utf-8

import time
import theano
import theano.tensor as T
from sklearn import metrics
import numpy as np
import myUtils


class CNNELM(object):
    def __init__(self):
        self.X = T.tensor4('X')
        self.y = T.ivector('y')
        self.wconv = theano.shared(np.random.randn(64, 1, 3, 3).astype(theano.config.floatX), borrow=True)
        self.bconv = theano.shared(np.random.randn(64).astype(theano.config.floatX), borrow=True)
        beta = self._buildtrain()
        self.trainfn = theano.function([self.X, self.y], beta, allow_input_downcast=True)
        self.betatrained = T.fmatrix()
        out = self._buildpredict()
        self.predictfn = theano.function([self.X, self.betatrained], out, allow_input_downcast=True)

    def _buildtrain(self):
        forwardout = T.nnet.conv2d(self.X, self.wconv, border_mode='half') + self.bconv.dimshuffle('x', 0, 'x', 'x')
        forwardout = forwardout.transpose((0, 2, 3, 1))
        shape = forwardout.shape
        forwardout = forwardout.reshape((shape[0] * shape[1] * shape[2], shape[3]))
        target = self._reverseTarget().transpose((0, 2, 3, 1))
        target = target.reshape((shape[0] * shape[1] * shape[2], 10))
        beta = T.nlinalg.pinv(forwardout).dot(target)
        return beta

    def _reverseTarget(self):
        target = T.zeros((self.y.shape[0], 10, 28, 28))
        index0 = T.arange(self.y.shape[0])
        return T.set_subtensor(target[index0, self.y, :, :], T.ones((28, 28)))

    def _buildpredict(self):
        forwardout = T.nnet.conv2d(self.X, self.wconv, border_mode='half') + self.bconv.dimshuffle('x', 0, 'x', 'x')
        forwardout = forwardout.transpose((0, 2, 3, 1))
        shape = forwardout.shape
        forwardout = forwardout.reshape((shape[0] * shape[1] * shape[2], shape[3]))
        forwardout = T.dot(forwardout, self.betatrained)
        forwardout = forwardout.reshape((shape[0], shape[1], shape[2], 10)).transpose((0, 3, 1, 2))
        forwardout = T.mean(forwardout, axis=(2, 3))
        return forwardout

    def train(self, trainX, trainy):
        self.beta = self.trainfn(trainX, trainy)

    def predict(self, X):
        return self.predictfn(X, self.beta)

    def score(self, testX, testy):
        ypredict = self.predict(testX)
        ypredict = np.argmax(ypredict, axis=1)
        return np.sqrt(metrics.accuracy_score(testy, ypredict))


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist()
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    model = CNNELM()
    model.train(tr_X, tr_y)
    model.score(te_X, te_y)


if __name__ == '__main__':
    main()
