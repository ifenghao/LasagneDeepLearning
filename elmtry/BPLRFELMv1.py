# coding:utf-8
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy, softmax
from theano.tensor.signal import pool
from numpy.linalg import solve
from lasagne.utils import floatX
import myUtils
from myUtils import basicUtils, gradient


class LRFELM(object):
    def __init__(self, filterShape, C):
        self.C = C
        filterRand = myUtils.elm.convfilterinit(filterShape)
        self.filterShared = theano.shared(floatX(filterRand), borrow=True)
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.forwardout = self._forward()
        self.forwardfn = theano.function([self.X], self.forwardout, allow_input_downcast=True)
        self.sharedBeta = theano.shared(floatX(np.zeros((5760, 10))), borrow=True)
        predictout = self.forwardout.dot(self.sharedBeta)
        predictout = softmax(predictout)  # 输出必须有softmax限制在0和1之间
        self.predictfn = theano.function([self.X], predictout, allow_input_downcast=True)
        crossentropy = categorical_crossentropy(predictout, self.Y)
        cost = T.mean(crossentropy) + basicUtils.regularizer([self.filterShared])
        updates = gradient.sgdm(cost, [self.filterShared])
        self.trainfn = theano.function([self.X, self.Y], cost, updates=updates, allow_input_downcast=True)

    def _forward(self):
        convout = T.nnet.conv2d(self.X, self.filterShared, border_mode='valid')
        squreout = T.sqr(convout)
        poolout = pool.pool_2d(squreout, (3, 3), st=(1, 1), padding=(1, 1), ignore_border=True, mode='sum')
        sqrtout = T.sqrt(poolout)
        flatout = T.flatten(sqrtout, outdim=2)
        return flatout

    def _computeBeta(self, Hmat, Tmat):
        rows, cols = Hmat.shape
        if rows <= cols:
            beta = Hmat.T.dot(solve(np.eye(rows) / self.C + Hmat.dot(Hmat.T), Tmat))
        else:
            beta = solve(np.eye(cols) / self.C + Hmat.T.dot(Hmat), Hmat.T.dot(Tmat))
        return beta

    def trainBeta(self, inputX, inputY):
        Hmat = self.forwardfn(inputX)
        beta = self._computeBeta(Hmat, inputY)
        self.sharedBeta.set_value(beta)

    def trainRandFilters(self, inputX, inputY):
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
    model = LRFELM(filterShape=(10, 1, 5, 5), C=1)
    for i in range(10):
        model.trainBeta(tr_X[:10000], tr_y[:10000])
        print model.trainRandFilters(tr_X[:10000], tr_y[:10000])
        print model.score(tr_X[:10000], tr_y[:10000])
        print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
