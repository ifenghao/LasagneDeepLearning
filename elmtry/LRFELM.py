# coding:utf-8
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from numpy.linalg import solve
from lasagne.utils import floatX
import myUtils


class LRFELM(object):
    def __init__(self, filterShape, C):
        self.C = C
        randFilters = myUtils.elm.convfilterinit(filterShape)  # 正交初始化权重
        self.sharedFilters = theano.shared(floatX(randFilters), borrow=True)
        self.X = T.ftensor4()
        self.forwardout = self._forward()  # 卷积池化后的输出
        self.forwardfn = theano.function([self.X], self.forwardout, allow_input_downcast=True)

    def _forward(self):  # 前向传播：卷积和池化并展开为2维
        convout = T.nnet.conv2d(self.X, self.sharedFilters, border_mode='valid')
        squreout = T.sqr(convout)
        poolout = pool.pool_2d(squreout, (3, 3), st=(1, 1), padding=(1, 1), ignore_border=True, mode='sum')
        sqrtout = T.sqrt(poolout)
        flatout = T.flatten(sqrtout, outdim=2)
        return flatout

    def _computeBeta(self, Hmat, Tmat):  # 使用solve而不用显式的inv操作
        rows, cols = Hmat.shape
        if rows <= cols:
            beta = Hmat.T.dot(solve(np.eye(rows) / self.C + Hmat.dot(Hmat.T), Tmat))
        else:
            beta = solve(np.eye(cols) / self.C + Hmat.T.dot(Hmat), Hmat.T.dot(Tmat))
        return beta

    def train(self, inputX, inputY):
        Hmat = self.forwardfn(inputX)
        beta = self._computeBeta(Hmat, inputY)
        self.sharedBeta = theano.shared(beta, borrow=True)

    def predict(self, inputX):
        assert hasattr(self, 'sharedBeta')
        predictout = self.forwardout.dot(self.sharedBeta)
        self.predictfn = theano.function([self.X], predictout, allow_input_downcast=True)
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
    model.train(tr_X[:1000], tr_y[:1000])
    print model.score(tr_X[:1000], tr_y[:1000])
    print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
