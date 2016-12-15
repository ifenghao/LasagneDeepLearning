# coding:utf-8
'''
分类器过拟合问题明显
使用dropout进行模型平均
'''
import gc
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.neighbours import images2neibs
from numpy.linalg import solve
from scipy.linalg import orth, svd
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization
from lasagne.utils import floatX
from lasagne.theano_extensions.padding import pad as lasagnepad
from collections import OrderedDict
import myUtils


def compute_beta(Hmat, Tmat, C):
    rows, cols = Hmat.shape
    if rows <= cols:
        beta = np.dot(Hmat.T, solve(np.eye(rows) / C + np.dot(Hmat, Hmat.T), Tmat))
    else:
        beta = solve(np.eye(cols) / C + np.dot(Hmat.T, Hmat), np.dot(Hmat.T, Tmat))
    return beta


def relu(x):
    return 0.5 * (x + abs(x))


def dropout(X, p=0.5):
    retain_prob = 1. - p
    binomial = np.random.uniform(low=0., high=1., size=X.shape[1])
    binomial = np.asarray(binomial < retain_prob, dtype=theano.config.floatX)
    return X * binomial, binomial


class ELM(object):
    def __init__(self, C, hidden_unit):
        self.C = C
        self.hidden_unit = hidden_unit
        self.binomials = []
        self.betas = []

    def get_train_output(self, inputX, inputy):
        self.W = init.GlorotNormal().sample((inputX.shape[1], self.hidden_unit))
        self.b = init.Normal().sample(self.hidden_unit)
        H = np.dot(inputX, self.W) + self.b
        H = relu(H)
        self.beta = compute_beta(H, inputy, self.C)
        out = np.dot(H, self.beta)
        return out

    def get_train_output_ensemble(self, inputX, inputy, n=10):
        self.W = init.GlorotNormal().sample((inputX.shape[1], self.hidden_unit))
        self.b = init.Normal().sample(self.hidden_unit)
        outputs = []
        for _ in xrange(n):
            inputX, binomial1 = dropout(inputX, p=0.5)
            H = np.dot(inputX, self.W) + self.b
            H = relu(H)
            H, binomial2 = dropout(H, p=0.5)
            beta = compute_beta(H, inputy, self.C)
            out = np.dot(H, beta)
            outputs.append(np.copy(out))
            self.binomials.append((np.copy(binomial1), np.copy(binomial2)))
            self.betas.append(np.copy(beta))
        return outputs

    def get_test_output(self, inputX):
        H = np.dot(inputX, self.W) + self.b
        H = relu(H)
        out = np.dot(H, self.beta)
        return out

    def get_test_output_ensemble(self, inputX):
        outputs = []
        for binomial, beta in zip(self.binomials, self.betas):
            inputX *= binomial[0]
            H = np.dot(inputX, self.W) + self.b
            H = relu(H)
            H *= binomial[1]
            out = np.dot(H, beta)
            outputs.append(np.copy(out))
        return outputs

    def train(self, inputX, inputy):
        predout = self.get_train_output_ensemble(inputX, inputy)
        idx = np.arange(inputX.shape[0])
        vote = np.zeros_like(inputy, dtype=np.int)
        for pred in predout:
            ypred = np.argmax(pred, axis=1)
            vote[idx, ypred] += 1
        yvote = np.argmax(vote, axis=1)
        ytrue = np.argmax(inputy, axis=1)
        return np.mean(yvote == ytrue)

    def score(self, inputX, inputy):
        predout = self.get_test_output_ensemble(inputX)
        idx = np.arange(inputX.shape[0])
        vote = np.zeros_like(inputy, dtype=np.int)
        for pred in predout:
            ypred = np.argmax(pred, axis=1)
            vote[idx, ypred] += 1
        yvote = np.argmax(vote, axis=1)
        ytrue = np.argmax(inputy, axis=1)
        return np.mean(yvote == ytrue)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=True)
    tr_X = myUtils.pre.norm4d_per_sample_channel(tr_X)
    te_X = myUtils.pre.norm4d_per_sample_channel(te_X)
    tr_X = tr_X.reshape((-1, 784))
    te_X = te_X.reshape((-1, 784))
    model = ELM(C=0.1, hidden_unit=5000)
    print model.train(tr_X, tr_y)
    print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
