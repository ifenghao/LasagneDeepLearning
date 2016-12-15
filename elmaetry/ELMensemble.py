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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.decomposition.pca import PCA
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


class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_hidden, C):
        self.n_hidden = n_hidden
        self.C = C

    def fit(self, X, y):
        y = myUtils.load.one_hot(y, len(np.unique(y)))
        self.W = init.GlorotNormal().sample((X.shape[1], self.n_hidden))
        self.b = init.Normal().sample(self.n_hidden)
        H = np.dot(X, self.W) + self.b
        H = relu(H)
        self.beta = compute_beta(H, y, self.C)
        return self

    def fit_transform(self, X, y):
        y = myUtils.load.one_hot(y, len(np.unique(y)))
        self.W = init.GlorotNormal().sample((X.shape[1], self.n_hidden))
        self.b = init.Normal().sample(self.n_hidden)
        H = np.dot(X, self.W) + self.b
        H = relu(H)
        self.beta = compute_beta(H, y, self.C)
        out = np.dot(H, self.beta)
        return out

    def predict(self, X):
        H = np.dot(X, self.W) + self.b
        H = relu(H)
        out = np.dot(H, self.beta)
        return out


class ELMensemble(object):
    def __init__(self, n_hidden, C, n_estimators):
        self.n_hidden = n_hidden
        self.C = C
        self.ensemble = BaggingClassifier(base_estimator=ELMClassifier(n_hidden=n_hidden, C=C), n_jobs=-1,
                                          n_estimators=n_estimators, max_samples=0.5, max_features=0.5,
                                          bootstrap=True, bootstrap_features=False, oob_score=False)

    def train(self, X, y):
        self.ensemble.fit(X, y)
        return self.ensemble.score(X, y)

    def score(self, X, y):
        return self.ensemble.score(X, y)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist(onehot=False)
    tr_X = myUtils.pre.norm4d_per_sample_channel(tr_X)
    te_X = myUtils.pre.norm4d_per_sample_channel(te_X)
    tr_X = tr_X.reshape((-1, 784))
    te_X = te_X.reshape((-1, 784))
    # tr_X = PCA(0.99).fit_transform(tr_X)
    # te_X = PCA(0.99).fit_transform(te_X)
    model = ELMensemble(C=0.1, n_hidden=5000, n_estimators=10)
    print model.train(tr_X, tr_y)
    print model.score(te_X, te_y)


if __name__ == '__main__':
    main()
