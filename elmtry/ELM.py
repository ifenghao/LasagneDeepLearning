# coding:utf-8
import time
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets
from sklearn import preprocessing
from sklearn import cross_validation


class ELMRegressor(object):
    def __init__(self, nhiddens, nfeature):
        self.nhiddens = nhiddens
        self.nfeature = nfeature
        self.whiddens = np.random.randn(nfeature, nhiddens)
        self.bhiddens = np.random.randn(nhiddens)

    def _forward(self, inputX, reinit=False):
        if reinit:
            self.whiddens = np.random.randn(self.nfeature, self.nhiddens)
            self.bhiddens = np.random.randn(self.nhiddens)
        H = np.dot(inputX, self.whiddens) + self.bhiddens
        H = np.tanh(H)
        return H

    def train(self, trainX, trainy):
        H = self._forward(trainX)
        self.wout = np.linalg.pinv(H).dot(trainy)

    def predict(self, inputX):
        H = self._forward(inputX)
        return np.dot(H, self.wout)

    def score(self, testX, testy):
        ypredict = self.predict(testX)
        return np.sqrt(metrics.mean_squared_error(testy, ypredict))


class ELMClassifier(ELMRegressor):
    def __init__(self, nhiddens, nfeature):
        super(ELMClassifier, self).__init__(nhiddens, nfeature)
        self.binarizer = LabelBinarizer(0, 1)  # 将类别标签转化为二值矩阵

    def train(self, trainX, trainy):
        ybin = self.binarizer.fit_transform(trainy)
        super(ELMClassifier, self).train(trainX, ybin)

    def predict(self, inputX):
        rawpredict = super(ELMClassifier, self).predict(inputX)
        classpredict = self.binarizer.inverse_transform(rawpredict)
        return classpredict

    def score(self, testX, testy):
        ypredict = self.predict(testX)
        return metrics.accuracy_score(testy, ypredict)


if __name__ == '__main__':
    stdsc = preprocessing.StandardScaler()
    # regression
    diabetes = datasets.load_diabetes()
    dbx, dby = stdsc.fit_transform(diabetes.data), diabetes.target
    dbx_train, dbx_test, dby_train, dby_test = cross_validation.train_test_split(dbx, dby, test_size=0.2)

    mrx, mry = datasets.make_regression(n_samples=20000, n_features=100)
    mrx_train, mrx_test, mry_train, mry_test = cross_validation.train_test_split(mrx, mry, test_size=0.2)
    # classification
    iris = datasets.load_iris()
    irx, iry = stdsc.fit_transform(iris.data), iris.target
    irx_train, irx_test, iry_train, iry_test = cross_validation.train_test_split(irx, iry, test_size=0.2)

    digits = datasets.load_digits()
    dgx, dgy = stdsc.fit_transform(digits.data / 16.0), digits.target
    dgx_train, dgx_test, dgy_train, dgy_test = cross_validation.train_test_split(dgx, dgy, test_size=0.2)

    tr_X, te_X, tr_y, te_y = dbx_train, dbx_test, dby_train, dby_test

    elmr = ELMRegressor(50, tr_X.shape[1])
    elmtrain = []
    elmtest = []
    for _ in xrange(10):
        startTime = time.time()
        elmr.train(tr_X, tr_y)
        print time.time() - startTime
        elmtrain.append(elmr.score(tr_X, tr_y))
        elmtest.append(elmr.score(te_X, te_y))

    print np.min(elmtrain), np.mean(elmtrain), np.min(elmtest), np.mean(elmtest)
    print elmtrain
    print elmtest

    tr_X, te_X, tr_y, te_y = irx_train, irx_test, iry_train, iry_test

    elmc = ELMClassifier(50, tr_X.shape[1])
    elmtrain = []
    elmtest = []
    for _ in xrange(10):
        startTime = time.time()
        elmc.train(tr_X, tr_y)
        print time.time() - startTime
        elmtrain.append(elmc.score(tr_X, tr_y))
        elmtest.append(elmc.score(te_X, te_y))

    print np.min(elmtrain), np.mean(elmtrain), np.min(elmtest), np.mean(elmtest)
    print elmtrain
    print elmtest
