# coding:utf-8
import time
import numpy as np
from numpy.linalg import pinv, solve
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets
from sklearn import preprocessing
from sklearn import cross_validation


class OSELMRegressor(object):
    def __init__(self, nhiddens, nfeature, C):
        self.C = C
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

    def initial(self, trainX, trainy):
        assert trainX.shape[0] >= self.nhiddens
        H0 = self._forward(trainX)
        rows, cols = H0.shape
        self.P = pinv(np.eye(cols) / self.C + H0.T.dot(H0))
        self.beta = solve(np.eye(cols) / self.C + H0.T.dot(H0), H0.T.dot(trainy))

    def train(self, trainX, trainy):
        assert trainX.shape[0] >= self.nhiddens
        H = self._forward(trainX)
        size1 = self.P.shape[0]
        Ptmp = self.P - self.P.dot(solve(np.eye(size1) + self.P / self.C, self.P / self.C))
        rows, cols = H.shape
        self.P = Ptmp - Ptmp.dot(H.T).dot(solve(np.eye(cols) + H.dot(Ptmp).dot(H.T), H.dot(Ptmp)))
        self.beta = self.beta + self.P.dot(H.T).dot(trainy - H.dot(self.beta)) - self.P.dot(self.beta) / self.C

    def predict(self, inputX):
        H = self._forward(inputX)
        return np.dot(H, self.beta)

    def score(self, testX, testy):
        ypredict = self.predict(testX)
        return np.sqrt(metrics.mean_squared_error(testy, ypredict))


class OSELMClassifier(OSELMRegressor):
    def __init__(self, nhiddens, nfeature, C):
        super(OSELMClassifier, self).__init__(nhiddens, nfeature, C)
        self.binarizer = LabelBinarizer(0, 1)  # 将类别标签转化为二值矩阵

    def initial(self, trainX, trainy):
        ybin = self.binarizer.fit_transform(trainy)
        super(OSELMClassifier, self).initial(trainX, ybin)

    def train(self, trainX, trainy):
        ybin = self.binarizer.fit_transform(trainy)
        super(OSELMClassifier, self).train(trainX, ybin)

    def predict(self, inputX):
        rawpredict = super(OSELMClassifier, self).predict(inputX)
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

    oselmr = OSELMRegressor(50, tr_X.shape[1], 1)
    elmtrain = []
    elmtest = []
    for _ in xrange(10):
        startTime = time.time()
        oselmr.initial(tr_X[:60], tr_y[:60])
        for i in xrange(60, tr_X.shape[0], 50):
            oselmr.train(tr_X[i:i + 50], tr_y[i:i + 50])
        print time.time() - startTime
        elmtrain.append(oselmr.score(tr_X, tr_y))
        elmtest.append(oselmr.score(te_X, te_y))

    print np.min(elmtrain), np.mean(elmtrain), np.min(elmtest), np.mean(elmtest)
    print elmtrain
    print elmtest

    tr_X, te_X, tr_y, te_y = irx_train, irx_test, iry_train, iry_test

    oselmc = OSELMClassifier(50, tr_X.shape[1], 1)
    elmtrain = []
    elmtest = []
    for _ in xrange(10):
        startTime = time.time()
        oselmc.initial(tr_X[:60], tr_y[:60])
        for i in xrange(60, tr_X.shape[0], 50):
            oselmc.train(tr_X[i:i + 50], tr_y[i:i + 50])
        print time.time() - startTime
        elmtrain.append(oselmc.score(tr_X, tr_y))
        elmtest.append(oselmc.score(te_X, te_y))

    print np.min(elmtrain), np.mean(elmtrain), np.min(elmtest), np.mean(elmtest)
    print elmtrain
    print elmtest
