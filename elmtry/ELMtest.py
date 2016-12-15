# coding:utf-8
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
import time
import theano
import theano.tensor as T
from lasagne import layers, init, nonlinearities, regularization, objectives, updates


class ELMRegressor(object):
    def __init__(self, nhiddens, nfeature, whiddens, bhiddens):
        self.nhiddens = nhiddens
        self.nfeature = nfeature
        self.whiddens = whiddens
        self.bhiddens = bhiddens

    def _forward(self, inputX):
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


class ELMRegressor_iter(object):
    def __init__(self, nhiddens, nfeature, whiddens, bhiddens):
        self.nhiddens = nhiddens
        self.nfeature = nfeature
        self.whiddens = whiddens
        self.bhiddens = bhiddens
        X = T.matrix()
        y = T.vector()
        network = self._build(X)
        self.params = network.get_params()  # 只取最后一层的参数
        reg = regularization.regularize_layer_params(network, regularization.l2)
        shapes = [p.get_value().shape for p in self.params]
        counts = [np.prod(shape) for shape in shapes]
        reg /= sum(counts)
        yout = layers.get_output(network)
        trerror = objectives.squared_error(yout, y)
        trCost = trerror.mean() + reg
        updatesDict = updates.nesterov_momentum(trCost, self.params, 0.01, 0.9)
        self.trainfn = theano.function([X, y], [trCost], updates=updatesDict, allow_input_downcast=True)
        self.predictfn = theano.function([X], [yout], allow_input_downcast=True)

    def _build(self, inputX):
        layer = layers.InputLayer(shape=(None, self.nfeature), input_var=inputX)
        layer = layers.DenseLayer(layer, num_units=self.nhiddens,
                                  W=init.Normal(mean=0, std=1), b=init.Normal(mean=0, std=1),
                                  nonlinearity=nonlinearities.tanh)
        layer = layers.DenseLayer(layer, num_units=1,
                                  W=init.Normal(mean=0, std=1), b=init.Normal(mean=0, std=1),
                                  nonlinearity=nonlinearities.tanh)
        return layer

    def train(self, trainX, trainy, iters=100):
        costlist = []
        for i in xrange(iters):
            costlist.append(self.trainfn(trainX, trainy))
        return costlist

    def predict(self, inputX):
        return self.predictfn(inputX)

    def score(self, testX, testy):
        ypredict = self.predict(testX)
        return np.sqrt(metrics.mean_squared_error(testy, ypredict))


class ELMRegressor_gpu():
    def __init__(self, nhiddens, nfeature, whiddens, bhiddens):
        self.nhiddens = nhiddens
        self.nfeature = nfeature
        self.whiddens = whiddens
        self.bhiddens = bhiddens
        X = T.matrix()
        y = T.vector()
        self.wshared = theano.shared(whiddens, borrow=True)
        self.bshared = theano.shared(bhiddens, borrow=True)
        G = T.dot(X, self.wshared) + self.bshared
        G = T.tanh(G)
        w_elm = T.nlinalg.pinv(G).dot(y)
        self.trainfn = theano.function([X, y], w_elm, allow_input_downcast=True)

    def train(self, trainX, trainy):
        self.w_elm = self.trainfn(trainX, trainy)

    def predict(self, X):
        G = np.tanh(X.dot(self.whiddens) + self.bhiddens)
        return G.dot(self.w_elm)

    def score(self, testX, testy):
        ypredict = self.predict(testX)
        return np.sqrt(metrics.mean_squared_error(testy, ypredict))


if __name__ == '__main__':
    stdsc = preprocessing.StandardScaler()
    diabetes = datasets.load_diabetes()
    dbx, dby = stdsc.fit_transform(diabetes.data), diabetes.target
    dbx_train, dbx_test, dby_train, dby_test = cross_validation.train_test_split(dbx, dby, test_size=0.2)

    mrx, mry = datasets.make_regression(n_samples=20000, n_features=100)
    mrx_train, mrx_test, mry_train, mry_test = cross_validation.train_test_split(mrx, mry, test_size=0.2)

    tr_X, te_X, tr_y, te_y = dbx_train, dbx_test, dby_train, dby_test

    whiddens = np.random.randn(tr_X.shape[1], 50)
    bhiddens = np.random.randn(50)

    elmr = ELMRegressor(50, tr_X.shape[1], whiddens, bhiddens)
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

    elmr = ELMRegressor_iter(50, tr_X.shape[1], whiddens, bhiddens)
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

    # elmr = ELMRegressor_gpu(500, tr_X, tr_y)
    # elmtrain = []
    # elmtest = []
    # for _ in xrange(10):
    #     startTime = time.time()
    #     elmr.fit()
    #     print time.time() - startTime
    #     trainpred = elmr.predict(tr_X)
    #     elmtrain.append(metrics.mean_absolute_error(tr_y, trainpred))
    #     testpred = elmr.predict(te_X)
    #     elmtest.append(metrics.mean_absolute_error(te_y, testpred))
    #
    # print np.min(elmtrain), np.mean(elmtrain), np.min(elmtest), np.mean(elmtest)

    # mlpr = MLPRegressor(hidden_layer_sizes=(500,), activation='tanh', max_iter=100, early_stopping=True)
    # mlptrain = []
    # mlptest = []
    # for _ in xrange(10):
    #     startTime = time.time()
    #     mlpr.fit(tr_X, tr_y)
    #     print time.time() - startTime
    #     trainpred = mlpr.predict(tr_X)
    #     mlptrain.append(metrics.mean_absolute_error(tr_y, trainpred))
    #     testpred = mlpr.predict(te_X)
    #     mlptest.append(metrics.mean_absolute_error(te_y, testpred))
    #
    # print np.min(mlptrain), np.mean(mlptrain), np.min(mlptest), np.mean(mlptest)
