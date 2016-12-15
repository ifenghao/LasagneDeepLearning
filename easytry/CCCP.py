# coding:utf-8
'''
全部使用1×1卷积核的CCCP层代替一般的3×3卷积层
'''
import time

import theano.tensor as T
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization

import myUtils


class Model(object):
    def __init__(self, lr, C, momentum):
        self.lr = lr
        self.C = C
        self.momentum = momentum
        self.X = T.tensor4('X')
        self.y = T.ivector('y')
        network = self._build()
        self.params = layers.get_all_params(network, trainable=True)
        reg = regularization.regularize_network_params(network, regularization.l2)
        reg /= layers.helper.count_params(network)
        # 训练集
        yDropProb = layers.get_output(network)
        self.trEqs = myUtils.basic.eqs(yDropProb, self.y)
        trCrossentropy = objectives.categorical_crossentropy(yDropProb, self.y)
        self.trCost = trCrossentropy.mean() + C * reg
        # 验证、测试集
        yFullProb = layers.get_output(network, deterministic=True)
        self.vateEqs = myUtils.basic.eqs(yFullProb, self.y)
        vateCrossentropy = objectives.categorical_crossentropy(yFullProb, self.y)
        self.vateCost = vateCrossentropy.mean() + C * reg
        self.yPred = yFullProb
        # 训练函数，输入训练集，输出训练损失和误差
        updatesDict = updates.nesterov_momentum(self.trCost, self.params, lr, momentum)
        self.trainfn = myUtils.basic.makeFunc([self.X, self.y], [self.trCost, self.trEqs], updatesDict)
        # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
        self.vatefn = myUtils.basic.makeFunc([self.X, self.y], [self.vateCost, self.vateEqs], None)

    def _build(self):
        layer = layers.InputLayer(shape=(None, 1, 28, 28), input_var=self.X)
        layer = layers.Conv2DLayer(layer, num_filters=128, filter_size=(1, 1), stride=(1, 1), pad='same',
                                   untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                                   nonlinearity=nonlinearities.rectify)
        layer = layers.Conv2DLayer(layer, num_filters=128, filter_size=(1, 1), stride=(1, 1), pad='same',
                                   untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                                   nonlinearity=nonlinearities.rectify)
        layer = layers.Pool2DLayer(layer, pool_size=(2, 2), stride=None, pad=(0, 0), ignore_border=False,
                                   mode='average_exc_pad')
        layer = layers.Conv2DLayer(layer, num_filters=512, filter_size=(1, 1), stride=(1, 1), pad='same',
                                   untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                                   nonlinearity=nonlinearities.rectify)
        layer = layers.Conv2DLayer(layer, num_filters=512, filter_size=(1, 1), stride=(1, 1), pad='same',
                                   untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                                   nonlinearity=nonlinearities.rectify)
        layer = layers.Pool2DLayer(layer, pool_size=(2, 2), stride=None, pad=(0, 0), ignore_border=False,
                                   mode='average_exc_pad')
        layer = layers.Conv2DLayer(layer, num_filters=2048, filter_size=(1, 1), stride=(1, 1), pad='same',
                                   untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                                   nonlinearity=nonlinearities.rectify)
        layer = layers.Conv2DLayer(layer, num_filters=2048, filter_size=(1, 1), stride=(1, 1), pad='same',
                                   untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                                   nonlinearity=nonlinearities.rectify)
        layer = layers.Pool2DLayer(layer, pool_size=(2, 2), stride=None, pad=(0, 0), ignore_border=False,
                                   mode='max')
        layer = layers.flatten(layer, outdim=2)  # 不加入展开层也可以，DenseLayer自动展开
        layer = layers.DropoutLayer(layer, p=0.5)
        layer = layers.DenseLayer(layer, num_units=256,
                                  W=init.GlorotUniform(), b=init.Constant(0.),
                                  nonlinearity=nonlinearities.rectify)
        layer = layers.DropoutLayer(layer, p=0.5)
        layer = layers.DenseLayer(layer, num_units=10,
                                  W=init.GlorotUniform(), b=init.Constant(0.),
                                  nonlinearity=nonlinearities.softmax)
        return layer

    def trainModel(self, tr_X, va_X, tr_y, va_y, batchSize=128, maxIter=100, verbose=True,
                   start=5, period=2, threshold=10, earlyStopTol=2, totalStopTol=2):
        trainfn = self.trainfn
        validatefn = self.vatefn
        lr = self.lr

        earlyStop = myUtils.basic.earlyStopGen(start, period, threshold, earlyStopTol)
        earlyStop.next()  # 初始化生成器
        totalStopCount = 0
        for epoch in xrange(maxIter):  # every epoch
            # In each epoch, we do a full pass over the training data:
            trEqCount = 0
            trCostSum = 0.
            startTime = time.time()
            for batch in myUtils.basic.miniBatchGen(tr_X, tr_y, batchSize, shuffle=True):
                Xb, yb = batch
                trCost, trEqs = trainfn(Xb, yb)
                trCostSum += trCost
                trEqCount += trEqs
            trIter = len(tr_X) // batchSize
            if len(tr_X) % batchSize != 0: trIter += 1
            trCostMean = trCostSum / trIter
            trAccuracy = float(trEqCount) / len(tr_X)
            # And a full pass over the validation data:
            vaEqCount = 0
            vaCostSum = 0.
            for batch in myUtils.basic.miniBatchGen(va_X, va_y, batchSize, shuffle=False):
                Xb, yb = batch
                vaCost, vaEqs = validatefn(Xb, yb)
                vaCostSum += vaCost
                vaEqCount += vaEqs
            vaIter = len(va_X) // batchSize
            if len(va_X) % batchSize != 0: vaIter += 1
            vaCostMean = vaCostSum / vaIter
            vaAccuracy = float(vaEqCount) / len(va_X)
            if verbose:
                print 'epoch ', epoch, ' time: %.3f' % (time.time() - startTime),
                print 'trcost: %.5f  tracc: %.5f' % (trCostMean, trAccuracy),
                print 'vacost: %.5f  vaacc: %.5f' % (vaCostMean, vaAccuracy)
            # Then we decide whether to early stop:
            if earlyStop.send((trCostMean, vaCostMean)):
                lr /= 10  # 如果一次早停止发生，则学习率降低继续迭代
                updatesDict = updates.nesterov_momentum(self.trCost, self.params, lr, self.momentum)
                trainfn = myUtils.basic.makeFunc([self.X, self.y], [self.trCost, self.trEqs], updatesDict)
                totalStopCount += 1
                if totalStopCount > totalStopTol:  # 如果学习率降低仍然发生早停止，则退出迭代
                    print 'stop'
                    break
                if verbose: print 'learning rate decreases to ', lr

    def testModel(self, te_X, te_y, batchSize=128, verbose=True):
        testfn = self.vatefn
        teEqCount = 0
        teCostSum = 0.
        for batch in myUtils.basic.miniBatchGen(te_X, te_y, batchSize, shuffle=False):
            Xb, yb = batch
            vaCost, vaEqs = testfn(Xb, yb)
            teCostSum += vaCost
            teEqCount += vaEqs
        teIter = len(te_X) // batchSize
        if len(te_X) % batchSize != 0: teIter += 1
        teCostMean = teCostSum / teIter
        teAccuracy = float(teEqCount) / len(te_X)
        if verbose:
            print 'tecost: %.5f  teacc: %.5f' % (teCostMean, teAccuracy)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.load.mnist()
    tr_X, te_X = myUtils.pre.norm4d(tr_X, te_X)
    tr_X, va_X = tr_X[:-10000], tr_X[-10000:]
    tr_y, va_y = tr_y[:-10000], tr_y[-10000:]
    model = Model(lr=0.01, C=1, momentum=0.9)
    model.trainModel(tr_X, va_X, tr_y, va_y)
    model.testModel(te_X, te_y)


if __name__ == '__main__':
    main()
