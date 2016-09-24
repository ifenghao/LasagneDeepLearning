# coding:utf-8
__author__ = 'zfh'
import cPickle
import os
from copy import copy
from random import uniform

import numpy as np
import theano
from theano import shared, function, In, Out
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()

'''
theano相关的操作
'''


def makeFunc(inList, outList, updates):
    inputs = []
    for i in inList:
        inputs.append(In(i, borrow=True, allow_downcast=True))
    outputs = []
    for o in outList:
        outputs.append(Out(o, borrow=True))
    return function(
        inputs=inputs,
        outputs=outputs,  # 减少返回参数节省时间
        updates=updates,
        allow_input_downcast=True
    )


'''
网络结构中需要计算的参量
'''


# 使用GPU时误差的计算，输入都必须是TensorType
def eqs(yProb, y):
    assert yProb.ndim == 2
    assert y.ndim == 1
    yProb = T.argmax(yProb, axis=1)
    return T.sum(T.eq(yProb, y))  # 返回相等元素个数


'''
训练过程中的方法
'''


def miniBatchGen(X, y, batchSize, shuffle=False):
    assert len(X) == len(y)
    size = len(X)
    startRange = range(0, size - batchSize + 1, batchSize)  # 作为索引的取值，end要加1
    endRange = range(batchSize, size + 1, batchSize)
    if size % batchSize != 0:
        startRange.append(size - size % batchSize)
        endRange.append(size)
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    for start, end in zip(startRange, endRange):
        excerpt = indices[start:end]
        yield X[excerpt], y[excerpt]


def earlyStopGen(start=5, period=3, threshold=10, tol=2):
    '''
    早停止生成器，生成器可以保存之前传入的参数，从而在不断send入参数的过程中判断是否早停止训练
    :param start: 开始检测早停止的epoch，即至少完成多少epoch后才可以早停止
    :param period: 监视早停止标志的周期，每period个epoch计算一次stopMetric
    :param threshold: stopMetric的阈值，超过此阈值则早停止标志计数一次
    :param tol: 早停止标志计数的容忍限度，当计数超过此限度则立即执行早停止
    :return: 是否执行早停止
    '''
    trCostPeriod = []
    vaCostPeriod = []
    vaCostOpt = np.inf
    epoch = 0
    stopSign = False
    stopCount = 0
    while True:
        newCosts = (yield stopSign)  # 返回是否早停止
        epoch += 1
        if stopSign:  # 返回一个早停止标志后，重新检测
            stopSign = False
            stopCount = 0
        if epoch > start and newCosts is not None:  # send进来的元组在newCosts中
            trCost, vaCost = newCosts
            trCostPeriod.append(trCost)
            vaCostPeriod.append(vaCost)
            if vaCost < vaCostOpt:
                vaCostOpt = vaCost
            if len(trCostPeriod) >= period:
                P = np.mean(trCostPeriod) / np.min(trCostPeriod) - 1
                GL = np.mean(vaCostPeriod) / vaCostOpt - 1
                stopMetric = GL / P  # 停止的度量策略
                if stopMetric >= threshold:
                    stopCount += 1
                    if stopCount >= tol:
                        stopSign = True
                trCostPeriod = []  # 清空列表以继续判断下个周期
                vaCostPeriod = []


def randomSearch(nIter):
    '''
    随机生成超参数组合搜索最优结果
    :param nIter: 迭代次数，即超参数组合个数
    :return: 超参数组合的二维列表
    '''
    lr = [uniform(1, 20) * 1e-3 for _ in range(nIter)]
    C = [uniform(1, 500) * 1e-1 for _ in range(nIter)]
    # pDropConv = [uniform(0., 0.3) for _ in range(nIter)]
    # pDropHidden = [uniform(0., 0.5) for _ in range(nIter)]
    return zip(lr, C)


# 保存网络参数
def dumpModel(convNet):
    modelPickleFile = os.path.join(os.getcwd(), 'convNet' + '.pkl')
    with open(modelPickleFile, 'w') as file:
        cPickle.dump(convNet.params, file)
