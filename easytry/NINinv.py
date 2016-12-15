# coding:utf-8

import time
import cPickle
import os
import theano
import theano.tensor as T
from lasagne import layers
from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization

import myUtils
import matplotlib.pylab as plt


def model(x):
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32), input_var=x)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp1'] = ConvLayer(
        net['conv1'], num_filters=160, filter_size=1, flip_filters=False)
    net['cccp2'] = ConvLayer(
        net['cccp1'], num_filters=96, filter_size=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['cccp2'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['drop3'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv2'] = ConvLayer(net['drop3'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp3'] = ConvLayer(
        net['conv2'], num_filters=192, filter_size=1, flip_filters=False)
    net['cccp4'] = ConvLayer(
        net['cccp3'], num_filters=192, filter_size=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['cccp4'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['drop6'] = DropoutLayer(net['pool2'], p=0.5)
    net['conv3'] = ConvLayer(net['drop6'],
                             num_filters=192,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['cccp5'] = ConvLayer(
        net['conv3'], num_filters=192, filter_size=1, flip_filters=False)
    net['cccp6'] = ConvLayer(
        net['cccp5'], num_filters=10, filter_size=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['cccp6'],
                             pool_size=8,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['out'] = NonlinearityLayer(FlattenLayer(net['pool3']), nonlinearity=nonlinearities.softmax)
    return net


def modelinv(y, fnet):
    net = {}
    net['input'] = layers.InputLayer(shape=(None, 10), input_var=y)
    net['input'] = layers.ReshapeLayer(net['input'], (-1, 10, 1, 1))
    net['ipool3'] = layers.Upscale2DLayer(net['input'], 8)
    biasremove = myUtils.layers.RemoveBiasLayer(net['ipool3'], fnet['cccp6'].b)
    net['icccp6'] = layers.Deconv2DLayer(biasremove,
                                         num_filters=fnet['cccp6'].input_shape[1],
                                         filter_size=fnet['cccp6'].filter_size,
                                         stride=fnet['cccp6'].stride,
                                         crop=fnet['cccp6'].pad,
                                         W=fnet['cccp6'].W, b=None,
                                         flip_filters=not fnet['cccp6'].flip_filters)
    biasremove = myUtils.layers.RemoveBiasLayer(net['icccp6'], fnet['cccp5'].b)
    net['icccp5'] = layers.Deconv2DLayer(biasremove,
                                         num_filters=fnet['cccp5'].input_shape[1],
                                         filter_size=fnet['cccp5'].filter_size,
                                         stride=fnet['cccp5'].stride,
                                         crop=fnet['cccp5'].pad,
                                         W=fnet['cccp5'].W, b=None,
                                         flip_filters=not fnet['cccp5'].flip_filters)
    biasremove = myUtils.layers.RemoveBiasLayer(net['icccp5'], fnet['conv3'].b)
    net['iconv3'] = layers.Deconv2DLayer(biasremove,
                                         num_filters=fnet['conv3'].input_shape[1],
                                         filter_size=fnet['conv3'].filter_size,
                                         stride=fnet['conv3'].stride,
                                         crop=fnet['conv3'].pad,
                                         W=fnet['conv3'].W, b=None,
                                         flip_filters=not fnet['conv3'].flip_filters)
    net['ipool2'] = layers.Upscale2DLayer(net['iconv3'], 2)
    biasremove = myUtils.layers.RemoveBiasLayer(net['ipool2'], fnet['cccp4'].b)
    net['icccp4'] = layers.Deconv2DLayer(biasremove,
                                         num_filters=fnet['cccp4'].input_shape[1],
                                         filter_size=fnet['cccp4'].filter_size,
                                         stride=fnet['cccp4'].stride,
                                         crop=fnet['cccp4'].pad,
                                         W=fnet['cccp4'].W, b=None,
                                         flip_filters=not fnet['cccp4'].flip_filters)
    biasremove = myUtils.layers.RemoveBiasLayer(net['icccp4'], fnet['cccp3'].b)
    net['icccp3'] = layers.Deconv2DLayer(biasremove,
                                         num_filters=fnet['cccp3'].input_shape[1],
                                         filter_size=fnet['cccp3'].filter_size,
                                         stride=fnet['cccp3'].stride,
                                         crop=fnet['cccp3'].pad,
                                         W=fnet['cccp3'].W, b=None,
                                         flip_filters=not fnet['cccp3'].flip_filters)
    biasremove = myUtils.layers.RemoveBiasLayer(net['icccp3'], fnet['conv2'].b)
    net['iconv2'] = layers.Deconv2DLayer(biasremove,
                                         num_filters=fnet['conv2'].input_shape[1],
                                         filter_size=fnet['conv2'].filter_size,
                                         stride=fnet['conv2'].stride,
                                         crop=fnet['conv2'].pad,
                                         W=fnet['conv2'].W, b=None,
                                         flip_filters=not fnet['conv2'].flip_filters)
    net['ipool1'] = layers.Upscale2DLayer(net['iconv2'], 2)
    biasremove = myUtils.layers.RemoveBiasLayer(net['ipool1'], fnet['cccp2'].b)
    net['icccp2'] = layers.Deconv2DLayer(biasremove,
                                         num_filters=fnet['cccp2'].input_shape[1],
                                         filter_size=fnet['cccp2'].filter_size,
                                         stride=fnet['cccp2'].stride,
                                         crop=fnet['cccp2'].pad,
                                         W=fnet['cccp2'].W, b=None,
                                         flip_filters=not fnet['cccp2'].flip_filters)
    biasremove = myUtils.layers.RemoveBiasLayer(net['icccp2'], fnet['cccp1'].b)
    net['icccp1'] = layers.Deconv2DLayer(biasremove,
                                         num_filters=fnet['cccp1'].input_shape[1],
                                         filter_size=fnet['cccp1'].filter_size,
                                         stride=fnet['cccp1'].stride,
                                         crop=fnet['cccp1'].pad,
                                         W=fnet['cccp1'].W, b=None,
                                         flip_filters=not fnet['cccp1'].flip_filters)
    biasremove = myUtils.layers.RemoveBiasLayer(net['icccp1'], fnet['conv1'].b)
    net['iconv1'] = layers.Deconv2DLayer(biasremove,
                                         num_filters=fnet['conv1'].input_shape[1],
                                         filter_size=fnet['conv1'].filter_size,
                                         stride=fnet['conv1'].stride,
                                         crop=fnet['conv1'].pad,
                                         W=fnet['conv1'].W, b=None,
                                         flip_filters=not fnet['conv1'].flip_filters)
    net['out'] = net['iconv1']
    return net


class NINinv(object):
    def __init__(self):
        self.X = T.tensor4('X')
        self.y = T.matrix('y')

    def loadParams(self, filename):
        valueList = cPickle.load(open(filename, 'r'))
        layers.set_all_param_values(self.fnet['out'], valueList)

    def buildforward(self):
        self.fnet = model(self.X)
        fout = layers.get_output(self.fnet['out'])
        self.ffunction = theano.function([self.X], fout, allow_input_downcast=True)

    def buildbackward(self):
        self.bnet = modelinv(self.y, self.fnet)
        bout = layers.get_output(self.bnet['out'])
        self.bfunction = theano.function([self.y], bout, allow_input_downcast=True)

    def forward(self, inputX):
        return self.ffunction(inputX)

    def backward(self, inputy):
        return self.bfunction(inputy)


def main():
    tr_X, te_X, tr_y, te_y = myUtils.pre.cifarWhiten('cifar10')
    tr_y = myUtils.load.one_hot(tr_y, 10)
    te_y = myUtils.load.one_hot(te_y, 10)
    nininv = NINinv()
    nininv.buildforward()
    nininv.loadParams('/home/zhufenghao/pretrained_models/cifar10_nin.pkl')
    nininv.buildbackward()
    img = tr_X[0, :, :, :]
    x = img.reshape(1, 3, 32, 32)
    img = img.transpose((1, 2, 0))
    plt.figure()
    plt.imshow(img)
    fout = nininv.forward(x)
    print fout
    label = tr_y[0, :].reshape(1, 10)
    bout = nininv.backward(label)
    bout = bout.squeeze().transpose((1, 2, 0))
    plt.figure()
    plt.imshow(bout)
    plt.show()


if __name__ == '__main__':
    main()
