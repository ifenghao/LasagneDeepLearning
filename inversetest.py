# coding:utf-8
import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.layers import InputLayer, Conv2DLayer, DenseLayer, NonlinearityLayer, Deconv2DLayer
from lasagne.layers import InverseLayer, ReshapeLayer
import matplotlib.pylab as plt
import myUtils.load as load
from theano.tensor.nlinalg import pinv
from theano.tensor.slinalg import solve
import myUtils

x = T.tensor4()
l_in = InputLayer((1, 3, 32, 32), input_var=x)
conv = Conv2DLayer(l_in, num_filters=5, filter_size=3, nonlinearity=lasagne.nonlinearities.rectify)
removebias = myUtils.layers.RemoveBiasLayer(conv, conv.b)
deconv1 = Deconv2DLayer(removebias, conv.input_shape[1],
                        conv.filter_size, stride=conv.stride, crop=conv.pad,
                        W=conv.W, b=None, flip_filters=not conv.flip_filters)
deconv2 = InverseLayer(conv, conv)
deconv2 = NonlinearityLayer(deconv2)

out = lasagne.layers.get_output(conv)
out1 = lasagne.layers.get_output(deconv1)
out2 = lasagne.layers.get_output(deconv2)
func1 = theano.function([x], [out, out1, out2])

x_i = T.tensor4()
conv_in = InputLayer((1, 5, 30, 30), input_var=x_i)
removebias = myUtils.layers.RemoveBiasLayer(conv_in, conv.b)
deconv_i = Deconv2DLayer(removebias, conv.input_shape[1],
                         conv.filter_size, stride=conv.stride, crop=conv.pad,
                         W=conv.W, b=None, flip_filters=False)
out_i = lasagne.layers.get_output(deconv_i)
func_i = theano.function([x_i], out_i)

tr_X, te_X, tr_y, te_y = load.cifar()
img = tr_X[0, :, :, :].reshape(1, 3, 32, 32)
convout, deconvout1, deconvout2 = func1(img)
deconvout = func_i(convout)
print deconvout.shape, deconvout1.shape, deconvout2.shape
deconvout3 = func_i(convout)
deconvout1 = 255. * deconvout1 / deconvout1.max()
deconvout2 = 255. * deconvout2 / deconvout2.max()
deconvout3 = 255. * deconvout3 / deconvout3.max()
deconvout1 = deconvout1[0, :, :, :].transpose((1, 2, 0))
deconvout2 = deconvout2[0, :, :, :].transpose((1, 2, 0))
deconvout3 = deconvout3[0, :, :, :].transpose((1, 2, 0))

plt.figure()
img = img[0, :, :, :].transpose((1, 2, 0))
plt.imshow(img)
plt.figure()
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(convout[0, i, :, :])

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(np.squeeze(deconvout1))
plt.subplot(1, 3, 2)
plt.imshow(np.squeeze(deconvout2))
plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(deconvout3))

plt.show()
