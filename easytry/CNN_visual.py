import theano.tensor as T
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from lasagne import regularization

import myUtils
import cPickle


def _build(X):
    layer = layers.InputLayer(shape=(None, 1, 28, 28), input_var=X)
    layer = layers.Conv2DLayer(layer, num_filters=32, filter_size=(5, 5), stride=(1, 1), pad='same',
                               untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                               nonlinearity=nonlinearities.rectify)
    visual1 = layers.get_output(layer)
    layer = layers.MaxPool2DLayer(layer, pool_size=(2, 2), stride=None, pad=(0, 0), ignore_border=False)
    layer = layers.Conv2DLayer(layer, num_filters=32, filter_size=(5, 5), stride=(1, 1), pad='same',
                               untie_biases=False, W=init.GlorotUniform(), b=init.Constant(0.),
                               nonlinearity=nonlinearities.rectify)
    visual2 = layers.get_output(layer)
    layer = layers.MaxPool2DLayer(layer, pool_size=(2, 2), stride=None, pad=(0, 0), ignore_border=False)
    layer = layers.flatten(layer, outdim=2)
    layer = layers.DropoutLayer(layer, p=0.5)
    layer = layers.DenseLayer(layer, num_units=256,
                              W=init.GlorotUniform(), b=init.Constant(0.),
                              nonlinearity=nonlinearities.rectify)
    layer = layers.DropoutLayer(layer, p=0.5)
    layer = layers.DenseLayer(layer, num_units=10,
                              W=init.GlorotUniform(), b=init.Constant(0.),
                              nonlinearity=nonlinearities.softmax)
    return layer, visual1, visual2


X = T.tensor4('X')
network, visual1, visual2 = _build(X)
valueList = cPickle.load(open('mnist_cnn.pkl', 'r'))
layers.set_all_param_values(network, valueList)
visualfn1 = myUtils.basic.makeFunc([X, ], [visual1, ], None)
visualfn2 = myUtils.basic.makeFunc([X, ], [visual2, ], None)

tr_X, te_X, tr_y, te_y = myUtils.load.mnist()
tr_X = myUtils.pre.norm4d_per_sample(tr_X)
te_X = myUtils.pre.norm4d_per_sample(te_X)
output1 = visualfn1(tr_X[:3])
myUtils.visual.show_map(output1)
output2 = visualfn2(tr_X[:3])
myUtils.visual.show_map(output2)
