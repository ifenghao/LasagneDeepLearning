# coding:utf-8
import lasagne


class RemoveBiasLayer(lasagne.layers.Layer):
    def __init__(self, incoming, bias, **kwargs):
        super(RemoveBiasLayer, self).__init__(incoming, **kwargs)
        if bias.ndim == 1:
            bias = bias.dimshuffle('x', 0, 'x', 'x')
        self.remove_bias = bias

    def get_output_for(self, input, **kwargs):
        return input - self.remove_bias
