# taken from Dan:
#
# https://github.com/dguest/flow-network/blob/8acc708469ab45ee221d46fbd036f61de20fc2a5/SumLayer.py#L4-L29
#
# Also exists in umami
#
# https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/5431cce00f0a120e31e6becf3b06592c3de4edd8/umami/train_tools/NN_tools.py#L1232-1260

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class Sum(Layer):
    """
    Simple sum layer

    The tricky bits are getting masking to work properly, but given
    that time distributed dense layers _should_ compute masking on
    their own.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        if mask is not None:
            x = x * K.cast(mask, K.dtype(x))[:,:,None]
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def compute_mask(self, inputs, mask):
        return None
