import tensorflow as tf

from tensorflow.contrib.layers import fully_connected as fc_layer

from .mlp import MLPModel
from .rnnd import RNNDModel


class MLPERNNDModel(RNNDModel):

    _build_encoder = MLPModel._build_encoder
