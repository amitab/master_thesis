import tensorflow as tf
import numpy as np

import math


# Memory Fusion from Multiple Neural Networks
def conv2d_transformer(w):
    dims = w.shape
    assert len(dims) == 4

    return w.reshape(dims[0], dims[1] * dims[2] * dims[3])


layer_weights_transformer_mapping = {'conv2d': conv2d_transformer}


def layer_weight_transformer(layer):
    if len(layer.weights[0].shape) == 4:
        return layer_weights_transformer_mapping['conv2d'](
            layer.weights[0].numpy())
    assert len(layer.weights[0].shape) <= 2
    return layer.weights[0].numpy()


def layer_bytes(layer):
    return (layer.count_params() * 8) / 1e+6