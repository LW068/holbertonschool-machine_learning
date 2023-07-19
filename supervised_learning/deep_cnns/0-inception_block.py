#!/usr/bin/env python3
"""Inception Block

This module contains a function that builds an inception block using Keras
as described in "Going Deeper with Convolutions (2014)".
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ Builds an inception block

    Args:
        A_prev (keras layer): output from the previous layer
        filters (list or tuple): containing F1, F3R, F3, F5R, F5, FPP
        F1: number of filters in the 1x1 convolution
        F3R: number of filters in the 1x1 before the 3x3 convolution
        F3: number of filters in the 3x3 convolution
        F5R: number of filters in the 1x1 before the 5x5 convolution
        F5: number of filters in the 5x5 convolution
        FPP: number of filters in the 1x1 after the max pooling

    Returns:
        keras layer: the concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(F1, 1, activation='relu')(A_prev)

    conv3R = K.layers.Conv2D(F3R, 1, activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(F3, 3, padding='same', activation='relu')(conv3R)

    conv5R = K.layers.Conv2D(F5R, 1, activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(F5, 5, padding='same', activation='relu')(conv5R)

    pool = K.layers.MaxPooling2D(pool_size=3, strides=1,
                                 padding='same')(A_prev)
    convPool = K.layers.Conv2D(FPP, 1, activation='relu')(pool)

    output = K.layers.concatenate([conv1, conv3, conv5, convPool])

    return output
