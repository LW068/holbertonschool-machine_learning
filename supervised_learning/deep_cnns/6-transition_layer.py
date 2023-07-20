#!/usr/bin/env python3
"""
Module to define the Transition Layer
based on Densely Connected Convolutional Networks
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    Densely Connected Convolutional Networks:

    - X is the output from the previous layer
    - nb_filters is an integer representing the number of filters in X
    - compression is the compression factor for the transition layer

    Returns: 
    The output of the transition layer and the number of
    filters within the output, respectively
    """
    # Batch Normalization -> ReLU
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Compression: Convolution with 1x1 kernel
    nb_filters_compressed = int(nb_filters * compression)
    X = K.layers.Conv2D(nb_filters_compressed, (1, 1),
                        padding='same', kernel_initializer='he_normal')(X)

    # Average Pooling 2x2
    X = K.layers.AveragePooling2D((2, 2), strides=2)(X)

    return X, nb_filters_compressed
