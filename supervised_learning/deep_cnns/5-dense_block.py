#!/usr/bin/env python3
""" 
Module to define the Dense Block 
based on Densely Connected Convolutional Networks
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected Convolutional Networks:

    - X is the output from the previous layer
    - nb_filters is an integer representing the number of filters in X
    - growth_rate is the growth rate for the dense block
    - layers is the number of layers in the dense block

    Returns:
    The concatenated output of each layer within the Dense Block 
    and the number of filters within the concatenated outputs, respectively
    """
    for i in range(layers):
        # Batch Normalization -> ReLU -> Bottleneck convolution
        X_copy = X
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same', kernel_initializer='he_normal')(X)

        # Batch Normalization -> ReLU -> 3x3 convolution
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(growth_rate, (3, 3), padding='same', kernel_initializer='he_normal')(X)

        # Concatenate the output of the 3x3 convolution with the inputs for the next layer
        X = K.layers.concatenate([X_copy, X])
        nb_filters += growth_rate

    return X, nb_filters
