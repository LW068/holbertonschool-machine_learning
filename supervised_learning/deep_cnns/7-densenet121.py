#!/usr/bin/env python3
"""
Module to define DenseNet-121
based on Densely Connected Convolutional Networks
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks:

    - growth_rate is the growth rate
    - compression is the compression factor

    All weights should use he normal initialization

    Returns: 
    The keras model
    """
    X_input = K.Input(shape=(224, 224, 3))

    # Initial Convolution
    X = K.layers.BatchNormalization(axis=3)(X_input)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(2 * growth_rate, (7, 7), strides=2,
                        padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.MaxPool2D((3, 3), strides=2, padding='same')(X)

    # Dense Block 1
    X, nb_filters = dense_block(X, 2 * growth_rate, growth_rate, 6)
    # Transition Layer 1
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 2
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    # Transition Layer 2
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 3
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    # Transition Layer 3
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # Dense Block 4
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Global Average Pooling
    X = K.layers.AveragePooling2D((7, 7))(X)

    # Fully Connected Layer
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer='he_normal')(X)

    model = K.models.Model(inputs=X_input, outputs=X, name='DenseNet')

    return model
