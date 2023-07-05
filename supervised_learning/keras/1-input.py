#!/usr/bin/env python3
"""build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number of...
    ...nodes in each layer of the network
    activations is a list containing the activation...
    ...functions used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout

    Returns: the keras model
    """
    inputs = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)
    outputs = inputs

    for i in range(len(layers)):
        outputs = K.layers.Dense(units=layers[i], activation=activations[i],
                                 kernel_regularizer=reg)(outputs)
        if i != len(layers) - 1:
            outputs = K.layers.Dropout(rate=(1 - keep_prob))(outputs)

    model = K.Model(inputs=inputs, outputs=outputs)

    return model
