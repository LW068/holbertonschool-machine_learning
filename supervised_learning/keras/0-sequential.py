#!/usr/bin/env python3
"""build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number...
    ...of nodes in each layer of the network
    activations is a list containing the activation...
    ...functions used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout

    Returns: the keras model
    """
    model = K.models.Sequential()

    reg = K.regularizers.l2(lambtha)

    # Add first layer
    model.add(K.layers.Dense(units=layers[0], activation=activations[0],
                             kernel_regularizer=reg, input_shape=(nx,)))
    if len(layers) > 1:  # Only add dropout if there are more than one layers
        model.add(K.layers.Dropout(1 - keep_prob))

    # Add subsequent layers
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(units=layers[i], activation=activations[i],
                                 kernel_regularizer=reg))
        if i != len(layers) - 1:  # no dropout after last layer
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
