#!/usr/bin/env python3
"""droput_create_layer"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of a neural network using dropout

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function to be used on the layer
        keep_prob: probability that a node will be kept

    Returns:
        the output of the new layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(1 - keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init)
    return dropout(layer(prev))
