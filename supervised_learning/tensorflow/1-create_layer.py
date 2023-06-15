#!/usr/bin/env python3
"""Create Layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            name='layer')
    return layer(prev)
