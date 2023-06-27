#!/usr/bin/env python3
"""
Module for task 14-batch_norm
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    mean, var = tf.nn.moments(layer(prev), axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")
    BN = tf.nn.batch_normalization(layer(prev), mean, var, offset=beta,
                                   scale=gamma, variance_epsilon=1e-8)
    return activation(BN)
