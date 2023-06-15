#!/usr/bin/env python3
"""Create Train Operation"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Create training operation for the network"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
