#!/usr/bin/env python3
"""
Module for 8-RMSProp.py
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Creates the training operation for a neural network in tensorflow using
    the RMSProp optimization algorithm.

    Args:
        loss: loss of the network
        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero

    Returns:
        The RMSProp optimization operation
    """
    return (tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon).
            minimize(loss))
