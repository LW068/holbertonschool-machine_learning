#!/usr/bin/env python3
"""
Module for task 12-learning_rate_decay
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Creates a learning rate decay operation...
    ...in tensorflow using inverse time decay
    """
    alpha = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True
    )

    return alpha
