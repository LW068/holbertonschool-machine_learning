#!/usr/bin/env python3
"""
Module for task 11-learning_rate_decay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy
    """
    alpha /= (1 + decay_rate * np.floor(global_step / decay_step))
    return alpha
