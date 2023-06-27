#!/usr/bin/env python3
"""
Module for 7-RMSProp.py
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Args:
        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        s: previous second moment of var

    Returns:
        The updated variable and the new moment, respectively
    """
    Sd = beta2 * s + (1 - beta2) * grad ** 2
    var_updated = var - alpha * grad / (np.sqrt(Sd) + epsilon)
    return var_updated, Sd
