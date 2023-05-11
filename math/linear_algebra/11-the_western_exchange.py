#!/usr/bin/env python3
"""
This module contains the function np_transpose that transposes
a numpy.ndarray
"""


import numpy as np


def np_transpose(matrix):
    """
    Function that transposes a numpy.ndarray
    Args:
        matrix (numpy.ndarray): The input array.
    Returns:
        numpy.ndarray: The transposed array.
    """
    return np.transpose(matrix)
