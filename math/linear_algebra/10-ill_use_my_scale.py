#!/usr/bin/env python3

"""
This module contains a function that calculates
the shape of a numpy.ndarray.
"""


import numpy as np

def np_shape(matrix):
    """
    Calculates the shape of a numpy.ndarray.

    Args:
        matrix (numpy.ndarray): A numpy array.

    Returns:
        A tuple of integers representing the
        shape of the input matrix.
    """
    return matrix.shape
