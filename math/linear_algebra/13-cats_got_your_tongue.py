#!/usr/bin/env python3
"""
This module contains the function np_cat that concatenates two matrices along a specific axis.
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Function to concatenate two numpy arrays along a specific axis.
    Args:
        mat1 (numpy.ndarray): The first input array.
        mat2 (numpy.ndarray): The second input array.
        axis (int): The axis along which the concatenation should happen.
    Returns:
        numpy.ndarray: The resulting array after concatenation.
    """
    return np.concatenate((mat1, mat2), axis)
