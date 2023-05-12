#!/usr/bin/env python3
"""
This module contains the function np_matmul
that performs matrix multiplication.
"""


import numpy as np


def np_matmul(mat1, mat2):
    """
    Function to perform matrix multiplication.
    Args:
        mat1 (numpy.ndarray): The first input matrix.
        mat2 (numpy.ndarray): The second input matrix.
    Returns:
        numpy.ndarray: The resulting matrix after multiplication.
    """
    return np.matmul(mat1, mat2)
