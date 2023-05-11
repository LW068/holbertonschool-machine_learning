#!/usr/bin/env python3
"""
This module contains the function np_transpose that transposes
a numpy.ndarray without using any loops or conditional statements.
"""


def np_transpose(matrix):
    """
    Function that transposes a numpy.ndarray
    Args:
        matrix (numpy.ndarray): The input array.
    Returns:
        numpy.ndarray: The transposed matrix.
    """
    return matrix.T
