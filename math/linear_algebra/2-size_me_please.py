#!/usr/bin/env python3

"""
This module contains a function that calculates the shape of a matrix.
"""

def matrix_shape(matrix):
    """
    Returns the shape of a matrix as a list of integers.

    Args:
        matrix (list): A matrix of any number of dimensions.

    Returns:
        A list of integers representing the shape of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
