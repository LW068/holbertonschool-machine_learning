#!/usr/bin/env python3

"""
This module contains a function that returns the transpose of a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.

    Args:
        matrix (list): A 2D matrix.

    Returns:
        A new matrix that is the transpose of the input matrix.
    """
    transpose = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return transpose
