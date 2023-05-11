#!/usr/bin/env python3

"""
This module contains a function that performs
matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication.

    Args:
        mat1 (list): A 2D matrix containing ints or floats.
        mat2 (list): A 2D matrix containing ints or floats.

    Returns:
        A new 2D matrix that is the result of
        matrix multiplication of mat1 and mat2,
        or None if the matrices cannot be multiplied.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            value = sum(mat1[i][k] * mat2[k][j] for k in range(len(mat2)))
            row.append(value)
        result.append(row)

    return result
