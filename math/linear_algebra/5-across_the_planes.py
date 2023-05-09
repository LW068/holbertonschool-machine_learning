#!/usr/bin/env python3

"""
This module contains a function that adds two matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise.

    Args:
        mat1 (list): A 2D matrix containing ints or floats.
        mat2 (list): A 2D matrix containing ints or floats.

    Returns:
        A new 2D matrix that is the element-wise sum of mat1 and mat2,
        or None if mat1 and mat2 are not the same shape.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
