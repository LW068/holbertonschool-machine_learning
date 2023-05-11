#!/usr/bin/env python3

"""
This module contains a function that concatenates two matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1 (list): A 2D matrix containing ints or floats.
        mat2 (list): A 2D matrix containing ints or floats.
        axis (int, optional): The axis along which to concatenate the matrices.
            Defaults to 0.

    Returns:
        A new 2D matrix that is the concatenation of mat1 and mat2 along the specified axis,
        or None if the matrices cannot be concatenated.
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
