#!/usr/bin/env python3

"""
This module contains a function that adds two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): A list of ints or floats.
        arr2 (list): A list of ints or floats.

    Returns:
        A new list that is the element-wise sum of arr1 and arr2,
        or None if arr1 and arr2 are not the same shape.
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
