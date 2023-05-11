#!/usr/bin/env python3
"""
This module contains the function np_elementwise that performs element-wise
addition, subtraction, multiplication, and division of two numpy.ndarrays
"""


def np_elementwise(mat1, mat2):
    """
    Function that performs element-wise addition, subtraction,
    multiplication, and division of two numpy.ndarrays
    Args:
        mat1 (numpy.ndarray): The first input array.
        mat2 (numpy.ndarray): The second input array.
    Returns:
        tuple: A tuple containing the element-wise sum, difference,
        product, and quotient, respectively.
    """
    addition = np.add(mat1, mat2)
    subtraction = np.subtract(mat1, mat2)
    multiplication = np.multiply(mat1, mat2)
    division = np.divide(mat1, mat2)

    return (addition, subtraction, multiplication, division)
