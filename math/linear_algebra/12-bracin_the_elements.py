#!/usr/bin/env python3
"""
This module contains the function np_elementwise that performs 
element-wise addition, subtraction, multiplication, and division
without using any loops or conditional statements.
"""


def np_elementwise(mat1, mat2):
    """
    Function that performs element-wise operations on two numpy arrays.
    Args:
        mat1 (numpy.ndarray): The first input array.
        mat2 (numpy.ndarray): The second input array.
    Returns:
        tuple: The element-wise sum, difference, product, and quotient 
               of mat1 and mat2, respectively.
    """
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
