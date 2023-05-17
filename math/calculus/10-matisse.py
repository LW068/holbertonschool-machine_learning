#!/usr/bin/env python3
"""
This module contains a function that calculates the derivative of a polynomial.
"""


def poly_derivative(poly):
    """
    This function calculates the derivative of a polynomial.

    Parameters:
    poly (list): A list of coefficients representing a polynomial.

    Returns:
    list: The coefficients of the derivative of the polynomial, or None if poly is not valid.
    """
    if not isinstance(poly, list) or len(poly) == 0 or not all(isinstance(i, (int, float)) for i in poly):
        return None
    if len(poly) == 1:
        return [0]
    return [i*poly[i] for i in range(1, len(poly))]

if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_derivative(poly))
