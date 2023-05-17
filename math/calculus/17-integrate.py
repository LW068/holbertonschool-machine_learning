#!/usr/bin/env python3
"""
This module contains a function that calculates the integral of a polynomial.
"""


def poly_integral(poly, C=0):
    """
    This function calculates the integral of a polynomial.

    Parameters:
    poly (list): A list of coefficients representing a polynomial.
    C (int): The integration constant.

    Returns:
    list: The coefficients of the integral of the polynomial,
          or None if poly or C is not valid.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(i, (int, float)) for i in poly):
        return None
    if not isinstance(C, (int, float)):
        return None
    integral = [C]
    for i in range(len(poly)):
        if poly[i] % (i+1) == 0:
            integral.append(poly[i] // (i+1))
        else:
            integral.append(poly[i] / (i+1))
    return integral


if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_integral(poly))
