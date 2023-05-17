#!/usr/bin/env python3
"""
This module contains a function that calculates the sum of squares of numbers
from 1 to n (inclusive).
"""


def summation_i_squared(n):
    """
    This function calculates the sum of squares of
    numbers from 1 to n (inclusive).

    Parameters:
    n (int): The end of the range.

    Returns:
    int: The sum of squares, or None if n is not a valid number.
    """
    if not isinstance(n, int) or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6


if __name__ == "__main__":
    n = 5
    print(summation_i_squared(n))
