#!/usr/bin/env python3
"""Module to calculate the correlation matrix from a given covariance matrix."""

import numpy as np


def correlation(C):
    """
    Calculate the correlation matrix from a given covariance matrix.

    Parameters:
    - C (numpy.ndarray of shape (d, d)): The covariance matrix.

    Returns:
    - numpy.ndarray of shape (d, d): The correlation matrix.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    d, d_check = C.shape
    if d != d_check:
        raise ValueError("C must be a 2D square matrix")

    # Calculate the standard deviations for each dimension
    std_devs = np.sqrt(np.diag(C))

    # Create a matrix where each element (i, j) is std_devs[i] * std_devs[j]
    outer_std_devs = np.outer(std_devs, std_devs)

    # Calculate the correlation matrix
    corr_matrix = C / outer_std_devs

    return corr_matrix


if __name__ == '__main__':
    import numpy as np

    C = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    Co = correlation(C)
    print(C)
    print(Co)
