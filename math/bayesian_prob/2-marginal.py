#!/usr/bin/env python3
"""Module for calculating marginal probability in Bayesian data."""

import numpy as np
from scipy.special import comb


def marginal(x, n, P, Pr):
    """
    Calculate the marginal probability of obtaining this data.

    Parameters:
    - x (int): patients with severe side effects
    - n (int): total number of patients
    - P (1D numpy.ndarray): hypothetical probabilities
    - Pr (1D numpy.ndarray): prior beliefs of P

    Returns:
    - float: the marginal probability of obtaining x and n
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater "
            "than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same "
            "shape as P"
        )
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    binom_coeff = comb(n, x)
    likelihood_values = binom_coeff * (P ** x) * ((1 - P) ** (n - x))
    intersection_values = likelihood_values * Pr

    return np.sum(intersection_values)


if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11  # Uniform prior
    print(marginal(26, 130, P, Pr))
