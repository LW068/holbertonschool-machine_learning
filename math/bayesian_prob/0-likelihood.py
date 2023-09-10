#!/usr/bin/env python3
"""Bayesian Probability - Likelihood"""

import numpy as np
from scipy.special import comb


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    binom_coeff = comb(n, x)
    likelihood_values = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihood_values


if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    print(likelihood(26, 130, P))
