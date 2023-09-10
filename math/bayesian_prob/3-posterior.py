#!/usr/bin/env python3
"""Module for calculating posterior probability in Bayesian data."""

import numpy as np
from scipy.special import comb

def posterior(x, n, P, Pr):
    """
    Calculate the posterior probability given the data.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)) or np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in P and Pr must be in the range [0, 1]")
    if not np.isclose([np.sum(Pr)], [1])[0]:
        raise ValueError("Pr must sum to 1")

    binom_coeff = comb(n, x)
    likelihood_values = binom_coeff * (P ** x) * ((1 - P) ** (n - x))
    marginal_prob = np.sum(Pr * likelihood_values)
    posterior_prob = (likelihood_values * Pr) / marginal_prob

    return posterior_prob

if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(posterior(26, 130, P, Pr))
