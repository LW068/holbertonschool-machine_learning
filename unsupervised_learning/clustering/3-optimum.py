#!/usr/bin/env python3
"""fucntion that tests for the optimum number of clusters"""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """fucntion that tests for the optimum number of clusters"""
    if kmax is None:
          kmax = len(X)

    # imports the required functions
    kmeans = __import__('1-kmeans').kmeans  # still wrong but we'll see...
    variance = __import__('2-variance').variance

    if kmax < kmin or kmax > len(X) or kmin < 1:
        return None, None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))

        var = variance(X, C)
        if k == kmin:
            smallest_var = var

        d_vars.append(var - smallest_var)

    return results, d_vars
