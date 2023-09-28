#!/usr/bin/env python3
"""function that calculates the expectation step for GMM"""
import numpy as np


pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """fucntion that calculates the expectation step for GMM"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None

    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None

    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    # normalize pi if it doesn't sum up to 1...
    if not np.isclose(np.sum(pi), 1):
        pi = pi / np.sum(pi)

    n, d = X.shape
    k = pi.shape[0]

    if (k, d) != m.shape or (k, d, d) != S.shape:
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        P = pdf(X, m[i], S[i])
        g[i] = P * pi[i]

    total = np.sum(g, axis=0, keepdims=True)

    g /= total

    log_likelihood = np.sum(np.log(total))

    return g, log_likelihood
