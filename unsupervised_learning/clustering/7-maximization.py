#!/usr/bin/env python3
"""fucntion that calculates the maximization
step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """ calculates the maximization step
    in the EM algorithm for a GMM """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, _ = g.shape

    if n != g.shape[1]:
        return None, None, None

    # updatse the priors
    pi = np.sum(g, axis=1) / n

    # updates the means
    m = np.dot(g, X) / np.sum(g, axis=1, keepdims=True)

    # updates the covariances
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.dot((g[i][:, np.newaxis] * diff.T), diff) / np.sum(g[i])

    return pi, m, S
