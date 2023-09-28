#!/usr/bin/env python3
"""fucntion that calculates the probability
density function of a Gaussian Distribution"""
import numpy as np


def pdf(X, m, S):
    """fucntion that calculates the probability
    density function of a Gaussian Distribution"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None

    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape

    if d != m.shape[0] or (d, d) != S.shape:
        return None

    # calculate the probability density function
    # should be simple enough:
    S_inv = np.linalg.inv(S)
    S_det = np.linalg.det(S)
    den = np.sqrt((2 * np.pi) ** d * S_det)
    fac = np.einsum('...k,kl,...l->...', X - m, S_inv, X - m)
    P = np.exp(-fac / 2) / den

    return np.maximum(P, 1e-300)
