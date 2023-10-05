#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def regular(P):
    """
    Function to determine the steady state probabilities
    of a regular markov chain
    """
    n = P.shape[0]
    for k in range(1, n + 1):
        Pt = np.linalg.matrix_power(P, k)
        if (Pt > 0).all():
            break
    else:
        return None  # return None if P is not regular

    try:
        # calculate eigenvalues and eigenvectors
        w, v = np.linalg.eig(np.transpose(P))
        # get the index of eigenvalue 1
        idx = np.argmin(np.abs(w - 1))
        # normalize the corresponding eigenvector to get the steady state
        steady = np.real(v[:, idx] / np.sum(v[:, idx]))
        return steady[np.newaxis, :]  # shape it as (1, n)
    except Exception:
        return None
