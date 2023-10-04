#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def regular(P):
    """ 
    Function to determine the steady state probabilities of a regular markov chain 
    """
    try:
        # clculate eigenvalues and eigenvectors
        w, v = np.linalg.eig(np.transpose(P))
        # get the index of eigenvalue 1
        idx = np.argmin(np.abs(w - 1))
        # normalize the corresponding eigenvector to get the steady state
        steady = np.real(v[:, idx] / np.sum(v[:, idx]))
        return steady[np.newaxis, :]  # shape it as (1, n)
    except Exception:
        return None
