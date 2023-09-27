#!/usr/bin/env python3
"""fucntion that calculates the total intra-cluster variance"""
import numpy as np


def variance(X, C):
    """fucntion that calculates the total intra-cluster variance"""
    # input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    try:
        # calculate the distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

        # find the closest centroid f0r each point
        clss = np.argmin(distances, axis=1)

        # calculate the variance
        min_distances = np.min(distances, axis=1)
        var = np.sum(min_distances ** 2)

        return var
    except Exception as e:
        return None
