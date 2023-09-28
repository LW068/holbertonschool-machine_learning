#!/usr/bin/env python3
"""function that initializes variables f0r a Gaussian Mixture Model"""
import numpy as np


def initialize(X, k):
    """function that initializes variables f0r a Gaussian Mixture Model"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if type(k) is not int or k <= 0 or k >= X.shape[0]:
        return None, None, None

    kmeans = __import__('1-kmeans').kmeans

    # initialize the priors so f0r each cluster evenly
    pi = np.full((k,), 1 / k)

    # initialize the centriod means f0r each cluster using k-means
    m, _ = kmeans(X, k)

    # initialize the covariance matrices f0r each cluster as identity matricse
    d = X.shape[1]
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
