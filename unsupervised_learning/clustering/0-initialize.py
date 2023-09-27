#!/usr/bin/env python3
"""initialize K-means centroids"""
import numpy as np


def initialize(X, k):
    """initializes cluster centroids for K-means."""

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
