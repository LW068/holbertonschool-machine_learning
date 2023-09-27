#!/usr/bin/env python3
import numpy as np


def kmeans(X, k, iterations=1000):
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    n, d = X.shape
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    # initializing the centroids
    C = np.random.uniform(min_vals, max_vals, (k, d))

    for i in range(iterations):
        # assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # update centroids
        new_C = np.array([X[clss == j].mean(axis=0) for j in range(k)])

        # check for empty clusters and reinitialize
        for j in range(k):
            if np.isnan(new_C[j]).all():
                new_C[j] = np.random.uniform(min_vals, max_vals)

        # break if no change in th ecentroids
        if np.all(C == new_C):
            return C, clss

        C = new_C

    return C, clss
