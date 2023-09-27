#!/usr/bin/env python3
import numpy as np
"""Performs k-means on a dataset"""


def kmeans(X, k, iterations=1000):
    """Performs k-means on a dataset"""
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
        new_C = np.array([np.mean(X[clss == j], axis=0) for j in range(k)])

        # check f0r empty clusters and reinitialize
        empty_clusters = np.isnan(new_C).any(axis=1)
        if np.any(empty_clusters):
            new_C[empty_clusters] = np.random.uniform(min_vals, max_vals, (empty_clusters.sum(), d))


        # break if no change in th ecentroids
        if np.all(C == new_C):
            return C, clss

        C = new_C

    return C, clss


# test the function
if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    X = np.concatenate((a, b, c, d, e), axis=0)
    np.random.shuffle(X)
    C, clss = kmeans(X, 5)
    print("Centroids:", C)
