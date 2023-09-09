#!/usr/bin/env python3
"""Module to represent a Multivariate Normal distribution."""

import numpy as np


class MultiNormal:
    """Class that represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Initialize MultiNormal instance.

        Parameters:
        - data (numpy.ndarray of shape (d, n)): The data set.

        Sets the public instance variables:
        - mean (numpy.ndarray of shape (d, 1)): The mean of the data set.
        - cov (numpy.ndarray of shape (d, d)):...
        ...The covariance matrix of the data set.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.dot(data - self.mean, (data - self.mean).T) / (n - 1)


if __name__ == '__main__':
    import numpy as np

    np.random.seed(0)
    data = np.random.multivariate_normal(
        [12, 30, 10], [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 10000).T
    mn = MultiNormal(data)
    print(mn.mean)
    print(mn.cov)
