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

    
    def pdf(self, x):
        """Calculate the PDF at a data point.

        Parameters:
            x (numpy.ndarray): shape (d, 1) data point for PDF calculation.

        Returns:
            float: value of the PDF.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        x_m = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)

        pdf = (1. / (np.sqrt((2 * np.pi) ** d * cov_det)) *
               np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2))

        return pdf.flatten()[0]


if __name__ == '__main__':
    import numpy as np

    np.random.seed(0)
    data = np.random.multivariate_normal(
        [12, 30, 10], 
        [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 
        10000
    ).T
    mn = MultiNormal(data)
    x = np.random.multivariate_normal(
        [12, 30, 10], 
        [[36, -30, 15], [-30, 100, -20], [15, -20, 25]], 
        1
    ).T
    print(x)
    print(mn.pdf(x))
