#!/usr/bin/env python3
""" Module to hold the GaussianProcess Class """
import numpy as np


class GaussianProcess:
    """ Gaussian Process Class """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Constructor"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices"""
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1)
        sqdist = sqdist - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """Predicts the mean and standard deviation of
        points in a Gaussian process"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = mu_s.reshape((mu_s.shape[0],))

        sigma_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma_s = np.diag(sigma_s)

        return mu_s, sigma_s
