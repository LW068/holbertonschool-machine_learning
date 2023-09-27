#!/usr/bin/env python3
"""PCA on a dataset"""
import numpy as np

def pca(X, ndim):
    """Perform PCA on a dataset to reduce it to ndim dimensions."""
    # cnter the dataset
    X_centered = X - np.mean(X, axis=0) # is centers the dataset by
    # subtracting th mean of each feature.

    # calculate the covarance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # sort the eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # select the first ndim eigenvectors
    W = sorted_eigenvectors[:, :ndim]

    # transform the dataset
    T = np.dot(X_centered, W)

    return T
