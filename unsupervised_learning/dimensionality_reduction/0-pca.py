#!/usr/bin/env python3
"""PCA on a dataset"""
import numpy as np

def pca(X, var=0.95):
  """PCA on a dataset"""
  # calculatign the covariance matrix
    cov_matrix = np.cov(X, rowvar=False)

# performign eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# sorting the  eigenvalues and the corresponding eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

return np.array([])
