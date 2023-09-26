#!/usr/bin/env python3
"""PCA on a dataset"""
import numpy as np

def pca(X, ndim):
    """Perform PCA on a dataset to reduce it to ndim dimensions."""
    # cnter the dataset
    X_centered = X - np.mean(X, axis=0)

    # calculate the covarance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
