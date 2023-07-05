#!/usr/bin/env python3
"""Module for calculating specificity."""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
    confusion : np.ndarray of shape (classes, classes) where row indices
                represent the correct labels and column...
                ...indices represent the predicted labels.
    classes : int, the number of classes

    Returns:
    np.ndarray of shape (classes,) containing the specificity of each class.
    """
    tp = np.diagonal(confusion)
    fn = np.sum(confusion, axis=1) - tp
    fp = np.sum(confusion, axis=0) - tp
    tn = np.sum(confusion) - (tp + fp + fn)
    specificity = tn / (tn + fp)

    return specificity
