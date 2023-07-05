#!/usr/bin/env python3
"""Module for calculating sensitivity."""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
    confusion : np.ndarray of shape (classes, classes) where row indices
                represent the correct labels and column...
                ...indices represent the predicted labels.
    classes : int, the number of classes

    Returns:
    np.ndarray of shape (classes,) containing the sensitivity of each class.
    """
    tp = np.diagonal(confusion)
    fn = np.sum(confusion, axis=1) - tp
    sensitivity = tp / (tp + fn)

    return sensitivity
