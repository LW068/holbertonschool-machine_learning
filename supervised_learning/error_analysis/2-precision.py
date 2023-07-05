#!/usr/bin/env python3
"""Module for calculating precision."""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
    confusion : np.ndarray of shape (classes, classes) where row indices
                represent the correct labels and column...
                ...indices represent the predicted labels.
    classes : int, the number of classes

    Returns:
    np.ndarray of shape (classes,) containing the precision of each class.
    """
    tp = np.diagonal(confusion)
    fp = np.sum(confusion, axis=0) - tp
    precision = tp / (tp + fp)

    return precision
