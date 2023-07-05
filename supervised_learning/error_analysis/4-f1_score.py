#!/usr/bin/env python3
"""Module for calculating F1 score."""

import numpy as np


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
    confusion : np.ndarray of shape (classes, classes) where row indices
                represent the correct labels and column...
                ...indices represent the predicted labels.
    classes : int, the number of classes

    Returns:
    np.ndarray of shape (classes,) containing the F1 score of each class.
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision

    sen = sensitivity(confusion)
    pre = precision(confusion)

    f1 = 2 * ((pre * sen) / (pre + sen))

    return f1
