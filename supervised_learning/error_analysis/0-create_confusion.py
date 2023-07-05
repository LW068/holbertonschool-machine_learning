#!/usr/bin/env python3
"""Module for creating confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Create a confusion matrix.

    Args:
    labels : np.ndarray of shape (m, classes) with correct labels.
    logits : np.ndarray of shape (m, classes) with predicted labels.

    Returns:
    np.ndarray of shape (classes, classes) with row indices representing
    correct labels and column indices representing predicted labels.
    """
    m, classes = labels.shape
    confusion_matrix = np.zeros((classes, classes))

    for i in range(m):
        true_label = np.argmax(labels[i])
        pred_label = np.argmax(logits[i])
        confusion_matrix[true_label][pred_label] += 1

    return confusion_matrix
