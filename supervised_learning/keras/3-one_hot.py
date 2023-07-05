#!/usr/bin/env python3
"""one_hot"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix.

    Args:
        labels: numpy.ndarray of shape (m,) with input labels
        classes: the maximum number of classes found in labels
    Returns:
        A one-hot encoding of labels. The last dimension of the one-hot
        matrix must be the number of classes.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
