#!/usr/bin/env python3
"""
25-one_hot_decode.py
Module that defines a function called one_hot_decode
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.

    Args:
    - one_hot (numpy.ndarray): A one-hot encoded matrix...
        ...with shape (classes, m).
      classes is the maximum number of classes.
      m is the number of examples.

    Returns:
    - numpy.ndarray: A vector with shape (m,) containing the numeric labels
      for each example, or None on failure.
    """
    # Check if the input is valid!
    if not isinstance(one_hot, np.ndarray):
        print("Invalid input in one_hot_decode:", one_hot)
        return None

    try:
        # Find the index of the maximum value in each column
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
