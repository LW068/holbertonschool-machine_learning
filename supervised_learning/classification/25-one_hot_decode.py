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
    - one_hot (numpy.ndarray): A one_hot encoded...
    ...matrix with shape (classes, m).
        classes represents the maximum number of classes.
        m represents the number of examples.

    Returns:
    - numpy.ndarray: A vector of shape (m,) containing the numeric labels
      for each example, or None on failure.
    """

    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    # Determine the number of examples
    num_of_examples = one_hot.shape[1]

    # Creating an empty numpy array of zeros with the right shape
    numeric_labels = np.zeros((num_of_examples,), dtype=int)

    for i in range(num_of_examples):
        # Find the index of the maximum value in each column
        max_value_index = np.argmax(one_hot[:, i])

        # Update the corresponding position in the labels vector with the index
        numeric_labels[i] = max_value_index

    return numeric_labels
