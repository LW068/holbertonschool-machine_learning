#!/usr/bin/env python3
"""
24-one_hot_encode.py
Module that defines a function called one_hot_encode
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.

    Args:
    - Y (numpy.ndarray): A numeric label vector with shape (m,).
      m is the number of examples.
    - classes (int): The maximum number of classes found in Y.

    Returns:
    - numpy.ndarray: A one-hot encoding of Y with shape (classes, m),
      or None on failure.
    """
    # Check if the input types are valid
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        print("Invalid input in one_hot_encode:", Y, classes)
        return None

    try:
        # Create an identity matrix of size 'classes x classes'
        identity_matrix = np.eye(classes)

        # Select rows corresponding to the labels in Y
        selected_rows = identity_matrix[Y]

        # Transpose the resulting matrix to get the one-hot encoding
        one_hot = selected_rows.T

        return one_hot
    except Exception:
        return None
