#!/usr/bin/env python3
"""
Module for forward propagation over a convolutional layer of a neural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network

    Parameters:
    - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
      containing the output of the previous layer
    - W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
      containing the kernels for the convolution
    - b is a numpy.ndarray of shape (1, 1, 1, c_new)
      containing the biases applied to the convolution
    - activation is an activation function applied to the convolution
    - padding is a string that is either same or valid
    - stride is a tuple of (sh, sw)

    Returns:
    The output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    ph, pw = 0, 0
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1

    h_new = (h_prev - kh + 2 * ph) // sh + 1
    w_new = (w_prev - kw + 2 * pw) // sw + 1

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')
    A_new = np.zeros((m, h_new, w_new, c_new))

    for i in range(h_new):
        for j in range(w_new):
            for c in range(c_new):
                A_slice = A_prev_pad[:, i*sh: i*sh+kh, j*sw: j*sw+kw, :]
                weights = W[:, :, :, c]
                biases = b[0, 0, 0, c]
                A_new[:, i, j, c] = activation(np.sum(A_slice * weights,
                                                      axis=(1, 2, 3)) + biases)
    return np.around(A, decimals=x)
