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
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = max((h_prev - 1) * sh + kh - h_prev, 0) // 2
        pw = max((w_prev - 1) * sw + kw - w_prev, 0) // 2
    else:
        ph = pw = 0

    h_new = (h_prev - kh + 2 * ph) // sh + 1
    w_new = (w_prev - kw + 2 * pw) // sw + 1

    A = np.zeros((m, h_new, w_new, c_new))

    A_prev_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    for i in range(m): 
        for h in range(h_new): 
            for w in range(w_new): 
                for c in range(c_new): 
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw
                    A_slice = A_prev_pad[i, h_start:h_end, w_start:w_end]
                    weights = W[:, :, :, c]
                    biases = b[0, 0, 0, c]
                    A[i, h, w, c] = activation(np.sum(A_slice * weights) + biases)

    return A