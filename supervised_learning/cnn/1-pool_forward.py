#!/usr/bin/env python3
"""
Module for forward propagation over a pooling layer of a neural network
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    Parameters:
    - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
      containing the output of the previous layer
    - kernel_shape is a tuple of (kh, kw) containing the size
      of the kernel for the pooling
    - stride is a tuple of (sh, sw) containing the strides for the pooling
    - mode is a string containing either max or avg,
      indicating whether to perform maximum or average pooling, respectively

    Returns:
    The output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    A = np.zeros((m, h_new, w_new, c_prev))

    for h in range(h_new):
        for w in range(w_new):
            h_start = h * sh
            h_end = h_start + kh
            w_start = w * sw
            w_end = w_start + kw
            A_slice = A_prev[:, h_start:h_end, w_start:w_end, :]
            if mode == 'max':
                A[:, h, w, :] = np.max(A_slice, axis=(1, 2))
            elif mode == 'avg':
                A[:, h, w, :] = np.mean(A_slice, axis=(1, 2))

    return A
