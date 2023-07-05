#!/usr/bin/env python3
"""pool"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs pooling on images.

    Parameters:
    - images (numpy.ndarray): with shape...
    ...(m, h, w, c) containing multiple images
    - kernel_shape (tuple): of (kh, kw) containing...
    ...the kernel shape for the pooling
    - stride (tuple): of (sh, sw)
    - mode (str): indicates the type of pooling, 'max' or 'avg'

    Returns:
    - output (numpy.ndarray): containing the pooled images
    """

    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            if mode == 'max':
                output[:, i, j, :] = np.max(images[:, i*sh: i*sh + kh, j*sw:
                                                   j*sw + kw, :], axis=(1,2))
            else:  # mode == 'avg'
                output[:, i, j, :] = np.mean(images[:, i*sh: i*sh + kh, j*sw:
                                                    j*sw + kw, :], axis=(1,2))

    return output
