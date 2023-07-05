#!/usr/bin/env python3
"""convolve_grayscale"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.

    Args:
        images (numpy.ndarray): Input grayscale images of shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel of shape (kh, kw).
        padding (str or tuple): Padding type or values. Default is 'same'.
        stride (tuple): Stride values (sh, sw). Default is (1, 1).

    Returns:
        numpy.ndarray: Convolved images of shape (m, h', w').
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Calculate padding based on 'same', 'valid' or custom
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    # Create an array with zero padding
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # Calculate the output shape
    h_prime = (h - kh + 2 * ph) // sh + 1
    w_prime = (w - kw + 2 * pw) // sw + 1

    # Initialize an array for the convolved images
    convolved_images = np.zeros((m, h_prime, w_prime))

    # Perform the convolution
    for i in range(h_prime):
        for j in range(w_prime):
            image_patch = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            convolved_images[:, i, j] = np.sum(image_patch * kernel,
                                               axis=(1, 2))

    return convolved_images
