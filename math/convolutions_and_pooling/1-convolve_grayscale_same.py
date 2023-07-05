#!/usr/bin/env python3
"""
convolve_grayscale_same
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): Input grayscale images of shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel of shape (kh, kw).

    Returns:
        numpy.ndarray: Convolved images of shape (m, h, w).
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the padding size
    pad_h = kh // 2
    pad_w = kw // 2

    # Create an array with zero padding
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           'constant')

    # Initialize an array for the convolved images
    convolved_images = np.zeros((m, h, w))

    # Perform the convolution
    for i in range(h):
        for j in range(w):
            image_patch = padded_images[:, i:i+kh, j:j+kw]
            convolved_images[:, i, j] = np.sum(image_patch * kernel,
                                               axis=(1, 2))

    return convolved_images
