#!/usr/bin/env python3
"""convolve grayscale valid"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images: numpy.ndarray with shape (m, h, w) containing grayscale images
        kernel: numpy.ndarray with shape (kh, kw) containing the convolution kernel

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv_h = h - kh + 1
    conv_w = w - kw + 1
    convolved_images = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            image_patch = images[:, i:i+kh, j:j+kw]
            convolved_images[:, i, j] = np.sum(image_patch * kernel, axis=(1, 2))

    return convolved_images
