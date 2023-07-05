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

    # Calculate the output shape based on padding type
    if padding == 'same':
        ph = int(np.ceil((h - kh + 1) / sh))
        pw = int(np.ceil((w - kw + 1) / sw))
        pad_h = max((ph - 1) * sh + kh - h, 0)
        pad_w = max((pw - 1) * sw + kw - w, 0)
    elif padding == 'valid':
        ph, pw = 0, 0
        pad_h, pad_w = 0, 0
    else:
        ph, pw = padding
        pad_h = max((h - kh + 2 * ph) % sh, 0)
        pad_w = max((w - kw + 2 * pw) % sw, 0)

    # Create an array with zero padding
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # Calculate the output shape
    h_prime = int((h + 2 * ph - kh + pad_h) / sh + 1)
    w_prime = int((w + 2 * pw - kw + pad_w) / sw + 1)

    # Initialize an array for the convolved images
    convolved_images = np.zeros((m, h_prime, w_prime))

    # Perform the convolution
    for i in range(h_prime):
        for j in range(w_prime):
            image_patch = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            convolved_images[:, i, j] = np.sum(image_patch * kernel,
                                               axis=(1, 2))

    return convolved_images
