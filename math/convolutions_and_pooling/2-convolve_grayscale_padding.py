#!/usr/bin/env python3
"""convolve_grayscale_padding"""
import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Args:
        images (numpy.ndarray): Input grayscale images of shape (m, h, w).
        kernel (numpy.ndarray): Convolution kernel of shape (kh, kw).
        padding (tuple): Padding values (ph, pw) for height and width.

    Returns:
        numpy.ndarray: Convolved images of shape (m, h', w').
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Calculate the output shape
    h_prime = h + 2 * ph - kh + 1
    w_prime = w + 2 * pw - kw + 1

    # Create an array with zero padding
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # Initialize an array for the convolved images
    convolved_images = np.zeros((m, h_prime, w_prime))

    # Perform the convolution
    for i in range(h_prime):
        for j in range(w_prime):
            image_patch = padded_images[:, i:i+kh, j:j+kw]
            convolved_images[:, i, j] = np.sum(image_patch * kernel,
                                               axis=(1, 2))

    return convolved_images
