#!/usr/bin/env python3
"""convolve_channels"""
import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images with channels.

    Parameters:
    - images (numpy.ndarray): with shape (m, h, w, c) containing multiple images
    - kernel (numpy.ndarray): with shape (kh, kw, c) containing the kernel for
                              the convolution
    - padding (tuple or str): either a tuple of (ph, pw), ‘same’, or ‘valid’
    - stride (tuple): of (sh, sw)

    Returns:
    - output (numpy.ndarray): containing the convolved images
    """

    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    pad_images = np.pad(images, ((0,0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    output_h = (h - kh + 2 * ph) // sh + 1
    output_w = (w - kw + 2 * pw) // sw + 1

    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (pad_images[:, i*sh: i*sh + kh, j*sw: j*sw + kw,
                                          :] * kernel).sum(axis=(1,2,3))
            
    return output
