#!/usr/bin/env python3
"""
This module performs back propagation over a convolutional layer
of a neural network
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Perform back propagation over a convolutional layer of a neural network.

    Arguments:
    dZ: np.ndarray of shape (m, h_new, w_new, c_new) containing the partial
    derivatives with respect to the unactivated output of the conv layer.
    A_prev: np.ndarray of shape (m, h_prev, w_prev, c_prev) containing the
    output of the previous layer.
    W: np.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels for
    the convolution.
    b: np.ndarray of shape (1, 1, 1, c_new) containing the biases applied to
    the convolution.
    padding: a string that is either same or valid, indicating the type of
    padding used.
    stride: a tuple of (sh, sw) containing the strides for the convolution.

    Returns:
    The partial derivatives with respect to the previous layer (dA_prev),
    the kernels (dW), and the biases (db), respectively.
    """
    # Retrieve dimensions from A_prev's shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    # Retrieve dimensions from W's shape
    (kh, kw, _, _) = W.shape

    # Retrieve information from "stride"
    (sh, sw) = stride

    # Retrieve dimensions from dZ's shape
    (_, h_new, w_new, c_new) = dZ.shape

    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))

    # Pad A_prev and dA_prev
    if padding == "same":
        ph = kh // 2
        pw = kw // 2
        pad_width = ((0, 0), (ph, ph), (pw, pw), (0, 0))
        A_prev = np.pad(A_prev, pad_width, 'constant', constant_values=0)
        dA_prev = np.pad(dA_prev, pad_width, 'constant', constant_values=0)

    # Loop over the training examples
    for i in range(m):
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev[i]
        da_prev_pad = dA_prev[i]

        for h in range(h_new):  # loop over vertical axis of the output volume
            for w in range(w_new):  # loop over horizontal axis of the output
                for c in range(c_new):  # loop over the channels of the output

                    # Find the corners of the current "slice"
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]

                    # Update gradients for the window and the filter's params
                    da_prev_pad[vert_start:vert_end,
                                horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        # Set the ith training example's dA_prev to the unpaded da_prev_pad
        if padding == "same":
            dA_prev[i, :, :, :] = da_prev_pad[ph:-ph, pw:-pw, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, h_prev, w_prev, c_prev))

    return dA_prev, dW, db
