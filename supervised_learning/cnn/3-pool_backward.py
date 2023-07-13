#!/usr/bin/env python3
"""pool backward"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """pool backward, dA, A_prev, etc"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    # find the corners of the current "slice"
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == "max":
                        # Use the corners and "c" to define the...
                        # ...current slice from a_prev
                        a_prev_slice = A_prev[i, vert_start:vert_end,
                                              horiz_start:horiz_end, c]
                        # Create the mask
                        mask = a_prev_slice == np.max(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask...
                        # ...multiplied by the correct entry of dA)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                c] += np.multiply(mask, dA[i, h, w, c])

                    elif mode == "avg":
                        # Get the value a from dA
                        da = dA[i, h, w, c]
                        # Define the shape of the filter as fxf
                        shape = (kh, kw)
                        # Distribute it to get the correct slice of...
                        # ...dA_prev. i.e. Add the distributed value of da.
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                c] += distribute_value(da, shape)

    return dA_prev


def distribute_value(dz, shape):
    """Distributes the input value in the matrix of dimension shape."""
    # Retrieve dimensions from shape
    (n_H, n_W) = shape
    # Compute the value to distribute on the matrix
    average = dz / (n_H * n_W)
    # Create a matrix where every entry is the "average" value
    a = np.ones(shape) * average
    return a
