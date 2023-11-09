#!/usr/bin/env python3
"""
forw prop for a bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """perfomrs forward prop for bidirectional RNN"""
    t, m, i = X.shape
    _, h = h_0.shape

    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    h_next = h_0
    h_prev = h_t

    # forward direction
    for step in range(t):
        h_next = bi_cell.forward(h_next, X[step])
        H_forward[step] = h_next

    # backward direction
    for step in range(t - 1, -1, -1):
        h_prev = bi_cell.backward(h_prev, X[step])
        H_backward[step] = h_prev

    H = np.concatenate((H_forward, H_backward), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
