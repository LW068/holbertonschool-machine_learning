#!/usr/bin/env python3

import numpy as np


class RNNCell:
    """ represents a cell of a simple RNN """

    def __init__(self, i, h, o):
        """ initialize the RNNCell with the given parameters """
        self.Wh = np.random.randn(h + i, h)  # Weight for the...
        # ...concatenated input and hidden state
        self.Wy = np.random.randn(h, o)      # Weight for the output
        self.bh = np.zeros((1, h))           # Bias for the hidden state
        self.by = np.zeros((1, o))           # Bias for the output

    def softmax(self, x):
        """applyign softmax to an array """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ performing forward propagation for one time step"""
        # concatenating h_prev and x_t to match the dimension for Wh
        h_x_combined = np.concatenate((h_prev, x_t), axis=1)

        # computing the next hidden state
        h_next = np.tanh(h_x_combined.dot(self.Wh) + self.bh)

        # computing the output
        y_raw = h_next.dot(self.Wy) + self.by
        y = self.softmax(y_raw)

        return h_next, y
