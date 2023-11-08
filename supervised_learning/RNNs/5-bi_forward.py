#!/usr/bin/env python3
""" Bidirectional CEll Forward """
import numpy as np


class BidirectionalCell:
    """ Bidirectional cell class for an RNN """
    def __init__(self, i, h, o):
        """ initializes a bidirectional cell"""
        self.Whf = np.random.randn(h, i + h)  # weight for forw direction
        self.Whb = np.random.randn(h, i + h)  # weight for back direction
        self.Wy = np.random.randn(o, 2 * h)   # weight for outputs
        self.bhf = np.zeros((1, h))           # bias for forw direction
        self.bhb = np.zeros((1, h))           # bias for back direction
        self.by = np.zeros((1, o))            # bais for outputs

    def forward(self, h_prev, x_t):
        """ alculates the hidden state in the
        forward direction for one time step """
        # concatenating the previous hidden state and the current input
        concat_h_x = np.hstack((h_prev, x_t))

        # computing the dot product with the weight transpose
        whf_dot = np.dot(concat_h_x, self.Whf.T)

        # adding the bias term
        whf_dot_plus_bias = whf_dot + self.bhf

        # aplying the tanh activation function to get the next hidden state
        h_next = np.tanh(whf_dot_plus_bias)
        return h_next
