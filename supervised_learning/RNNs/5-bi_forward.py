#!/usr/bin/env python3
""" Bidirectional CEll Forward """
import numpy as np


class BidirectionalCell:
    """ Bidirectional cell class for an RNN """
    def __init__(self, i, h, o):
        """ initializes a bidirectional cell"""
        self.Whf = np.random.randn(i + h, h)  # weight for forw direction
        self.Whb = np.random.randn(i + h, h)  # weight for back direction
        self.Wy = np.random.randn(2 * h, o)   # weight for outputs
        self.bhf = np.zeros((1, h))           # bias for forw direction
        self.bhb = np.zeros((1, h))           # bias for back direction
        self.by = np.zeros((1, o))            # bais for outputs

    def forward(self, h_prev, x_t):
        """ alculates the hidden state in the
        forward direction for one time step """
        # concatenating the previous hidden state and the current input
        concat_h_x = np.concatenate((h_prev, x_t), axis=1)

        # debugging
        print("concat_h_x shape: {}".format(concat_h_x.shape))
        print("self.Whf shape: {}".format(self.Whf.shape))

        # aplying the tanh activation function to get the next hidden state
        h_next = np.tanh(np.dot(concat_h_x, self.Whf) + self.bhf)
        return h_next
