#!/usr/bin/env python3
""" Bidirectional CEll Forward """
import numpy as np


class BidirectionalCell:
    """ Bidirectional cell class for an RNN """
    def __init__(self, i, h, o):
        """ initializes a bidirectional cell"""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

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

    def backward(self, h_next, x_t):
        """ Calculates the hidden state in
        the backward direction for one time step """
        # concatenation of the next hidden state and the current input
        concat_h_x = np.concatenate((h_next, x_t), axis=1)

        # backward direction calculation
        h_prev = np.tanh(np.dot(concat_h_x, self.Whb) + self.bhb)

        return h_prev
