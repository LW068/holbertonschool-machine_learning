#!/usr/bin/env python3
""" Gated Recurrent Unit """
import numpy as np


class GRUCell:
    """ Represents a gated recurrent unit. """
    def __init__(self, i, h, o):
        """ Initialize the GRU cell. """
        self.Wz = np.random.randn(i+h, h)  # this update gate weight
        self.Wr = np.random.randn(i+h, h)  # this resets gate weight
        self.Wh = np.random.randn(i+h, h)  # candidate hidden state weight
        self.Wy = np.random.randn(h, o)    # output weight

        self.bz = np.zeros((1, h))         # update gate bias
        self.br = np.zeros((1, h))         # reset gate bias
        self.bh = np.zeros((1, h))         # candidate hidden state bias
        self.by = np.zeros((1, o))         # output bias

    def softmax(self, x):
        """ apply softmax to an array """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """ apply sigmoid function to an array"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """ perform forward propagation for one time step """
        # concatenate h_prev and x_t for simplicity
        concat_hx = np.concatenate((h_prev, x_t), axis=1)

        # upate gate
        z_t = self.sigmoid(np.dot(concat_hx, self.Wz) + self.bz)
        # reset gate
        r_t = self.sigmoid(np.dot(concat_hx, self.Wr) + self.br)
        # canditate hidden state
        concat_hx_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_hat_t = np.tanh(np.dot(concat_hx_reset, self.Wh) + self.bh)
        # final hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_hat_t

        # outputt
        y_t = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y_t
