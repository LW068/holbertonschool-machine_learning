#!/usr/bin/env python3
""" Deep RNN! """
import numpy as np


class DeepRNN:
    """ represents a deep RNN network """

    def __init__(self, rnn_cells):
        """ initialize the deep RNN with a list of RNN cells """
        self.rnn_cells = rnn_cells

    def forward(self, X, h_0):
        """ perform forward propagation for the deep RNN """
        t, m, i = X.shape
        l, _, h = h_0.shape
        H = np.zeros((t + 1, l, m, h))
        H[0] = h_0
        Y = None

        for step in range(t):
            x = X[step]
            for layer, rnn_cell in enumerate(self.rnn_cells):
                h_prev = H[step, layer]
                h_next, y = rnn_cell.forward(h_prev, x)
                H[step + 1, layer] = h_next
                x = h_next

            if layer == l - 1:  # LAst layer
                if Y is None:
                    Y = y[np.newaxis, :]
                else:
                    Y = np.vstack((Y, y[np.newaxis, :]))

        return H, Y


class RNNCell:
    """ represents a single RNN cell"""

    def __init__(self, i, h, o):
        """initializign the RNN cell """
        self.Wx = np.random.randn(h, i)
        self.Wh = np.random.randn(h, h)
        self.Wy = np.random.randn(o, h)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ perform forward propagation for one time step """
        h_int = np.dot(self.Wx, x_t.T) + np.dot(self.Wh, h_prev.T) + self.bh.T
        h_next = np.tanh(h_int).T
        y = self.softmax(np.dot(h_next, self.Wy.T) + self.by)
        return h_next, y

    @staticmethod
    def softmax(x):
        """ compute softmax values for each set of scores in x """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
