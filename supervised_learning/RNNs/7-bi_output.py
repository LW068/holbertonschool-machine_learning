#!/usr/bin/env python3
"""
Imports
"""
import numpy as np


class BidirectionalCell:
    """
    BidirectionalCell Class
    """
    def __init__(self, i, h, o):
        """
        Constructor
        """
        self.Whf = np.random.randn(h + i, h)
        self.Whb = np.random.randn(h + i, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward
        """
        con = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(con, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Backward
        """
        con = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(con, self.Whb) + self.bhb)

        return h_prev

      def output(self, H):
        """calculates  all outputs for the RNN """
        T = H.shape[0]
        Y = []

        for t in range(T):
            y_t = np.dot(H[t], self.Wy) + self.by
            Y.append(y_t)

        Y = np.array(Y)
        Y = np.apply_along_axis(self.softmax, 1, Y)

        return Y

    @staticmethod
    def softmax(x):
        """computes the softmax values for each set of scores in x"""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
