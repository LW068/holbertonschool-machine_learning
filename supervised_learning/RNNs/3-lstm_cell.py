#!/usr/bin/env python3
""" LSTM Unit"""
import numpy as np


class LSTMCell:
    """ represents an LSTM unit """

    def __init__(self, i, h, o):
        """ initializing an LSTM cell"""
        # initializing the weights and biases
        self.Wf = np.random.randn(h, i + h)
        self.Wu = np.random.randn(h, i + h)
        self.Wc = np.random.randn(h, i + h)
        self.Wo = np.random.randn(h, i + h)
        self.Wy = np.random.randn(o, h)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def sigmoid(self, x):
        """sigmoid"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """ performs forward prop """
        # concatenate h_prev and x_t
        combined = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        ft = self.sigmoid(np.dot(combined, self.Wf.T) + self.bf)

        # update gate
        ut = self.sigmoid(np.dot(combined, self.Wu.T) + self.bu)

        # intermediate cell state
        cct = np.tanh(np.dot(combined, self.Wc.T) + self.bc)

        # current cell state
        c_next = ft * c_prev + ut * cct

        # output gate
        o = self.sigmoid(np.dot(combined, self.Wo.T) + self.bo)

        # next hidden state
        h_next = o * np.tanh(c_next)

        # output (with softmax activation)
        y = self.softmax(np.dot(h_next, self.Wy.T) + self.by)

        return h_next, c_next, y
