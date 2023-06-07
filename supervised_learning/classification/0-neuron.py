#!/usr/bin/env python3
"""
Module for creating a single neuron performing binary classification
"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Class constructor
        
        Arguments:
        nx {int} -- number of input features to the neuron
        """
        
        # check if nx is not an integer
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
            
        # check if nx is less than 1
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        
        # weights vector for the neuron using random normal distribution
        self.W = np.random.randn(1, nx)
        # bias for the neuron initialized to 0
        self.b = 0
        # activated output of the neuron initialized to 0
        self.A = 0
