#!/usr/bin/env python3
"""
This module contains a class Neuron which defines a single neuron perf0rming
binary classification. The binary classification is done based on the 
number of input features to the neuron which is passed during the instantiation 
of the Neuron class.
"""
import numpy as np  # We're importing this module to use it f0r generating the weights

class Neuron:
    """
    Class Neuron defining a single neuron perf0rming binary classification
    """
    def __init__(self, nx):
        """
        Initialize Neuron
        Parameters:
        nx (int): number of input features to the neuron
        Instance Attributes:
        __W: The weights vector f0r the neuron initialized using a
             random normal distribution
        __b: The bias f0r the neuron initialized to 0
        __A: The activated output of the neuron initialized to 0
        """
        # Here we are checking if nx is an integer, if not we are raising a TypeError
        if type(nx) != int:
            raise TypeError('nx must be a integer')
        # Here we are checking if nx is less than 1, if so we are raising a ValueError
        if nx < 1:
            raise ValueError('nx must be positive')
        # Here we are initializing the weights (__W), bias (__b) and activated output (__A)
        self.__W = np.random.normal(size=(1, nx))  # Weights are initialized using a random normal distribution
        self.__b = 0  # Bias is initialized to 0
        self.__A = 0  # Activated output is also initialized to 0

    @property
    def W(self):
        """
        Getter f0r weights vector
        """
        # Here we are just returning the private attribute __W
        return self.__W

    @property
    def b(self):
        """
        Getter f0r bias
        """
        # Here we are just returning the private attribute __b
        return self.__b

    @property
    def A(self):
        """
        Getter f0r activated output
        """
        # Here we are just returning the private attribute __A
        return self.__A
