#!/usr/bin/env python3
"""test_model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Parameters:
    network: The network model to test.
    data: The input data to test the model with.
    labels: The correct one-hot labels of data.
    verbose: A boolean that determines if output should be...
    ...printed during the testing process.

    Returns:
    The loss and accuracy of the model with the testing data, respectively.
    """
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)
    return [loss, accuracy]  # return a list
