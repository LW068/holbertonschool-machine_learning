#!/usr/bin/env python3
"""
Module to save and load a model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model

    Arguments:
    network -- model to save
    filename -- path of the file that the model should be saved to
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model

    Arguments:
    filename -- path of the file that the model should be loaded from
    """
    return K.models.load_model(filename)
