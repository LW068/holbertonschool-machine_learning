#!/usr/bin/env python3
"""save & config w json"""
import tensorflow.keras as K


def save_config(network, filename):
    """save_config"""
    # Get the configuration of the model as a JSON string
    model_config_string = network.to_json()

    # Write the JSON string to a file
    with open(filename, 'w') as f:
        f.write(model_config_string)


def load_config(filename):
    """load_config"""
    # Read the JSON string from a file
    with open(filename, 'r') as f:
        model_config_string = f.read()

    # Create a model from the JSON string
    network = K.models.model_from_json(model_config_string)

    return network
