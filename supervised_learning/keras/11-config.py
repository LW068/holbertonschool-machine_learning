#!/usr/bin/env python3
"""save & load config"""
import tensorflow.keras as K
import json


def save_config(network, filename):
    # Get the configuration of the model
    model_config = network.get_config()

    # Save the model configuration to the file
    with open(filename, 'w') as json_file:
        json.dump(model_config, json_file)


def load_config(filename):
    # Open the file containing the model configuration
    with open(filename, 'r') as json_file:
        model_config = json.load(json_file)

    # Re-construct the model from the configuration
    network = K.models.model_from_config(model_config)

    return network
