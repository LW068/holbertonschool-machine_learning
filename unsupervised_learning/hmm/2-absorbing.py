#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def absorbing(P):
    """ Funtion to determine if a mrkov chain is absorbing """
    if np.any(np.diag(P) == 1): # check if any state is absorbing
        return True
    return False
