#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def markov_chain(P, s, t=1):  # Function definition with default argument for t
    """
    Function to determine the state probabilities of a Markov chain after
    a specified number of iterations.
    """
    if not isinstance(t, int) or t < 1:  # check if t is a positive integer
        return None
    try:
        for _ in range(t):  # loop t times
            s = np.dot(s, P)  # multiply state vector s with transition matrix P
        return s  # return the final state vector
    except Exception:  # catch any exception that occurs
        return None
