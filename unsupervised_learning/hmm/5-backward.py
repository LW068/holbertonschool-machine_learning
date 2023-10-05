#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ performs a backward algorithm for hmm """
    N, M = Emission.shape
    T = Observation.shape[0]
    B = np.zeros((N, T))
    B[:, -1] = 1
    for t in range(T - 2, -1, -1):
        for n in range(N):
            temp = (B[:, t + 1] * Transition[n, :] *
                    Emission[:, Observation[t + 1]])
            B[n, t] = np.sum(temp)
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
