#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ Perform the forward algorithm for a hidden markov model """
    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for n in range(N):
            temp = (F[:, t - 1] * Transition[:, n] * Emission[n, Observation[t]])
            F[n, t] = np.sum(temp)

    P = np.sum(F[:, -1])
    return P, F
