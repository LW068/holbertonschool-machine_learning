#!/usr/bin/env python3
""" Hidden Markov Models """

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ calculates the most likely sequence of hidden states for a hmm """
    N = Emission.shape[0]
    T = Observation.shape[0]

    dp = np.zeros((N, T))
    path = np.zeros((N, T), dtype=int)

    dp[:, 0] = Initial.T * Emission[:, Observation[0]]

    for t in range(1, T):
        for n in range(N):
            trans_prob = dp[:, t - 1] * Transition[:, n]
            max_trans_prob = np.max(trans_prob)
            dp[n, t] = max_trans_prob * Emission[n, Observation[t]]
            path[n, t] = np.argmax(trans_prob)

    P = np.max(dp[:, -1])
    state_path = []
    last_state = np.argmax(dp[:, -1])

    for t in range(T - 1, -1, -1):
        state_path.append(last_state)
        last_state = path[last_state, t]

    return state_path[::-1], P
