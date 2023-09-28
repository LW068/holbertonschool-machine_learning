#!/usr/bin/env python3
"""PLACWHOLDER FOR NOW"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
  """PLACWHOLDER FOR NOW"""  
  if kmax is None:
        kmax = X.shape[0]
    
    bics = []
    log_likelihoods = []
    results = []
    
    for k in range(kmin, kmax + 1):
        pi, m, S, g, l = expectation_maximization(X, k, iterations, tol, verbose)
        p = k - 1 + k * X.shape[1] + k * X.shape[1] * (X.shape[1] + 1) // 2
        bic = p * np.log(X.shape[0]) - 2 * l

        bics.append(bic)
        log_likelihoods.append(l)
        results.append((pi, m, S))

    best_k = np.argmin(bics) + kmin
    best_result = results[np.argmin(bics)]
    
    return best_k, best_result, np.array(log_likelihoods), np.array(bics)
