#!/usr/bin/env python3
"""placeholder for now"""
import sklearn.mixture

def gmm(X, k):
    """placeholder for now"""
    gmm_model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)
    
    return pi, m, S, clss, bic
