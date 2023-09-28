#!/usr/bin/env python3
"""K-MEANS CLUSTERING (for now this)"""
import sklearn.cluster
import numpy as np


def kmeans(X, k):
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)
    C = model.cluster_centers_
    clss = model.labels_
    return C, clss
