#!/usr/bin/env python3
"""K-MEANS CLUSTERING (for now this)"""
from sklearn.cluster import KMeans


def kmeans(X, k):
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)
    C = model.cluster_centers_
    clss = model.labels_
    return C, clss
