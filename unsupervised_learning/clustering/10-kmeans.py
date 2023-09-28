#!/usr/bin/env python3
"""K-MEANS CLUSTERING (for now this)"""
import sklearn.cluster


def kmeans(X, k):
    """K-MEANS CLUSTERING (for now this)"""
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_
    return C, labels
