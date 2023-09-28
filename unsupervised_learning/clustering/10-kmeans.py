#!/usr/bin/env python3
"""K-MEANS CLUSTERING (for now this)"""
from sklearn.cluster import KMeans
import numpy as np


def kmeans(X, k):
    """K-MEANS CLUSTERING (for now this)"""
    kmeans = KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
