#!/usr/bin/env python3
"""
Module for task 4-moving_average.py
"""

def moving_average(data, beta):
    """
    Function that calculates the weighted moving average of a data set
    """
    V = 0
    moving_avg = []
    for i in range(len(data)):
        V = beta * V + (1 - beta) * data[i]
        moving_avg.append(V / (1 - beta ** (i + 1)))
    return moving_avg
