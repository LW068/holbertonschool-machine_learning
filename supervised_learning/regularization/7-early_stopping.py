#!/usr/bin/env python3
"""
Module that contains a function for early stopping during gradient descent
"""

def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early

    Args:
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost of the neural network
        threshold: threshold for early stopping
        patience: patience count for early stopping
        count: count of how long the threshold has not been met

    Returns:
        A tuple containing a boolean indicating whether to stop early, and the updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return True, count
    else:
        return False, count
