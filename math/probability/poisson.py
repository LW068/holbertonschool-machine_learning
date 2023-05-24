#!/usr/bin/env python3
"""
Module for Poisson distribution
"""


class Poisson:
    """
    Class that represents a poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        else:
            e = 2.7182818285
            lambtha_power_k = self.lambtha ** k
            e_power_neg_lambtha = e ** (-self.lambtha)
            k_factorial = 1
            for i in range(1, k + 1):
                k_factorial *= i
            pmf = (lambtha_power_k * e_power_neg_lambtha) / k_factorial
            return

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        else:
            cdf = 0
            for i in range(k + 1):
                cdf += self.pmf(i)
            return cdf
