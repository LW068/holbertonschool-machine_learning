#!/usr/bin/env python3
"""
Module for binomial distribution
"""


class Binomial:
    """
    Class that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor
        """
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1 - variance / mean
            self.n = round(mean / p)
            self.p = mean / self.n

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        else:
            combinations = self.factorial(self.n) / (self.factorial(k) *
                    self.factorial(self.n - k))
            pmf = combinations * (self.p ** k) * ((1 - self.p) ** (self.n - k))
            return pmf

    def factorial(self, n):
        """
        Calculates the factorial of a number
        """
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.factorial(n - 1)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
