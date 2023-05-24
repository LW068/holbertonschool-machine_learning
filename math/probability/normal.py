#!/usr/bin/env python3
"""
Module for Normal distribution
"""


class Normal:
    """
    Class that represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor
        """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = sum(data) / len(data)
                variance = sum((x - self.mean) ** 2 for x in data) / len(data)
                self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        coefficient = 1 / (self.stddev * (2 * 3.1415926536) ** 0.5)
        pdf = coefficient * 2.7182818285 ** exponent
        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = (2 / 3.1415926536 ** 0.5) * (z - z ** 3 / 3 + z ** 5 / 10 - z ** 7 / 42 + z ** 9 / 216)
        cdf = 0.5 * (1 + erf)
        return cdf
