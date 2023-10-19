#!/usr/bin/env python3
"""bayesian optimization module"""
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """bayesian optimization classs"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """constructor method"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """calculates the next best sample location using
        the Expected Improvement acquisition function"""

        mu, sigma = self.gp.predict(self.X_s)
        with np.errstate(divide='warn'):
            num = (mu - self.gp.Y.max() - self.xsi)
            Z = (num / sigma)
            num2 = (mu - self.gp.Y.max() - self.xsi) * (norm.cdf(Z))
            EI = (num2 + sigma * norm.pdf(Z))

        return self.X_s[np.argmax(EI)], EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function"""
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            Y_next = self.f(X_next)
            if X_next in self.gp.X:
                break
            self.gp.update(X_next, Y_next)
        X_opt = self.gp.X[np.argmax(self.gp.Y)]
        Y_opt = self.gp.Y.max()
        return X_opt, Y_opt
