#!/usr/bin/env python3
"""
Script for plotting a scatter plot of men's height vs weight
"""


import numpy as np
import matplotlib.pyplot as plt


mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.scatter(x, y, color='magenta')  # plot y against x as magenta points

# labels and title
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.title("Men's Height vs Weight")

plt.show()  # display the plot
