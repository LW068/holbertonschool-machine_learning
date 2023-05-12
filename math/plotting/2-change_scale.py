#!/usr/bin/env python3
"""
Script for plotting the decay of C-14
"""


import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y)  # plot y against x
plt.yscale('log')  # set the y-axis to be log-scaled

# labels and title
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')

plt.show()  # display the plot
